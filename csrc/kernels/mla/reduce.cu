// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <sstream>
#include <torch/python.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "aiter_hip_common.h"
#include "mla.h"

template <int32_t kSizeDV_,
          int32_t kNumHeadQ_,
          bool    kOutputLse_,
          bool    kOmitReduceFinalMap_>
struct MlaReduceKernelV1Traits
{
    static constexpr int32_t kSizeDV             = kSizeDV_;       // hidden dimension size of value/output
    static constexpr int32_t kNumHeadQ           = kNumHeadQ_;     // head count of q
    static constexpr int32_t kNumHeadQMask       = kNumHeadQ - 1;
    static constexpr int32_t kNumHeadQLog2       = __builtin_ctz(kNumHeadQ);
    static constexpr int32_t kNumWarps           = 2;
    static constexpr int32_t kNumThreads         = kNumWarps * ck_tile::get_warp_size();
    static constexpr int32_t kOccupancy          = 8;
    static constexpr int32_t kMaxVgprLocalLse    = 16;             // scratch buffer will be used with larger value
    static constexpr bool    kOutputLse          = kOutputLse_;
    // There is no reduce final map. In this case, qo len is uniform and
    // implicitly set by reduce_partial_map[1] - reduce_partial_map[0].
    static constexpr bool    kOmitReduceFinalMap = kOmitReduceFinalMap_;

    static_assert((kNumHeadQ & (kNumHeadQ - 1)) == 0, "kNumHeadQ must be power of 2!");
};

struct MlaReduceKernelV1Params
{
    const int32_t*            p_reduce_indptr;
    const MlaPartialTileInfo* p_reduce_final_map;
    const int32_t*            p_reduce_partial_map;

    void* __restrict__ p_final_lse;
    void* __restrict__ p_final_output;
    void* __restrict__ p_partial_lse;
    void* __restrict__ p_partial_output;

    int32_t stride_s_o;
    int32_t stride_h_o;
    int32_t max_splits;
    int32_t num_reduce_tile;
};

template <typename T>
CK_TILE_DEVICE T integer_divide_ceil_power2(T x, T y, T y_log2)
{
    return (x + y - 1) >> y_log2;
}

// Returns count of warps which don't contain any idle thread.
template <int32_t NumWarps, int32_t M, int32_t N>
CK_TILE_HOST_DEVICE static constexpr auto GetMaxNumWarpsForTile()
{
    static_assert(NumWarps == 1 || NumWarps == 2 || NumWarps == 4);
    constexpr int32_t ElemPerThread = (M * N) / (NumWarps * ck_tile::get_warp_size());
    if constexpr(0 < ElemPerThread)
    {
        return NumWarps;
    }
    else
    {
        return GetMaxNumWarpsForTile<NumWarps / 2, M, N>();
    }
}

// Returns vector size for given warp count for handing the specified matrix.
template <int32_t NumWarps, int32_t M, int32_t N, typename scalar_t>
CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeForTile()
{
    constexpr int32_t MaxNumWarps = GetMaxNumWarpsForTile<NumWarps, M, N>();
    constexpr int32_t ElemPerThread = (M * N) / (MaxNumWarps * ck_tile::get_warp_size());
    constexpr int32_t MaxNPerThread = 16 / sizeof(scalar_t);
    return ck_tile::min(MaxNPerThread, ElemPerThread);
}

template <typename Traits, typename scalar_t>
CK_TILE_DEVICE static constexpr auto MakeOutputTileDistribution()
{
    constexpr int32_t kVectorN     = GetVectorSizeForTile<Traits::kNumWarps, 1, Traits::kSizeDV, scalar_t>();
    constexpr int32_t kThrPerWarpN = ck_tile::get_warp_size();
    constexpr int32_t kNumWarpN    = Traits::kNumWarps;
    constexpr int32_t kNumRepeat   = ck_tile::max(1, Traits::kSizeDV / kThrPerWarpN / kNumWarpN / kVectorN);

    return ck_tile::make_static_tile_distribution(
        ck_tile::tile_distribution_encoding<
            ck_tile::sequence<>,    // no replicate
            ck_tile::tuple<ck_tile::sequence<1>,
                           ck_tile::sequence<kNumRepeat, kNumWarpN, kThrPerWarpN, kVectorN>>,
            ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2>>,
            ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<2>>,
            ck_tile::sequence<2, 1, 2>,
            ck_tile::sequence<0, 0, 3>>{});
}

template <typename Traits, typename scalar_t>
CK_TILE_DEVICE static auto MakeTileWindow(
    scalar_t* p_tile)
{
    const auto naive_view =
        ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            p_tile,
            ck_tile::make_tuple(1, Traits::kSizeDV),    // lengths
            ck_tile::make_tuple(Traits::kSizeDV, 1),    // strides
            ck_tile::number<Traits::kSizeDV>{},         // last dim alignment
            ck_tile::number<1>{});                      // last dim stride

    const auto tile_window = ck_tile::make_tile_window(
        naive_view,
        ck_tile::make_tuple(ck_tile::number<1>{},               // window size
                            ck_tile::number<Traits::kSizeDV>{}),
        {0, 0});                                                // origin

    return tile_window;
}

template <typename Traits, typename lse_t, typename out_t>
CK_TILE_DEVICE void mla_reduce_v1_impl(
    const MlaReduceKernelV1Params& params,
    const int32_t                  head_idx,
    const int32_t                  tile_idx,
    const int32_t                  reduce_tile_start,
    const int32_t                  reduce_tile_end,
    float*                         p_lds_lse_scale)
{
    // In theory, we can handle the case that #split = 1. However, it is meaningless and metadata should be in charge of
    // getting rid of this kind of scenaro.
    if (reduce_tile_start + 1 < reduce_tile_end)
    {
        const int32_t reduce_partial_map_0 = params.p_reduce_partial_map[reduce_tile_start];
        const int32_t reduce_partial_map_1 = params.p_reduce_partial_map[reduce_tile_start + 1];
        const MlaPartialTileInfo final_loc = [&]()
        {
            if constexpr (Traits::kOmitReduceFinalMap)
            {
                const int32_t qo_len = reduce_partial_map_1 - reduce_partial_map_0;
                return MlaPartialTileInfo{tile_idx * qo_len, (tile_idx + 1) * qo_len};
            }
            else
            {
                return params.p_reduce_final_map[tile_idx];
            }
        }();

        // Assuming that the layout of LSE final output is in [bs, h].
        // Thus, stride of head is 1 and stride of b/s is #heads.
        lse_t* p_final_lse_base = reinterpret_cast<lse_t*>(params.p_final_lse) + head_idx;
        const float* p_partial_lse_base =
            reinterpret_cast<const float*>(params.p_partial_lse) + head_idx;

        // Assuming that the layout of partial output is in [bs, h, d].
        // Thus, stride of hidden dim is 1, head is Traits::kSizeDV and b/s is Traits::kSizeDV * #heads
        // while the strides are 1, params.stride_h_o and params.stride_s_o for final output.
        out_t* p_final_out_base = reinterpret_cast<out_t*>(params.p_final_output) + head_idx * params.stride_h_o;
        const float* p_partial_output_base =
            reinterpret_cast<float*>(params.p_partial_output) + head_idx * Traits::kSizeDV;

        auto oaccu_window = ck_tile::make_tile_window(MakeTileWindow<Traits, const float>(nullptr),
                                                      MakeOutputTileDistribution<Traits, const float>());

        for (int32_t seq_idx = final_loc.q_start; seq_idx < final_loc.q_end; ++seq_idx)
        {
            const int32_t local_seqlen_idx = seq_idx - final_loc.q_start;
            const float* p_partial_lse_seq_base = p_partial_lse_base + local_seqlen_idx * Traits::kNumHeadQ;
            const float* p_partial_output_seq_base =
                p_partial_output_base + local_seqlen_idx * Traits::kNumHeadQ * Traits::kSizeDV;
            out_t* p_final_out = p_final_out_base + seq_idx * params.stride_s_o;

            const int64_t reduce_tile_pos_lse_start =
                params.p_reduce_partial_map[reduce_tile_start] * int64_t(Traits::kNumHeadQ);
            const int64_t reduce_tile_pos_out_start = reduce_tile_pos_lse_start * Traits::kSizeDV;

            oaccu_window.set_bottom_tensor_view_data_ptr(p_partial_output_seq_base + reduce_tile_pos_out_start);
            auto reg_out = ck_tile::load_tile(oaccu_window);
            const float lse = p_partial_lse_seq_base[reduce_tile_pos_lse_start];
            float max_lse = lse;
            float sum_e_lse = 1.0f;

            for (int32_t tile_idx = reduce_tile_start + 1; tile_idx < reduce_tile_end; ++tile_idx)
            {
                const int64_t reduce_tile_pos_lse = params.p_reduce_partial_map[tile_idx] * int64_t(Traits::kNumHeadQ);
                const int64_t reduce_tile_pos_out = reduce_tile_pos_lse * Traits::kSizeDV;

                oaccu_window.set_bottom_tensor_view_data_ptr(p_partial_output_seq_base + reduce_tile_pos_out);
                auto oaccu = ck_tile::load_tile(oaccu_window);

                const float lse = p_partial_lse_seq_base[reduce_tile_pos_lse];
                const float new_max_lse = ck_tile::max(max_lse, lse);
                const float old_scale = expf(max_lse - new_max_lse);
                const float new_scale = expf(lse - new_max_lse);

                ck_tile::sweep_tile(oaccu, [&](auto idx) {
                    reg_out(idx) = old_scale * reg_out(idx) + new_scale * oaccu(idx);
                });

                max_lse = new_max_lse;
                sum_e_lse = sum_e_lse * old_scale + new_scale;
            }

            reg_out = ck_tile::tile_elementwise_in(
                [&](const auto& elem) { return elem / sum_e_lse; },
                reg_out);

            auto dram_out = MakeTileWindow<Traits, out_t>(p_final_out);
            ck_tile::store_tile(dram_out, ck_tile::cast_tile<out_t>(reg_out));

            if constexpr(Traits::kOutputLse)
            {
                const float final_lse =
                    ((sum_e_lse == 0.f) || (sum_e_lse != sum_e_lse)) ? INFINITY : (logf(sum_e_lse) + max_lse);
                p_final_lse_base[seq_idx * Traits::kNumHeadQ] = ck_tile::type_convert<lse_t>(final_lse);
            }
        }
    }
}

template <typename Traits, typename lse_t, typename out_t>
__launch_bounds__(Traits::kNumThreads, Traits::kOccupancy)
__global__ void kn_mla_reduce_v1_ps(
    const MlaReduceKernelV1Params params)
{
    extern __shared__ float p_lds_lse_scale[];

    const int32_t last_reduce_tile = params.p_reduce_indptr[params.num_reduce_tile];
    const int32_t tot_work = Traits::kNumHeadQ * params.num_reduce_tile;
    for (int32_t work_idx = blockIdx.x; work_idx < tot_work; work_idx += gridDim.x)
    {
        const int32_t head_idx = work_idx & Traits::kNumHeadQMask;
        const int32_t tile_idx = work_idx >> Traits::kNumHeadQLog2;

        const int32_t reduce_tile_start = params.p_reduce_indptr[tile_idx];
        const int32_t reduce_tile_end = params.p_reduce_indptr[tile_idx + 1];

        if (reduce_tile_start == last_reduce_tile)
        {
            break;
        }

        mla_reduce_v1_impl<Traits, lse_t, out_t>(
            params, head_idx, tile_idx, reduce_tile_start, reduce_tile_end, p_lds_lse_scale);
    }
}

template <typename Traits, typename lse_t, typename out_t>
__launch_bounds__(Traits::kNumThreads, Traits::kOccupancy)
__global__ void kn_mla_reduce_v1(
    const MlaReduceKernelV1Params params)
{
    extern __shared__ float p_lds_lse_scale[];

    const int32_t head_idx = blockIdx.x;
    const int32_t tile_idx = blockIdx.y;

    const int32_t reduce_tile_start = params.p_reduce_indptr[tile_idx];
    const int32_t reduce_tile_end = params.p_reduce_indptr[tile_idx + 1];

    mla_reduce_v1_impl<Traits, lse_t, out_t>(
        params, head_idx, tile_idx, reduce_tile_start, reduce_tile_end, p_lds_lse_scale);
}

// NRFM: No Reduce Final Map
#define MLA_MERGE_CASE(NUM_HEAD_C, HEAD_DIM_C, OUTPUT_LSE_C, NRFM_C, NAME, ...)                             \
    constexpr int32_t NumHeads  = (NUM_HEAD_C);                                                             \
    constexpr int32_t HeadDim   = (HEAD_DIM_C);                                                             \
    constexpr bool    OutputLse = (OUTPUT_LSE_C);                                                           \
    constexpr bool    NoReduceFinalMap = (NRFM_C);                                                          \
    using Traits = MlaReduceKernelV1Traits<HeadDim, NumHeads, OutputLse, NoReduceFinalMap>;                 \
    __VA_ARGS__;

#define MLA_MERGE_CASE_IF(NUM_HEAD, NUM_HEAD_C,                                                             \
                          HEAD_DIM, HEAD_DIM_C,                                                             \
                          OUTPUT_LSE, OUTPUT_LSE_C,                                                         \
                          NRFM, NRFM_C,                                                                     \
                          NAME, ...)                                                                        \
    if (((NUM_HEAD) == (NUM_HEAD_C)) &&                                                                     \
        ((HEAD_DIM) == (HEAD_DIM_C)) &&                                                                     \
        ((OUTPUT_LSE) == (OUTPUT_LSE_C)) &&                                                                 \
        ((NRFM) == (NRFM_C)))                                                                               \
    {                                                                                                       \
        MLA_MERGE_CASE(NUM_HEAD_C, HEAD_DIM_C, OUTPUT_LSE_C, NRFM_C, NAME, __VA_ARGS__)                     \
    }

#define MLA_MERGE_CASE_EF(NUM_HEAD, NUM_HEAD_C,                                                             \
                          HEAD_DIM, HEAD_DIM_C,                                                             \
                          OUTPUT_LSE, OUTPUT_LSE_C,                                                         \
                          NRFM, NRFM_C,                                                                     \
                          NAME, ...)                                                                        \
    else if (((NUM_HEAD) == (NUM_HEAD_C)) &&                                                                \
             ((HEAD_DIM) == (HEAD_DIM_C)) &&                                                                \
             ((OUTPUT_LSE) == (OUTPUT_LSE_C)) &&                                                            \
             ((NRFM) == (NRFM_C)))                                                                          \
    {                                                                                                       \
        MLA_MERGE_CASE(NUM_HEAD_C, HEAD_DIM_C, OUTPUT_LSE_C, NRFM_C, NAME, __VA_ARGS__)                     \
    }

#define MLA_MERGE_ERROR(NUM_HEAD, HEAD_DIM, OUTPUT_LSE, NRFM, NAME)                                         \
    {                                                                                                       \
        std::stringstream ss;                                                                               \
        ss << "#heads: " << (NUM_HEAD)                                                                      \
           << ", head dimension: " << (HEAD_DIM)                                                            \
           << ", Output LSE: " << (OUTPUT_LSE)                                                              \
           << ", Has reduce final map: " << (NRFM);                                                         \
        TORCH_CHECK(false, NAME " doesn't support the specified settings: ", ss.str().c_str(), ".");        \
    }

#define MLA_MERGE_ROUTER(NUM_HEAD, HEAD_DIM, OUTPUT_LSE, NRFM, NAME, ...)                                   \
    MLA_MERGE_CASE_IF(                                                                                      \
        NUM_HEAD,   8, HEAD_DIM, 128, OUTPUT_LSE, true,  NRFM, true,  NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD,   8, HEAD_DIM, 128, OUTPUT_LSE, true,  NRFM, false, NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD,   8, HEAD_DIM, 128, OUTPUT_LSE, false, NRFM, true,  NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD,   8, HEAD_DIM, 128, OUTPUT_LSE, false, NRFM, false, NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD,  16, HEAD_DIM, 128, OUTPUT_LSE, true,  NRFM, true,  NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD,  16, HEAD_DIM, 128, OUTPUT_LSE, true,  NRFM, false, NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD,  16, HEAD_DIM, 128, OUTPUT_LSE, false, NRFM, true,  NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD,  16, HEAD_DIM, 128, OUTPUT_LSE, false, NRFM, false, NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD,  16, HEAD_DIM, 512, OUTPUT_LSE, true,  NRFM, true,  NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD,  16, HEAD_DIM, 512, OUTPUT_LSE, true,  NRFM, false, NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD,  16, HEAD_DIM, 512, OUTPUT_LSE, false, NRFM, true,  NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD,  16, HEAD_DIM, 512, OUTPUT_LSE, false, NRFM, false, NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD, 128, HEAD_DIM, 128, OUTPUT_LSE, true,  NRFM, true,  NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD, 128, HEAD_DIM, 128, OUTPUT_LSE, true,  NRFM, false, NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD, 128, HEAD_DIM, 128, OUTPUT_LSE, false, NRFM, true,  NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD, 128, HEAD_DIM, 128, OUTPUT_LSE, false, NRFM, true,  NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD, 128, HEAD_DIM, 512, OUTPUT_LSE, true,  NRFM, false, NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD, 128, HEAD_DIM, 512, OUTPUT_LSE, true,  NRFM, true,  NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD, 128, HEAD_DIM, 512, OUTPUT_LSE, false, NRFM, false, NAME, __VA_ARGS__)                    \
    MLA_MERGE_CASE_EF(                                                                                      \
        NUM_HEAD, 128, HEAD_DIM, 512, OUTPUT_LSE, false, NRFM, false, NAME, __VA_ARGS__)                    \
    else MLA_MERGE_ERROR(NUM_HEAD, HEAD_DIM, OUTPUT_LSE, NRFM, NAME);                                       \

#define DISPATCH_MLA_MERGE_KERNEL(LSE_TYPE, OUT_TYPE, NUM_HEAD, HEAD_DIM, OUTPUT_LSE, NRFM, NAME, ...)      \
    switch ((LSE_TYPE))                                                                                     \
    {                                                                                                       \
        case at::ScalarType::Float:                                                                         \
        {                                                                                                   \
            using lse_t = float;                                                                            \
            switch ((OUT_TYPE))                                                                             \
            {                                                                                               \
                case at::ScalarType::BFloat16:                                                              \
                {                                                                                           \
                    using out_t = ck_tile::bf16_t;                                                          \
                    MLA_MERGE_ROUTER(NUM_HEAD, HEAD_DIM, OUTPUT_LSE, NRFM, NAME, __VA_ARGS__)               \
                }                                                                                           \
                break;                                                                                      \
                case at::ScalarType::Half:                                                                  \
                {                                                                                           \
                    using out_t = ck_tile::fp16_t;                                                          \
                    MLA_MERGE_ROUTER(NUM_HEAD, HEAD_DIM, OUTPUT_LSE, NRFM, NAME, __VA_ARGS__)               \
                }                                                                                           \
                break;                                                                                      \
                default:                                                                                    \
                    TORCH_CHECK(false, NAME " doesn't support output type ", toString((OUT_TYPE)), ".");    \
            }                                                                                               \
        }                                                                                                   \
        break;                                                                                              \
        default:                                                                                            \
            TORCH_CHECK(false, NAME " doesn't support output LSE type ", toString((LSE_TYPE)), ".");        \
    }

template <typename Traits, typename lse_t, typename out_t>
void dispatch_mla_reduce_v1(
    const MlaReduceKernelV1Params& params,
    const int32_t                  num_cu,
    const hipStream_t&             stream)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const int32_t lds_size = params.max_splits * sizeof(float) * 2;
    if (lds_size <= (dev_prop.maxSharedMemoryPerMultiProcessor / Traits::kOccupancy))
    {
        if (Traits::kNumHeadQ * params.num_reduce_tile <= (num_cu * Traits::kOccupancy * 2))
        {
            const dim3 grid = dim3(Traits::kNumHeadQ, params.num_reduce_tile);
            kn_mla_reduce_v1<Traits, lse_t, out_t><<<grid, Traits::kNumThreads, lds_size, stream>>>(params);
        }
        else
        {
            const dim3 grid = dim3(num_cu * Traits::kOccupancy * 2);
            kn_mla_reduce_v1_ps<Traits, lse_t, out_t><<<grid, Traits::kNumThreads, lds_size, stream>>>(params);
        }
    }
    else
    {
        TORCH_CHECK(false, "kn_mla_reduce_v1: There are too much splits. We cannot handle them.");
    }
}

void mla_reduce_v1(
    const torch::Tensor&                partial_output,        // contiguous [max(reduce_partial_map)+s, h, dv]
    const torch::Tensor&                partial_lse,           // contiguous [max(reduce_partial_map)+s, h]
    const torch::Tensor&                reduce_indptr,         // contiguous [#work + 1]
    const std::optional<torch::Tensor>& reduce_final_map,      // contiguous [#work, 2]
    const torch::Tensor&                reduce_partial_map,    // contiguous [reduce_indptr[-1]]
    torch::Tensor&                      final_output,          //            [bs, h, dv]
    std::optional<torch::Tensor>&       final_lse)             // contiguous [bs, h]
{
    TORCH_CHECK((partial_output.scalar_type() == at::ScalarType::Float) &&
                (partial_lse.scalar_type() == at::ScalarType::Float),
                __func__, ": partial_out and partial_lse must be float32!");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(final_output));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const bool output_lse = final_lse.has_value();
    const bool no_reduce_final_map = (reduce_final_map.has_value() == false);
    const int32_t num_reduce_tile = reduce_indptr.size(0) - 1;
    const int32_t num_heads = partial_output.size(-2);
    const int32_t head_dim = final_output.size(-1);

    if (num_reduce_tile > 0)
    {
        MlaReduceKernelV1Params params = {};
        params.p_reduce_indptr = reduce_indptr.data_ptr<int32_t>();
        params.p_reduce_final_map =
            no_reduce_final_map ? nullptr : reinterpret_cast<const MlaPartialTileInfo*>(reduce_final_map->data_ptr());
        params.p_reduce_partial_map = reduce_partial_map.data_ptr<int32_t>();
        params.p_final_lse = output_lse ? final_lse.value().data_ptr() : nullptr;
        params.p_final_output = final_output.data_ptr();
        params.p_partial_lse = partial_lse.data_ptr();
        params.p_partial_output = partial_output.data_ptr();
        params.stride_s_o = final_output.stride(-3);
        params.stride_h_o = final_output.stride(-2);
        params.max_splits = dev_prop.multiProcessorCount;
        params.num_reduce_tile = num_reduce_tile;

        DISPATCH_MLA_MERGE_KERNEL(
            output_lse ? final_lse.value().scalar_type() : at::ScalarType::Float,
            final_output.scalar_type(),
            num_heads,
            head_dim,
            output_lse,
            no_reduce_final_map,
            "kn_mla_reduce_v1",
            dispatch_mla_reduce_v1<Traits, lse_t, out_t>(params, dev_prop.multiProcessorCount, stream)
        );
    }
}
