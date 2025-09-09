#include <metal_stdlib>
using namespace metal;

#ifndef Bm
#define Bm 8
#endif
#ifndef Bn
#define Bn 8
#endif
#ifndef Bk
#define Bk 8
#endif
#ifndef Sg_Bm
#define Sg_Bm 8
#endif
#ifndef Sg_Bn
#define Sg_Bn 8
#endif
#ifndef THREADS_PER_TG
#define THREADS_PER_TG 32
#endif
#ifndef RHS_TRANSPOSE
#define RHS_TRANSPOSE 0
#endif

struct MatmulParams {
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t add;
  uint32_t rhs_transpose = 0;
};

kernel void
matmul_f32_bf16w(constant MatmulParams &params [[buffer(0)]],
                 const device float *lhs [[buffer(1)]],
                 const device bfloat *rhs [[buffer(2)]],
                 const device bfloat *__restrict__ bias [[buffer(3)]],
                 device float *out [[buffer(4)]],
                 uint sg_id [[simdgroup_index_in_threadgroup]],
                 uint3 threads_per_tg [[threads_per_threadgroup]],
                 uint sg_count_per_tg [[dispatch_simdgroups_per_threadgroup]],
                 uint3 gid [[thread_position_in_grid]],
                 uint3 tg_id [[threadgroup_position_in_grid]],
                 uint3 local_tid [[thread_position_in_threadgroup]]) {
  // const uint M = params.M;
  const uint K = params.K;
  const uint N = params.N;

  // Get row and col tg.
  const uint row_tg = tg_id.y;
  const uint col_tg = tg_id.x;
  // Get row and col local tid.
  const uint row_tg_offset = row_tg * Bm;
  const uint col_tg_offset = col_tg * Bn;

  const uint sg_col_count = Bn / Sg_Bn;
  const uint row_sg = sg_id / sg_col_count;
  const uint col_sg = sg_id % sg_col_count;

  const uint row_sg_offset = row_sg * Sg_Bm;
  const uint col_sg_offset = col_sg * Sg_Bn;
  constexpr uint temp_result_size = (Sg_Bm / 8) * (Sg_Bn / 8);
  // Create an array of simdgroup_float8x8 to hold temp results.
  simdgroup_float8x8 OutTiles[temp_result_size];
#pragma clang loop unroll(full)
  for (uint i = 0; i < temp_result_size; i++) {
    OutTiles[i] =
        make_filled_simdgroup_matrix<float, 8, 8>(static_cast<float>(0.0));
  }

  for (uint k_offset = 0; k_offset < K; k_offset += Bk) {
#pragma clang loop unroll(full)
    for (uint k = 0; k < Bk; k += 8) {
#pragma clang loop unroll(full)
      for (uint m_subtile_ = 0; m_subtile_ < Sg_Bm; m_subtile_ += 8) {
        // const uint m_subtile = row_sg_offset + m_subtile_;
        // const uint row_index_in_out_tile = (m_subtile - row_sg_offset) / 8;
        const uint row_index_in_out_tile = m_subtile_ / 8;
        simdgroup_float8x8 LHStile;
        const uint k_id = k + k_offset;
        const uint row_offset = row_tg_offset + row_sg_offset + m_subtile_;
        simdgroup_load(LHStile, lhs, K, ulong2(k_id, row_offset));
        simdgroup_bfloat8x8 RHStile;
#pragma clang loop unroll(full)
        for (uint n_subtile_ = 0; n_subtile_ < Sg_Bn; n_subtile_ += 8) {
          // const uint n_subtile = col_sg_offset + n_subtile_;
          // const uint col_index_in_out_tile = (n_subtile - col_sg_offset) / 8;
          const uint col_index_in_out_tile = n_subtile_ / 8;
          const uint current_index_out_tile =
              row_index_in_out_tile * (Sg_Bn / 8) + col_index_in_out_tile;

          const uint col_offset = col_tg_offset + col_sg_offset + n_subtile_;
          // Gets optimized away by the compiler. But need RHS_TRANSPOSE to be a
          // compile-time constant.
          if (RHS_TRANSPOSE) {
            simdgroup_load(RHStile, rhs, K, ulong2(k_id, col_offset), true);
          } else {
            simdgroup_load(RHStile, rhs, N, ulong2(col_offset, k_id));
          }
          simdgroup_multiply_accumulate(OutTiles[current_index_out_tile],
                                        LHStile, RHStile,
                                        OutTiles[current_index_out_tile]);
        }
      }
    }
  }
  // Epilogue.
  threadgroup float scratch[Bm * Bn];
#pragma clang loop unroll(full)
  for (uint n_subtile_ = 0; n_subtile_ < Sg_Bn; n_subtile_ += 8) {
    const uint col_index_in_out_tile = n_subtile_ / 8;
    const uint local_col_offset = col_sg_offset + n_subtile_;
#pragma clang loop unroll(full)
    for (uint m_subtile_ = 0; m_subtile_ < Sg_Bm; m_subtile_ += 8) {
      const uint row_index_in_out_tile = m_subtile_ / 8;
      const uint local_row_offset = row_sg_offset + m_subtile_;
      const uint current_index_out_tile =
          row_index_in_out_tile * (Sg_Bn / 8) + col_index_in_out_tile;
      simdgroup_store(OutTiles[current_index_out_tile], scratch, Bn,
                      ulong2(local_col_offset, local_row_offset));
    }
  }
  threadgroup float bias_tile[Bn];
  const uint thread_count_per_tg =
      threads_per_tg.x * threads_per_tg.y * threads_per_tg.z;
  // TODO(ibahmed): vectorize these loads an maybe unroll the loop.
  for (uint c_local = local_tid.x; c_local < Bn;
       c_local += thread_count_per_tg) {
    const uint c_global = col_tg_offset + c_local;
    bias_tile[c_local] =
        (c_global < N) ? static_cast<float>(bias[c_global]) : 0.0f;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  // TODO(ibahmed): vectorize these stores and maybe unroll the loop.
  for (uint idx = local_tid.x; idx < Bm * Bn; idx += thread_count_per_tg) {
    const uint r = idx / Bn;
    const uint c = idx % Bn;

    const uint out_row = row_tg_offset + r;
    const uint out_col = col_tg_offset + c;

    if (out_row < params.M && out_col < N) {
      float acc = scratch[idx] + bias_tile[c];
      if (params.add) {
        acc += out[out_row * N + out_col];
      }
      out[out_row * N + out_col] = acc;
    }
  }
}