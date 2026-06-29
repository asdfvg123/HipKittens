// Standalone base kernel: TN row-wise x row-wise, 1Dx2D scale, e4m3 x e4m3 -> bf16.
// Full M/N/K only (no partial), no bias/gelu/beta. Single instance for fast
// iteration + profiling (ATT/PC/PMC tools target tk_kernel/micro_tk).
// Extracted from transformer_engine/common/gemm/kittens/blockwise_fp8_gemm.cpp
// (micro_tk_1d2d, basic template).

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_M     = 128;
constexpr int BLOCK_N     = 256;
constexpr int BLOCK_K     = 128;
constexpr int REG_M       = BLOCK_M / 4;
constexpr int REG_N       = BLOCK_N / 4;
constexpr int MFMA_K      = 32;
constexpr int SCALE_BLOCK = 128;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_A  = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_B  = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_C  = gl<bf16,    -1, -1, -1, -1>;
using _gl_SA = gl<float,   -1, -1, -1, -1>;   // scale_A [K/128, M]
using _gl_SB = gl<float,   -1, -1, -1, -1>;   // scale_B [N/128, K/128]

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    _gl_SA scale_a;
    _gl_SB scale_b;
    hipStream_t stream;
    int M() const { return (int)c.rows(); }
    int N() const { return (int)c.cols(); }
    int K() const { return (int)a.cols(); }
    dim3 grid()  { return dim3(((N() + BLOCK_N - 1) / BLOCK_N) * ((M() + BLOCK_M - 1) / BLOCK_M)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 16384; }   // As only; Bs_pre + smem_sa are static __shared__
};

typedef int int32x4_lds_t __attribute__((ext_vector_type(4)));
struct __attribute__((packed)) buf_res { const void *ptr; uint32_t range; uint32_t config; };
__device__ inline int32x4_lds_t make_buf_res(const void *ptr, uint32_t size) {
    buf_res r{ptr, size, 0x00020000u};
    return __builtin_bit_cast(int32x4_lds_t, r);
}
extern "C" __device__ void
llvm_amdgcn_raw_buffer_load_lds(int32x4_lds_t rsrc, __attribute__((address_space(3))) uint32_t *lds,
                               int size, int voffset, int soffset, int offset,
                               int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

// scale_A direct global->LDS: BLOCK_M floats, wave-contiguous (64/wave).
__device__ inline void load_scale_direct_lds(float *smem_dst, const float *gbase, int tid) {
    if (tid >= 128) return;
    int32x4_lds_t rsrc = make_buf_res(gbase, 0xffffffffu);
    auto lds = (__attribute__((address_space(3))) uint32_t *)(smem_dst + (tid / 64) * 64);
    int voffset = tid * 4;
    llvm_amdgcn_raw_buffer_load_lds(rsrc, lds, 4, voffset, 0, 0, 0);
}

extern "C" __device__ float llvm_amdgcn_s_buffer_load_f32(i32x4 rsrc, int offset, int cachepolicy)
    __asm("llvm.amdgcn.s.buffer.load.f32");

__device__ inline float rtne_bias(float v) {
    uint32_t bits = __builtin_bit_cast(uint32_t, v);
    if ((bits & 0x7f800000u) == 0x7f800000u) return v;
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return __builtin_bit_cast(float, bits);
}

template <typename AccType>
__device__ inline void apply_rtne_bias(AccType &Cacc) {
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            Cacc.tiles[i][j].data[0].x = rtne_bias(Cacc.tiles[i][j].data[0].x);
            Cacc.tiles[i][j].data[0].y = rtne_bias(Cacc.tiles[i][j].data[0].y);
            Cacc.tiles[i][j].data[1].x = rtne_bias(Cacc.tiles[i][j].data[1].x);
            Cacc.tiles[i][j].data[1].y = rtne_bias(Cacc.tiles[i][j].data[1].y);
        }
    }
}

template <typename AccType>
__device__ inline void apply_block_scale_1d2d(
    AccType &Cacc, const AccType &partial, const float *sa_lds, float sb, int local_m_base) {
    const int lane = kittens::laneid();
    const int row_g = 4 * (lane / 16);
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const int m0 = local_m_base + i * 16 + row_g;
        const float s0 = sa_lds[m0 + 0] * sb;
        const float s1 = sa_lds[m0 + 1] * sb;
        const float s2 = sa_lds[m0 + 2] * sb;
        const float s3 = sa_lds[m0 + 3] * sb;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            Cacc.tiles[i][j].data[0].x += partial.tiles[i][j].data[0].x * s0;
            Cacc.tiles[i][j].data[0].y += partial.tiles[i][j].data[0].y * s1;
            Cacc.tiles[i][j].data[1].x += partial.tiles[i][j].data[1].x * s2;
            Cacc.tiles[i][j].data[1].y += partial.tiles[i][j].data[1].y * s3;
        }
    }
}

// scale_A global->reg direct: each lane buffer_loads its 4 contiguous scale rows (vmcnt pool,
// bypasses LDS + the lgkmcnt-shared ds_read that bubbles the apply multiply).
template <int HEIGHT>
__device__ inline void load_scale_global_reg(float (&sa_reg)[HEIGHT * 4], const float *sa_base,
                                             int local_m_base, uint32_t range_bytes) {
    const int lane = kittens::laneid();
    const int row_g = 4 * (lane / 16);
    i32x4 srsrc = make_srsrc((const void*)sa_base, range_bytes);
    #pragma unroll
    for (int i = 0; i < HEIGHT; i++) {
        const int m0 = local_m_base + i * 16 + row_g;
        __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, m0 * 4, 0, 0);
        *reinterpret_cast<float4*>(&sa_reg[i * 4]) = *reinterpret_cast<float4*>(&raw);
    }
}

// scale_A reg prefetch: read this wave's scale rows from LDS into regs (HEIGHT*4 floats).
template <int HEIGHT>
__device__ inline void prefetch_scale_1d2d(float (&sa_reg)[HEIGHT * 4], const float *sa_lds, int local_m_base) {
    const int lane = kittens::laneid();
    const int row_g = 4 * (lane / 16);
    #pragma unroll
    for (int i = 0; i < HEIGHT; i++) {
        const int m0 = local_m_base + i * 16 + row_g;
        *reinterpret_cast<float4*>(&sa_reg[i * 4]) =
            *reinterpret_cast<const float4*>(&sa_lds[m0]);
    }
}

template <typename AccType>
__device__ inline void apply_block_scale_1d2d_reg(
    AccType &Cacc, const AccType &partial, const float (&sa_reg)[AccType::height * 4], float sb) {
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const float s0 = sa_reg[i * 4 + 0] * sb;
        const float s1 = sa_reg[i * 4 + 1] * sb;
        const float s2 = sa_reg[i * 4 + 2] * sb;
        const float s3 = sa_reg[i * 4 + 3] * sb;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            Cacc.tiles[i][j].data[0].x += partial.tiles[i][j].data[0].x * s0;
            Cacc.tiles[i][j].data[0].y += partial.tiles[i][j].data[0].y * s1;
            Cacc.tiles[i][j].data[1].x += partial.tiles[i][j].data[1].x * s2;
            Cacc.tiles[i][j].data[1].y += partial.tiles[i][j].data[1].y * s3;
        }
    }
}

// --- B preshuffle (shuffle_weight_NK(B,16,32)): each 16N x 32K base tile = 512 fp8 flat
//     [klane(4)][n(16)][kpack(8)]; lane L's 8 fp8 at base*512 + (L/16)*128 + (L%16)*8. ---
// Copy this block's B slab (NT_B n-tiles x KT_B k-tiles) global preshuffled -> LDS flat.
template <int NT_B, int KT_B, typename GL>
__device__ inline void copy_b_slab_global_to_lds(fp8e4m3 *lds, const GL &b, int col, int kstep,
                                                 int KT_global, int tid) {
    constexpr int F4_PER_TILE = 512 / 16;            // 32 float4 per base tile
    constexpr int TOTAL_F4 = NT_B * KT_B * F4_PER_TILE;
    float4 *dst = reinterpret_cast<float4*>(lds);
    const float4 *gbase = reinterpret_cast<const float4*>(b.raw_ptr);
    #pragma unroll
    for (int c = tid; c < TOTAL_F4; c += NUM_THREADS) {
        const int tile = c / F4_PER_TILE, f4 = c % F4_PER_TILE;
        const int nt_l = tile / KT_B, kt_l = tile % KT_B;
        const int gtile = (col * NT_B + nt_l) * KT_global + (kstep * KT_B + kt_l);
        dst[c] = gbase[gtile * F4_PER_TILE + f4];
    }
}
// 2-stage staging (like base A): global preshuffled slab -> reg buffer (stage1),
// reg buffer -> flat LDS (stage2). flat LDS order == chunk index c.
template <int NT_B, int KT_B, int NF4, typename GL>
__device__ inline void load_b_slab_to_reg(float4 (&reg)[NF4], const GL &b, int col, int kstep,
                                          int KT_global, int Ndim, int Kdim, int tid) {
    constexpr int F4_PER_TILE = 512 / 16;
    // whole preshuffled B as a raw buffer; per-chunk byte offset -> buffer_load_b128 (no scratch).
    i32x4 srsrc = make_srsrc(b.raw_ptr, (uint32_t)((long)Ndim * Kdim));
    #pragma unroll
    for (int r = 0; r < NF4; r++) {
        const int c = r * NUM_THREADS + tid;
        const int tile = c / F4_PER_TILE, f4 = c % F4_PER_TILE;
        const int nt_l = tile / KT_B, kt_l = tile % KT_B;
        const int gtile = (col * NT_B + nt_l) * KT_global + (kstep * KT_B + kt_l);
        const int byte_off = (gtile * F4_PER_TILE + f4) * 16;   // float4 = 16 bytes
        __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, byte_off, 0, 0);
        reg[r] = *reinterpret_cast<float4*>(&raw);
    }
}
template <int NT_B, int KT_B, int NF4>
__device__ inline void store_b_slab_lds(fp8e4m3 *lds, const float4 (&reg)[NF4], int tid) {
    float4 *dst = reinterpret_cast<float4*>(lds);
    #pragma unroll
    for (int r = 0; r < NF4; r++) dst[r * NUM_THREADS + tid] = reg[r];
}

// LDS flat slab -> register B fragment (no swizzle address calc): contiguous ds_read_b64.
template <typename RT>
__device__ inline void load_b_frag_lds(RT &dst, const fp8e4m3 *lds, int nt_local, int kt_local, int KT_B) {
    const int lane = kittens::laneid() % kittens::WARP_THREADS;
    const int lane_off = (lane / 16) * 128 + (lane % 16) * 8;
    #pragma unroll
    for (int i = 0; i < RT::height; i++) {
        const int off = ((nt_local + i) * KT_B + kt_local) * 512 + lane_off;
        *reinterpret_cast<uint64_t*>(&dst.tiles[i][0].data[0]) =
            *reinterpret_cast<const uint64_t*>(lds + off);
    }
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st<fp8e4m3, BLOCK_M, BLOCK_K> (&As) = al.allocate<st<fp8e4m3, BLOCK_M, BLOCK_K>>();
    constexpr int NT_B = BLOCK_N / 16;   // N base tiles per block
    constexpr int KT_B = BLOCK_K / 32;   // K base tiles per k-step
    __shared__ fp8e4m3 Bs_pre[NT_B * KT_B * 512];   // preshuffled B slab, flat (no swizzle)
    __shared__ float smem_sa[2][BLOCK_M];

    rt<fp8e4m3, REG_M, MFMA_K> at[5];
    rt<fp8e4m3, REG_N, MFMA_K> bt[4];   // one per K-tile
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> C_accum[2];
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> partial[2];
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }

    const int M = (int)g.c.rows();
    const int N = (int)g.c.cols();
    const int K = (int)g.a.cols();

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    constexpr int WGM = 4;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);

    const int num_pid_m = ceil_div(M, BLOCK_M);
    const int num_pid_n = ceil_div(N, BLOCK_N);
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    const int row = pid_m;
    const int col = pid_n;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;

    const int num_k_steps = ceil_div(K, BLOCK_K);

    const float *sa_block = g.scale_a.raw_ptr + row * BLOCK_M;

    const int sb_block0 = col * (BLOCK_N / SCALE_BLOCK) + warp_col / 2;
    const float *sb_base = g.scale_b.raw_ptr + sb_block0 * num_k_steps;
    i32x4 sb_srsrc = make_srsrc((const void*)sb_base, (uint32_t)num_k_steps * 4);
    const int local_m0 = warp_row * REG_M;
    const int local_m1 = (warp_row + 2) * REG_M;
    const int tid = threadIdx.x;

    const int KT_global = K / 32;
    const int nt_local = warp_col * (REG_N / 16);

    constexpr int B_SLAB_F4_P = (NT_B * KT_B * 512 / 16 + NUM_THREADS - 1) / NUM_THREADS;
    float4 b_slab0[B_SLAB_F4_P];
    G::load(As, g.a, {0, 0, row, 0});
    load_b_slab_to_reg<NT_B, KT_B>(b_slab0, g.b, col, 0, KT_global, N, K, tid);   // slab 0 -> reg
    store_b_slab_lds<NT_B, KT_B>(Bs_pre, b_slab0, tid);                      // -> flat LDS

    // Prologue
    float sb_cur = llvm_amdgcn_s_buffer_load_f32(sb_srsrc, 0, 0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    #pragma unroll
    for (int k_step = 0; k_step < num_k_steps - 1; ++k_step) {

        constexpr int A_ELEMS_PER_THREAD = (BLOCK_M * BLOCK_K) / NUM_THREADS;
        constexpr int B_SLAB_F4 = (NT_B * KT_B * 512 / 16 + NUM_THREADS - 1) / NUM_THREADS;
        float4 a_buffer_next[A_ELEMS_PER_THREAD * sizeof(fp8e4m3) / sizeof(float4)];
        float4 b_buffer_next[B_SLAB_F4];
        zero(partial[0]); zero(partial[1]);

        float sa_reg0[REG_M / 16 * 4];
        float sa_reg1[REG_M / 16 * 4];

        // Cluster 0: B fragments from LDS slab (contiguous ds_read) + prefetch next B slab global->reg
        load_b_slab_to_reg<NT_B, KT_B>(b_buffer_next, g.b, col, k_step + 1, KT_global, N, K, tid);
        float sb_next = llvm_amdgcn_s_buffer_load_f32(sb_srsrc, (k_step + 1) * 4, 0);
        load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 0}));
        load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 0}));
        load_b_frag_lds(bt[0], Bs_pre, nt_local, 0, KT_B);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[0], bt[0], partial[0]);
        mma_ABt(partial[1], at[1], bt[0], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2
        load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, A_ELEMS_PER_THREAD, g.a, {0, 0, row, k_step + 1}, As);
        load_b_frag_lds(bt[1], Bs_pre, nt_local, 1, KT_B);
        load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 1}));
        load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 1}));
        load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[2], bt[1], partial[0]);
        mma_ABt(partial[1], at[3], bt[1], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4
        load_b_frag_lds(bt[2], Bs_pre, nt_local, 2, KT_B);
        load_b_frag_lds(bt[3], Bs_pre, nt_local, 3, KT_B);
        load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 2}));
        load(at[4], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 3}));
        load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[0], bt[2], partial[0]);
        mma_ABt(partial[1], at[1], bt[2], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6: store next A + next B slab (from reg, prefetched at Cluster 0) into LDS
        asm volatile("s_waitcnt vmcnt(0)");     // A & B global prefetch landed in regs
        asm volatile("s_waitcnt lgkmcnt(0)");
        store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        store_b_slab_lds<NT_B, KT_B>(Bs_pre, b_buffer_next, tid);
        load_scale_global_reg<REG_M / 16>(sa_reg0, sa_block + k_step * M, local_m0, (uint32_t)M * 4);
        load_scale_global_reg<REG_M / 16>(sa_reg1, sa_block + k_step * M, local_m1, (uint32_t)M * 4);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7 — MMA s3 + apply merged (apply VALU fills the barrier wait window)
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[4], bt[3], partial[0]);
        mma_ABt(partial[1], at[3], bt[3], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        asm volatile("s_waitcnt vmcnt(0)");     // scale arrived
        apply_block_scale_1d2d_reg(C_accum[0], partial[0], sa_reg0, sb_cur);
        apply_block_scale_1d2d_reg(C_accum[1], partial[1], sa_reg1, sb_cur);
        sb_cur = sb_next;
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

    }

    // Epilogue
    zero(partial[0]); zero(partial[1]);
    float sa_reg0[REG_M / 16 * 4];
    float sa_reg1[REG_M / 16 * 4];
    load_scale_global_reg<REG_M / 16>(sa_reg0, sa_block + (num_k_steps - 1) * M, local_m0, (uint32_t)M * 4);
    load_scale_global_reg<REG_M / 16>(sa_reg1, sa_block + (num_k_steps - 1) * M, local_m1, (uint32_t)M * 4);
    const int nt_local_e = warp_col * (REG_N / 16);
    load_b_frag_lds(bt[0], Bs_pre, nt_local_e, 0, KT_B);
    load_b_frag_lds(bt[1], Bs_pre, nt_local_e, 1, KT_B);
    load_b_frag_lds(bt[2], Bs_pre, nt_local_e, 2, KT_B);
    load_b_frag_lds(bt[3], Bs_pre, nt_local_e, 3, KT_B);
    __builtin_amdgcn_sched_barrier(0);
    load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 0}));
    load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[0], bt[0], partial[0]);
    mma_ABt(partial[1], at[1], bt[0], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 1}));
    load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[2], bt[1], partial[0]);
    mma_ABt(partial[1], at[3], bt[1], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 2}));
    load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 2}));
    load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 3}));
    load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[0], bt[2], partial[0]);
    mma_ABt(partial[1], at[1], bt[2], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[2], bt[3], partial[0]);
    mma_ABt(partial[1], at[3], bt[3], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    apply_block_scale_1d2d_reg(C_accum[0], partial[0], sa_reg0, sb_cur);
    apply_block_scale_1d2d_reg(C_accum[1], partial[1], sa_reg1, sb_cur);

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    apply_rtne_bias(C_accum[0]);
    apply_rtne_bias(C_accum[1]);
    store(g.c, C_accum[0], {0, 0, row * 4 + warp_row,     col * 4 + warp_col});
    store(g.c, C_accum[1], {0, 0, row * 4 + warp_row + 2, col * 4 + warp_col});
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void *)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "blockwise fp8 gemm base (TN, 1Dx2D, e4m3->bf16, full M/N/K)";
    py::bind_kernel<micro_tk>(m, "micro_tk",
        &micro_globals::a, &micro_globals::b, &micro_globals::c,
        &micro_globals::scale_a, &micro_globals::scale_b);
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &micro_globals::a, &micro_globals::b, &micro_globals::c,
        &micro_globals::scale_a, &micro_globals::scale_b);
}
