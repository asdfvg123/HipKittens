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
    size_t dynamic_shared_memory() { return 49152; }
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

// scale_A reg prefetch: read this wave's scale rows from LDS into regs (HEIGHT*4 floats).
template <int HEIGHT>
__device__ inline void prefetch_scale_1d2d(float (&sa_reg)[HEIGHT * 4], const float *sa_lds, int local_m_base) {
    const int lane = kittens::laneid();
    const int row_g = 4 * (lane / 16);
    #pragma unroll
    for (int i = 0; i < HEIGHT; i++) {
        const int m0 = local_m_base + i * 16 + row_g;
        sa_reg[i * 4 + 0] = sa_lds[m0 + 0];
        sa_reg[i * 4 + 1] = sa_lds[m0 + 1];
        sa_reg[i * 4 + 2] = sa_lds[m0 + 2];
        sa_reg[i * 4 + 3] = sa_lds[m0 + 3];
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

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st<fp8e4m3, BLOCK_M, BLOCK_K> (&As) = al.allocate<st<fp8e4m3, BLOCK_M, BLOCK_K>>();
    st<fp8e4m3, BLOCK_N, BLOCK_K> (&Bs) = al.allocate<st<fp8e4m3, BLOCK_N, BLOCK_K>>();
    __shared__ float smem_sa[2][BLOCK_M];

    rt<fp8e4m3, REG_M, MFMA_K> at[5];
    rt<fp8e4m3, REG_N, MFMA_K> bt[3];
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
    const int local_m0 = warp_row * REG_M;
    const int local_m1 = (warp_row + 2) * REG_M;
    const int tid = threadIdx.x;

    G::load(As, g.a, {0, 0, row, 0});
    G::load(Bs, g.b, {0, 0, col, 0});

    // Prologue
    load_scale_direct_lds(smem_sa[0], sa_block, tid);
    float sb_cur = sb_base[0];
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    #pragma unroll
    for (int k_step = 0; k_step < num_k_steps - 1; ++k_step) {

        constexpr int A_ELEMS_PER_THREAD = (BLOCK_M * BLOCK_K) / NUM_THREADS;
        constexpr int B_ELEMS_PER_THREAD = (BLOCK_N * BLOCK_K) / NUM_THREADS;
        float4 a_buffer_next[A_ELEMS_PER_THREAD * sizeof(fp8e4m3) / sizeof(float4)];
        float4 b_buffer_next[B_ELEMS_PER_THREAD * sizeof(fp8e4m3) / sizeof(float4)];

        zero(partial[0]); zero(partial[1]);

        const float *sa_cur = smem_sa[k_step & 1];

        // Cluster 0
        load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, B_ELEMS_PER_THREAD, g.b, {0, 0, col, k_step + 1}, Bs);
        load_scale_direct_lds(smem_sa[(k_step + 1) & 1], sa_block + (k_step + 1) * M, tid);
        float sb_next = sb_base[k_step + 1];
        load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 0}));
        load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 0}));
        load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 0}));
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
        load(bt[1], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 1}));
        load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 1}));
        load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 1}));
        load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 2}));
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
        load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, A_ELEMS_PER_THREAD, g.a, {0, 0, row, k_step + 1}, As);
        load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 2}));
        load(bt[2], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 3}));
        load(at[4], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 3}));
        load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[0], bt[0], partial[0]);
        mma_ABt(partial[1], at[1], bt[0], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6
        asm volatile("s_waitcnt lgkmcnt(0)");
        store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7 — MMA s3
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[4], bt[2], partial[0]);
        mma_ABt(partial[1], at[3], bt[2], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 8
        apply_block_scale_1d2d(C_accum[0], partial[0], sa_cur, sb_cur, local_m0);
        apply_block_scale_1d2d(C_accum[1], partial[1], sa_cur, sb_cur, local_m1);
        sb_cur = sb_next;
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

    }

    // Epilogue
    zero(partial[0]); zero(partial[1]);
    const float *sa_cur = smem_sa[(num_k_steps - 1) & 1];
    __builtin_amdgcn_sched_barrier(0);
    load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 0}));
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

    load(bt[1], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 1}));
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

    load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 2}));
    load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 2}));
    load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 2}));
    load(bt[1], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 3}));
    load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 3}));
    load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[0], bt[0], partial[0]);
    mma_ABt(partial[1], at[1], bt[0], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[2], bt[1], partial[0]);
    mma_ABt(partial[1], at[3], bt[1], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    apply_block_scale_1d2d(C_accum[0], partial[0], sa_cur, sb_cur, local_m0);
    apply_block_scale_1d2d(C_accum[1], partial[1], sa_cur, sb_cur, local_m1);

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
