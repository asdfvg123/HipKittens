#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_M   = 256;
constexpr int BLOCK_N   = 256;
constexpr int BLOCK_K   = 128;
constexpr int WARPS_ROW = 2;
constexpr int WARPS_COL = 2;
constexpr int REG_M     = BLOCK_M / 2 / WARPS_ROW;  // 64
constexpr int REG_N     = BLOCK_N / 2 / WARPS_COL;  // 64

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#ifndef GEMM_SZ
#define GEMM_SZ 8192
#endif
constexpr int M = GEMM_SZ;
constexpr int K = GEMM_SZ;
constexpr int N = GEMM_SZ;

using _gl_A = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_B = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_C = gl<bf16,    -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    hipStream_t stream;
    dim3 grid()  { return dim3((N / BLOCK_N) * (M / BLOCK_M)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 65536; }
};

using ST_A = st_fp8e4m3<BLOCK_M / 2, BLOCK_K>;          // 128x128
using ST_B = st_fp8e4m3<BLOCK_N / 2, BLOCK_K>;          // 128x128
using RT_A = rt_fp8e4m3<REG_M, BLOCK_K>;                // 64x128 (K-whole, width=4)
using RT_B = rt_fp8e4m3<REG_N, BLOCK_K>;                // 64x128
using RT_C = rt_fl<REG_M, REG_N, ducks::rt_layout::col>; // 64x64

constexpr int NK = BLOCK_K / 32;  // 4 K sub-steps for 16x16x32 MFMA

// Load one (i,j) base-tile column of an fp8 fragment from shared -> register (single ds_read_b64).
template <typename RT, typename ST>
__device__ inline void load_frag_one(RT& dst, const ST& src, int slot) {
    const int laneid = kittens::laneid() % kittens::WARP_THREADS;
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    const int row_offset = laneid % 16;
    const int col_offset = 8 * (laneid / 16);
    const int j = slot / RT::height;
    const int i = slot % RT::height;
    const int col = j * dst.tile_size_col + col_offset;
    const int off = i * ST::underlying_cols * kittens::TILE_ROW_DIM<typename ST::dtype> * (int)sizeof(typename ST::dtype);
    uint32_t addr = src.idx(src_ptr, {row_offset, col}) + off;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
        : "=v"(*reinterpret_cast<uint64_t*>(&dst.tiles[i][j].data[0]))
        : "v"(addr) : "memory");
}

// Load one 16B chunk (buffer_load_dwordx4) global -> register buffer. bi in [0, BUF4).
template <int axis = 2, typename ST, typename GL, ducks::coord::tile COORD = coord<ST>>
__device__ inline void load_global_one(float4* reg_buffer, int bi, const GL& src, const COORD& idx, const ST& dst_template) {
    using T = typename ST::dtype;
    constexpr int N_THREADS = NUM_THREADS;
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);   // 16
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;    // 8
    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* base_ptr = (T*)&src[unit_coord];
    const int laneid = threadIdx.x % N_THREADS;
    const int total_bytes = row_stride * ST::rows * sizeof(T);
    i32x4 srsrc = make_srsrc(base_ptr, total_bytes, row_stride * (int)sizeof(T));

    const int chunk_idx = bi * N_THREADS + laneid;
    const int row = chunk_idx / memcpy_per_row;
    const int col = (chunk_idx % memcpy_per_row) * elem_per_memcpy;
    const int byte_offset = (row * row_stride + col) * sizeof(T);
    __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, byte_offset, 0, 0);
    reg_buffer[bi] = *reinterpret_cast<float4*>(&raw);
}

// Store one 16B chunk (reg buffer -> LDS, 2 ds_write_b64). bi in [0, BUF4).
template <typename ST>
__device__ inline void store_one(ST& dst, const float4* reg_buffer, int bi) {
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);   // 16
    constexpr int elem_per_half = sizeof(float2) / sizeof(T);     // 8
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;    // 8
    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % NUM_THREADS;
    const int load_idx = bi * NUM_THREADS + laneid;
    const int row = load_idx / memcpy_per_row;
    const int col = (load_idx % memcpy_per_row) * elem_per_memcpy;
    const float4& v = reg_buffer[bi];
    store_shared_vec(dst.idx(dst_ptr, {row, col}), {v.x, v.y});
    store_shared_vec(dst.idx(dst_ptr, {row, col + elem_per_half}), {v.z, v.w});
}

// One output c (64x64): NK*16 base MFMA, fenced; NL load_next calls spread evenly.
template <int NL, typename FNEXT>
__device__ inline void interleaved_block(RT_C& c, const RT_A& a, const RT_B& b, FNEXT load_next) {
    constexpr int TOTAL = NK * RT_C::height * RT_C::width;
    #pragma unroll
    for (int kk = 0; kk < NK; kk++) {
        #pragma unroll
        for (int n = 0; n < RT_C::height; n++) {
            #pragma unroll
            for (int m = 0; m < RT_C::width; m++) {
                const int step = kk * (RT_C::height * RT_C::width) + n * RT_C::width + m;
                __builtin_amdgcn_sched_barrier(0);
                mma_ABt_base(c.tiles[n][m], a.tiles[n][kk], b.tiles[m][kk], c.tiles[n][m]);
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for (int jj = 0; jj < NL; jj++)
                    if (step == jj * TOTAL / NL) load_next(jj);
                __builtin_amdgcn_sched_barrier(0);
            }
        }
    }
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

    RT_C c[2][2];
    zero(c[0][0]); zero(c[0][1]); zero(c[1][0]); zero(c[1][1]);

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    constexpr int WGM = 4;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM * WGM);
    const int num_pid_m = ceil_div(M, BLOCK_M);
    const int num_pid_n = ceil_div(N, BLOCK_N);
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    const int block_row = pid_m;
    const int block_col = pid_n;

    const int warp_m = warpid() / WARPS_COL;
    const int warp_n = warpid() % WARPS_COL;

    const int k_iters = K / BLOCK_K;
    constexpr int BUF = (BLOCK_M / 2 * BLOCK_K) / NUM_THREADS;
    constexpr int BUF4 = BUF * sizeof(fp8e4m3) / sizeof(float4);

    RT_A a[2];  // M-half 0,1
    RT_B b[2];  // N-half 0,1

    constexpr int NGLOB = 2 * BUF4;
    constexpr int NFRAG = RT_A::height * RT_A::width;
    // ab ping-pong in registers: prev (to store this iter) / cur (HBM-load this iter).
    float4 ab[2][4][BUF4];

    G::load(As[0], g.a, {0, 0, block_row * WARPS_ROW,     0});
    G::load(As[1], g.a, {0, 0, block_row * WARPS_ROW + 1, 0});
    G::load(Bs[0], g.b, {0, 0, block_col * WARPS_COL,     0});
    G::load(Bs[1], g.b, {0, 0, block_col * WARPS_COL + 1, 0});
    // prologue: HBM-load iter-1 into ab[0]
    if (k_iters > 1) {
        #pragma unroll
        for (int s = 0; s < BUF4; s++) {
            load_global_one(ab[0][0], s, g.a, {0, 0, block_row * WARPS_ROW,     1}, As[0]);
            load_global_one(ab[0][1], s, g.a, {0, 0, block_row * WARPS_ROW + 1, 1}, As[1]);
            load_global_one(ab[0][2], s, g.b, {0, 0, block_col * WARPS_COL,     1}, Bs[0]);
            load_global_one(ab[0][3], s, g.b, {0, 0, block_col * WARPS_COL + 1, 1}, Bs[1]);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    auto sa0 = subtile_inplace<REG_M, BLOCK_K>(As[0], {warp_m, 0});
    auto sa1 = subtile_inplace<REG_M, BLOCK_K>(As[1], {warp_m, 0});
    auto sb0 = subtile_inplace<REG_N, BLOCK_K>(Bs[0], {warp_n, 0});
    auto sb1 = subtile_inplace<REG_N, BLOCK_K>(Bs[1], {warp_n, 0});

    load(a[0], sa0);
    load(b[0], sb0);
    asm volatile("s_waitcnt lgkmcnt(0)");

    #pragma unroll 1
    for (int k = 0; k < k_iters; ++k) {
        const int pp = k & 1, np = (k + 1) & 1;   // ab ping-pong
        bool has_next = (k + 1 < k_iters);     // ab[pp] holds iter k+1 data (HBM-loaded last iter)
        bool ld_next  = (k + 2 < k_iters);     // HBM-load iter k+2 into ab[np]

        // c00 = a0*b0, prefetch b1 frag
        interleaved_block<NFRAG>(c[0][0], a[0], b[0], [&](int s){ load_frag_one(b[1], sb1, s); });
        asm volatile("s_waitcnt lgkmcnt(0)");
        // c01 = a0*b1, prefetch a1 frag  (after this, As/Bs LDS no longer read this iter)
        interleaved_block<NFRAG>(c[0][1], a[0], b[1], [&](int s){ load_frag_one(a[1], sa1, s); });
        asm volatile("s_waitcnt lgkmcnt(0)");
        if (has_next) asm volatile("s_waitcnt vmcnt(0)");  // ab[pp] HBM arrived
        __builtin_amdgcn_s_barrier();   // all warps done reading As/Bs this iter

        // c10: store ab[pp]->As (next-iter data) + HBM-load iter k+2 A into ab[np], both interleaved.
        interleaved_block<NGLOB>(c[1][0], a[1], b[0], [&](int s){
            if (has_next && s < 2 * BUF4) {
                if (s < BUF4) store_one(As[0], ab[pp][0], s);
                else          store_one(As[1], ab[pp][1], s - BUF4);
            }
            if (ld_next && s < 2 * BUF4) {
                if (s < BUF4) load_global_one(ab[np][0], s,        g.a, {0, 0, block_row * WARPS_ROW,     k + 2}, As[0]);
                else          load_global_one(ab[np][1], s - BUF4, g.a, {0, 0, block_row * WARPS_ROW + 1, k + 2}, As[1]);
            }
        });
        // c11: store ab[pp]->Bs + HBM-load iter k+2 B into ab[np].
        interleaved_block<NGLOB>(c[1][1], a[1], b[1], [&](int s){
            if (has_next && s < 2 * BUF4) {
                if (s < BUF4) store_one(Bs[0], ab[pp][2], s);
                else          store_one(Bs[1], ab[pp][3], s - BUF4);
            }
            if (ld_next && s < 2 * BUF4) {
                if (s < BUF4) load_global_one(ab[np][2], s,        g.b, {0, 0, block_col * WARPS_COL,     k + 2}, Bs[0]);
                else          load_global_one(ab[np][3], s - BUF4, g.b, {0, 0, block_col * WARPS_COL + 1, k + 2}, Bs[1]);
            }
        });

        if (has_next) {
            asm volatile("s_waitcnt lgkmcnt(0)");   // As/Bs store visible
            __builtin_amdgcn_s_barrier();
            load(a[0], sa0);
            load(b[0], sb0);
            asm volatile("s_waitcnt lgkmcnt(0)");
        }
    }

    store(g.c, c[0][0], {0, 0, block_row * 4 + warp_m,     block_col * 4 + warp_n});
    store(g.c, c[0][1], {0, 0, block_row * 4 + warp_m,     block_col * 4 + warp_n + 2});
    store(g.c, c[1][0], {0, 0, block_row * 4 + warp_m + 2, block_col * 4 + warp_n});
    store(g.c, c[1][1], {0, 0, block_row * 4 + warp_m + 2, block_col * 4 + warp_n + 2});
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "fp8 gemm 4-wave 16x16x32 interleave";
    py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::a, &micro_globals::b, &micro_globals::c);
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}
