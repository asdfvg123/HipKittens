#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_M   = 256;
constexpr int BLOCK_N   = 256;
constexpr int BLOCK_K   = 128;
constexpr int WARPS_ROW = 2;
constexpr int WARPS_COL = 2;
constexpr int REG_M     = BLOCK_M / 2 / WARPS_ROW;  // 32
constexpr int REG_N     = BLOCK_N / 2 / WARPS_COL;  // 32

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

using ST_A = st_fp8e4m3<BLOCK_M / 2, BLOCK_K>;          // 64x128
using ST_B = st_fp8e4m3<BLOCK_N / 2, BLOCK_K>;          // 64x128
// 32x32x16 MFMA base tiles: operand 32x16, accumulator 32x32.
using RT_A = rt<fp8e4m3, REG_M, BLOCK_K, ducks::rt_layout::row, 32, 16>;   // 32x128 -> height1 width8
using RT_B = rt<fp8e4m3, REG_N, BLOCK_K, ducks::rt_layout::row, 32, 16>;   // 32x128
using RT_C = rt<float,   REG_M, REG_N,   ducks::rt_layout::col, 32, 32>;   // 32x32 -> 1x1 base

constexpr int NK = BLOCK_K / 16;  // 8 K sub-steps for 32x32x16 MFMA

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

    RT_A a[2];  // M-half 0,1
    RT_B b[2];  // N-half 0,1

    #pragma unroll 1
    for (int k = 0; k < k_iters; ++k) {
        G::load(As[0], g.a, {0, 0, block_row * WARPS_ROW,     k});
        G::load(As[1], g.a, {0, 0, block_row * WARPS_ROW + 1, k});
        G::load(Bs[0], g.b, {0, 0, block_col * WARPS_COL,     k});
        G::load(Bs[1], g.b, {0, 0, block_col * WARPS_COL + 1, k});
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        auto sa0 = subtile_inplace<REG_M, BLOCK_K>(As[0], {warp_m, 0});
        auto sa1 = subtile_inplace<REG_M, BLOCK_K>(As[1], {warp_m, 0});
        auto sb0 = subtile_inplace<REG_N, BLOCK_K>(Bs[0], {warp_n, 0});
        auto sb1 = subtile_inplace<REG_N, BLOCK_K>(Bs[1], {warp_n, 0});

        load(a[0], sa0);
        load(a[1], sa1);
        load(b[0], sb0);
        load(b[1], sb1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        // c[mi][ni] = a[mi] * b[ni]^T. Each RT_C is RT_C::height x RT_C::width base tiles (32x32).
        #pragma unroll
        for (int kk = 0; kk < NK; kk++) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 2; ni++) {
                    #pragma unroll
                    for (int n = 0; n < RT_C::height; n++) {
                        #pragma unroll
                        for (int m = 0; m < RT_C::width; m++) {
                            RT_A& aa = (mi == 0) ? a[0] : a[1];
                            RT_B& bb = (ni == 0) ? b[0] : b[1];
                            mma_ABt_base(c[mi][ni].tiles[n][m], aa.tiles[n][kk], bb.tiles[m][kk], c[mi][ni].tiles[n][m]);
                        }
                    }
                }
            }
        }
        __builtin_amdgcn_s_barrier();
    }

    // each warp output 128x128 = c[2][2], each c is 64x64; block 256 = 4 warp-tiles of 64
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
    m.doc() = "fp8 gemm 32x32x16 simple";
    py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::a, &micro_globals::b, &micro_globals::c);
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}
