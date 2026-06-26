
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_M    = 128;
constexpr int BLOCK_N    = 256;
constexpr int K_STEP     = 128;             // fp8 K-slab per shared load
constexpr int REG_M      = BLOCK_M / 4;     // 32 rows of C per wave (M)
constexpr int REG_N      = BLOCK_N / 4;     // 64 cols of C per wave (N)
constexpr int DOT_SLICE  = 32;              // fp8 mma base-tile K (16x16x32)

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
    int Mdim() const { return (int)c.rows(); }
    int Ndim() const { return (int)c.cols(); }
    int Kdim() const { return (int)a.cols(); }
    dim3 grid()  { return dim3(((Ndim() + BLOCK_N - 1) / BLOCK_N) * ((Mdim() + BLOCK_M - 1) / BLOCK_M)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 49152; }  // As+Bs; smem_sa is static __shared__
};

__device__ inline float rtne_bias(float v) {
    uint32_t bits = __builtin_bit_cast(uint32_t, v);
    if ((bits & 0x7f800000u) == 0x7f800000u) return v;
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return __builtin_bit_cast(float, bits);
}

template <typename CT>
__device__ inline void apply_rtne_bias(CT &Cacc) {
    #pragma unroll
    for (int i = 0; i < CT::height; i++) {
        #pragma unroll
        for (int j = 0; j < CT::width; j++) {
            Cacc.tiles[i][j].data[0].x = rtne_bias(Cacc.tiles[i][j].data[0].x);
            Cacc.tiles[i][j].data[0].y = rtne_bias(Cacc.tiles[i][j].data[0].y);
            Cacc.tiles[i][j].data[1].x = rtne_bias(Cacc.tiles[i][j].data[1].x);
            Cacc.tiles[i][j].data[1].y = rtne_bias(Cacc.tiles[i][j].data[1].y);
        }
    }
}

template <typename CT, typename PT>
__device__ inline void apply_block_scale(
    CT &Cacc, const PT &partial, const float *sa_lds, float sb,
    int local_m_base) {
    const int lane = kittens::laneid();
    const int row_g = 4 * (lane / 16);
    #pragma unroll
    for (int i = 0; i < CT::height; i++) {
        // scale_A depends on the M-row only (i), not on j → compute once per i.
        // sa_lds points at this block's 128 staged scale_A floats (current slab).
        const int m0 = local_m_base + i * 16 + row_g;
        const float s0 = sa_lds[m0 + 0] * sb;
        const float s1 = sa_lds[m0 + 1] * sb;
        const float s2 = sa_lds[m0 + 2] * sb;
        const float s3 = sa_lds[m0 + 3] * sb;
        #pragma unroll
        for (int j = 0; j < CT::width; j++) {
            Cacc.tiles[i][j].data[0].x += partial.tiles[i][j].data[0].x * s0;
            Cacc.tiles[i][j].data[0].y += partial.tiles[i][j].data[0].y * s1;
            Cacc.tiles[i][j].data[1].x += partial.tiles[i][j].data[1].x * s2;
            Cacc.tiles[i][j].data[1].y += partial.tiles[i][j].data[1].y * s3;
        }
    }
}

// Bounds-checked col-layout store: mirrors HK's global_to_register col store but
// skips any global M-row >= Mdim, so the last M-block (M % BLOCK_M != 0) writes
// only valid rows. C is [Mdim, Ndim] row-major (row_stride = Ndim). The tile
// coord {Rtile, Ctile} matches HK's store(g.c, ..., {0,0,Rtile,Ctile}): unit_coord
// scales by the *whole* register-tile size (CT::rows = REG_M, CT::cols = REG_N),
// then each subtile (i,j) is offset by TILE_ROW_DIM/TILE_COL_DIM (=16).
template <typename CT>
__device__ inline void store_masked_M(bf16 *c_ptr, const CT &Cacc,
                                      int Rtile, int Ctile, int Mdim, int Ndim) {
    const int lane = kittens::laneid();
    const int m_base = Rtile * CT::rows + 4 * (lane / 16);
    const int n_base = Ctile * CT::cols + (lane % 16);
    #pragma unroll
    for (int i = 0; i < CT::height; i++) {
        const int m0 = m_base + i * 16;
        #pragma unroll
        for (int j = 0; j < CT::width; j++) {
            const int col = n_base + j * 16;
            if (col >= Ndim) continue;            // skip N-cols past the boundary
            const float v0 = Cacc.tiles[i][j].data[0].x;
            const float v1 = Cacc.tiles[i][j].data[0].y;
            const float v2 = Cacc.tiles[i][j].data[1].x;
            const float v3 = Cacc.tiles[i][j].data[1].y;
            if (m0 + 0 < Mdim) c_ptr[(m0 + 0) * Ndim + col] = base_types::convertor<bf16, float>::convert(v0);
            if (m0 + 1 < Mdim) c_ptr[(m0 + 1) * Ndim + col] = base_types::convertor<bf16, float>::convert(v1);
            if (m0 + 2 < Mdim) c_ptr[(m0 + 2) * Ndim + col] = base_types::convertor<bf16, float>::convert(v2);
            if (m0 + 3 < Mdim) c_ptr[(m0 + 3) * Ndim + col] = base_types::convertor<bf16, float>::convert(v3);
        }
    }
}

// Group (512-thread) global->LDS load of a tile [ST::rows, K_STEP] with a row
// bound: rows whose global index (row_blk*ST::rows + r) >= row_dim are filled
// with 0 instead of reading out of bounds. Mirrors HK's warp global_to_shared
// load but guards the global boundary (HK only checks r < dst.rows = tile
// height). Used for the partial last A-block (M % BLOCK_M != 0, row_dim=Mdim) and
// the partial last B-block (N % BLOCK_N != 0, row_dim=Ndim); full blocks keep the
// fast HK path. (B's tile rows are the N dimension since B is [N, K].)
template <typename ST, typename GL>
__device__ inline void load_tile_masked_rows(ST &dst, const GL &src, int row_blk,
                                             int k_blk, int row_dim) {
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);          // fp8: 16
    constexpr int elem_per_half_memcpy = sizeof(float2) / sizeof(T);     // fp8: 8
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total = (ST::rows * ST::cols) / elem_per_memcpy;       // vec chunks
    const int row_stride = src.template stride<2>();
    const int row_base = row_blk * ST::rows;
    kittens::coord<> uc = kittens::coord<ST>(0, 0, row_blk, k_blk).template unit_coord<2, 3>();
    T *src_ptr = (T *)&src[uc];
    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int tid = threadIdx.x;
    #pragma unroll
    for (int idx = tid; idx < total; idx += NUM_THREADS) {
        const int row = idx / memcpy_per_row;
        const int col = (idx % memcpy_per_row) * elem_per_memcpy;
        float4 v = {0.f, 0.f, 0.f, 0.f};
        if (row_base + row < row_dim) {
            v = load_global_vec4_async((float4 *)(src_ptr + (row * row_stride + col)));
            asm volatile("s_waitcnt vmcnt(0)");
        }
        store_shared_vec(dst.idx(dst_ptr, {row, col}), {v.x, v.y});
        store_shared_vec(dst.idx(dst_ptr, {row, col + elem_per_half_memcpy}), {v.z, v.w});
    }
}

template <bool IS_PARTIAL_M, bool IS_PARTIAL_N>
__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_fp8e4m3<BLOCK_M, K_STEP> (&As) = al.allocate<st_fp8e4m3<BLOCK_M, K_STEP>>();
    st_fp8e4m3<BLOCK_N, K_STEP> (&Bs) = al.allocate<st_fp8e4m3<BLOCK_N, K_STEP>>();
    // scale_A LDS double-buffer (static __shared__ so the compiler keeps the LDS
    // address space → ds_read in apply, not a generic flat_load_dwordx4 + vmcnt wait).
    __shared__ float smem_sa[2][BLOCK_M];

    rt_fp8e4m3<REG_M, DOT_SLICE> at[8];
    rt_fp8e4m3<REG_N, DOT_SLICE> bt[8];
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> C_accum[2];
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> partial[2];
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }

    const int Mdim = (int)g.c.rows();
    const int Ndim = (int)g.c.cols();
    const int Kdim = (int)g.a.cols();

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    constexpr int WGM = 4;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
    const int num_pid_m = ceil_div(Mdim, BLOCK_M);
    const int num_pid_n = ceil_div(Ndim, BLOCK_N);
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    const int row = pid_m;
    const int col = pid_n;
    // Last M-block may be partial when M % BLOCK_M != 0; it needs bounds-checked
    // A loads (full blocks use the faster unguarded HK path). Gated on the
    // template param so the aligned instance (IS_PARTIAL_M=false) compiles none of
    // the masked paths -> identical VGPR/perf to the original kernel.
    const bool is_last_m = IS_PARTIAL_M && (row * BLOCK_M + BLOCK_M > Mdim);
    const bool is_last_n = IS_PARTIAL_N && (col * BLOCK_N + BLOCK_N > Ndim);

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;

    const int num_tiles = Kdim / K_STEP;

    // scale_A is [K/128, M]; this block owns the 128 M-rows [row*BLOCK_M ..].
    // slab `tile` lives at sa_block + tile*Mdim. We stage those 128 floats into
    // smem_sa so apply reads LDS (no per-slab, per-lane global load).
    const float *sa_block = g.scale_a.raw_ptr + row * BLOCK_M;
    // scale_B is [ceil(N/128), K/128]. BLOCK_N=256 spans two 128 N-scale-blocks;
    // warp_col {0,1}->first, {2,3}->second. For a partial last N-block a warp's
    // scale-block can exceed the valid count -> guard the read (that warp_col's
    // output columns are all >= Ndim and get skipped by the masked store anyway).
    const int n_scale_blocks = (Ndim + 127) / 128;
    const int sb_block0 = col * (BLOCK_N / 128) + warp_col / 2;
    const bool sb_valid = (!is_last_n) || (sb_block0 < n_scale_blocks);
    const float *sb_base = g.scale_b.raw_ptr + (sb_valid ? sb_block0 : 0) * num_tiles;
    const int local_m0 = warp_row * REG_M;            // C_accum[0] M base in block
    const int local_m1 = (warp_row + 2) * REG_M;      // C_accum[1] M base in block
    const int tid = threadIdx.x;

    // Load first tile into shared memory
    if (is_last_m) load_tile_masked_rows(As, g.a, row, 0, Mdim);
    else           G::load(As, g.a, {0, 0, row, 0});
    if (is_last_n) load_tile_masked_rows(Bs, g.b, col, 0, Ndim);
    else           G::load(Bs, g.b, {0, 0, col, 0});
    // Stage slab-0 scale_A into smem_sa[0] and prefetch slab-0 scale_B.
    // Guard the global M-row only for the partial last M-block so aligned blocks
    // keep the original (unguarded) fast path.
    const bool m_in_range = (tid < BLOCK_M) && (!is_last_m || row * BLOCK_M + tid < Mdim);
    if (tid < BLOCK_M) smem_sa[0][tid] = m_in_range ? sa_block[tid] : 0.f;
    float sb_cur = sb_base[0];
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    #pragma unroll
    for (int tile = 0; tile < num_tiles - 1; ++tile) {

        constexpr int A_BUF = (BLOCK_M * K_STEP) / NUM_THREADS;
        constexpr int B_BUF = (BLOCK_N * K_STEP) / NUM_THREADS;
        float4 a_buffer_next[A_BUF * sizeof(fp8e4m3) / sizeof(float4)];
        float4 b_buffer_next[B_BUF * sizeof(fp8e4m3) / sizeof(float4)];

        zero(partial[0]); zero(partial[1]);

        // Cluster 0
        // Full M-blocks prefetch A global->reg here (overlapped, written to LDS at
        // cluster 6). The partial last M-block instead does a bounds-checked
        // global->LDS load at cluster 6 (avoids OOB reads past A's allocation).
        if (!is_last_m)
            load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, A_BUF, g.a, {0, 0, row, tile + 1}, As);
        // Prefetch next slab's scales (global->reg); latency hidden by clusters 1..7.
        float sa_next = m_in_range ? sa_block[(tile + 1) * Mdim + tid] : 0.f;
        float sb_next = sb_base[tile + 1];
        load(at[1], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 0}));
        load(at[2], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 0}));
        load(bt[0], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 0}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[1], bt[0], partial[0]);
        mma_ABt(partial[1], at[2], bt[0], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2
        load(bt[3], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 1}));
        load(at[4], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 1}));
        load(at[5], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 1}));
        load(bt[0], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 2}));
        load(at[1], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[4], bt[3], partial[0]);
        mma_ABt(partial[1], at[5], bt[3], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4
        // Like A: full N-blocks prefetch B global->reg; the partial last N-block
        // does a bounds-checked global->LDS load at cluster 6 instead.
        if (!is_last_n)
            load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, B_BUF, g.b, {0, 0, col, tile + 1}, Bs);
        load(at[2], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 2}));
        load(bt[6], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 3}));
        load(at[7], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 3}));
        load(at[5], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[1], bt[0], partial[0]);
        mma_ABt(partial[1], at[2], bt[0], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6
        asm volatile("s_waitcnt lgkmcnt(0)");
        if (is_last_m) load_tile_masked_rows(As, g.a, row, tile + 1, Mdim);
        else           store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        if (is_last_n) load_tile_masked_rows(Bs, g.b, col, tile + 1, Ndim);
        else           store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        // Stash next slab's scale_A into the other LDS buffer (reg->LDS).
        if (tid < BLOCK_M) smem_sa[(tile + 1) & 1][tid] = sa_next;
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[7], bt[6], partial[0]);
        mma_ABt(partial[1], at[5], bt[6], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 8 = apply (VALU only). Own cluster (barrier) so via ping-pong it
        // overlaps the OTHER slot's MMA clusters (VALU unit ∥ matrix unit).
        const float *sa_cur = smem_sa[tile & 1];
        apply_block_scale(C_accum[0], partial[0], sa_cur, sb_cur, local_m0);
        apply_block_scale(C_accum[1], partial[1], sa_cur, sb_cur, local_m1);
        sb_cur = sb_next;
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

    }

    // Epilogue
    zero(partial[0]); zero(partial[1]);
    // Cluster 0
    __builtin_amdgcn_sched_barrier(0);
    load(bt[0], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 0}));
    load(at[1], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 0}));
    load(at[2], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 1
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[1], bt[0], partial[0]);
    mma_ABt(partial[1], at[2], bt[0], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2
    load(bt[3], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 1}));
    load(at[4], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 1}));
    load(at[5], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[4], bt[3], partial[0]);
    mma_ABt(partial[1], at[5], bt[3], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4
    load(bt[0], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 2}));
    load(at[1], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 2}));
    load(at[2], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 2}));
    load(bt[3], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 3}));
    load(at[4], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 3}));
    load(at[5], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 5
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[1], bt[0], partial[0]);
    mma_ABt(partial[1], at[2], bt[0], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 7
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[4], bt[3], partial[0]);
    mma_ABt(partial[1], at[5], bt[3], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    {
        const int last = num_tiles - 1;
        const float *sa_cur = smem_sa[last & 1];
        apply_block_scale(C_accum[0], partial[0], sa_cur, sb_cur, local_m0);
        apply_block_scale(C_accum[1], partial[1], sa_cur, sb_cur, local_m1);
    }

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    apply_rtne_bias(C_accum[0]);
    apply_rtne_bias(C_accum[1]);
    if (is_last_m || is_last_n) {
        store_masked_M(g.c.raw_ptr, C_accum[0], row * 4 + warp_row,     col * 4 + warp_col, Mdim, Ndim);
        store_masked_M(g.c.raw_ptr, C_accum[1], row * 4 + warp_row + 2, col * 4 + warp_col, Mdim, Ndim);
    } else {
        store(g.c, C_accum[0], {0, 0, row * 4 + warp_row,     col * 4 + warp_col});
        store(g.c, C_accum[1], {0, 0, row * 4 + warp_row + 2, col * 4 + warp_col});
    }
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    // Aligned M and N use the IS_PARTIAL_*=false instance (no masked-path codegen
    // -> same VGPRs/perf as the original kernel). Only the partial dimension(s) pay
    // for the bounds-checked instance.
    const bool pm = (g.Mdim() % BLOCK_M != 0);
    const bool pn = (g.Ndim() % BLOCK_N != 0);
    auto launch = [&](auto kern) {
        hipFuncSetAttribute((void*)kern, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        kern<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    };
    if      (!pm && !pn) launch(micro_tk<false, false>);
    else if ( pm && !pn) launch(micro_tk<true,  false>);
    else if (!pm &&  pn) launch(micro_tk<false, true >);
    else                 launch(micro_tk<true,  true >);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "fp8 gemm test kernel";
    py::bind_kernel<micro_tk<false, false>>(m, "micro_tk", &micro_globals::a, &micro_globals::b, &micro_globals::c, &micro_globals::scale_a, &micro_globals::scale_b);
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c, &micro_globals::scale_a, &micro_globals::scale_b);
}
