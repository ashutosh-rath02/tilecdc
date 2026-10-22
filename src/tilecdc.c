/*
 * TileCDC: content-defined chunking on CPU matrix units (see include/tilecdc.h
 * for the normative definition). Three implementations of one function:
 *
 *   scalar  reference, any CPU
 *   avx512  AVX-512 VBMI + VNNI: vpermb builds 4-byte sliding windows, vpdpbusd
 *           multiplies them by the key (12 of each per 16 hashes)
 *   amx     Intel AMX: a tile load with a 16-byte stride over 64-byte rows makes
 *           the A tile a Hankel matrix of the byte stream; one TDPBUSD with a
 *           constant Toeplitz key tile gives 256 window hashes
 *
 * All three consume the same S-box-substituted block and produce bit-identical
 * candidate bitmaps, so chunk boundaries never depend on the hardware.
 */
#define _GNU_SOURCE
#include "tilecdc.h"

#include <cpuid.h>
#include <errno.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#define W TILECDC_WINDOW
#define BLOCK 1024                 /* hashes per block */
#define BLKBYTES (BLOCK + 64)      /* bytes needed per block (window + slack) */
#define MIXC 0x9E3779B1u

struct tilecdc {
    size_t min, avg, max;
    uint32_t smask, lmask;         /* top-bit masks for the strict / loose test */
    tilecdc_impl impl;
    int8_t key[W];
    uint8_t sbox[256] __attribute__((aligned(64)));
    int8_t btile[16][64] __attribute__((aligned(64)));   /* Toeplitz key tile for AMX */
    uint8_t idx[12][64] __attribute__((aligned(64)));    /* vpermb sliding-window indices */
    int32_t keyq[12];                                    /* KEY[4g..4g+3] packed */
};

typedef void (*block_fn)(const tilecdc *t, const uint8_t *y, uint64_t S[16], uint64_t L[16]);

/* ------------------------------ constants ------------------------------ */
static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void derive(tilecdc *t, uint64_t seed) {
    uint64_t s = seed ^ 0x54696C6543444321ull;   /* "TileCDC!" */
    for (int i = 0; i < 256; i++) t->sbox[i] = (uint8_t)i;
    for (int i = 255; i > 0; i--) {
        int j = (int)(splitmix64(&s) % (uint64_t)(i + 1));
        uint8_t x = t->sbox[i]; t->sbox[i] = t->sbox[j]; t->sbox[j] = x;
    }
    for (int d = 0; d < W; d++) {
        int v;
        do { v = (int)(splitmix64(&s) & 0xFF) - 128; } while (v == 0 || v == -128);
        t->key[d] = (int8_t)v;
    }
    /* AMX: C[r][j] = sum_k A[r][k] * B[k][j] with A[r][k] = y[16r + k];
       B[k][j] = KEY[k - j] gives C[r][j] = h(16r + j). TDPBUSD wants B packed
       as 16 rows of (16 columns x 4 consecutive k). */
    memset(t->btile, 0, sizeof t->btile);
    for (int k = 0; k < 64; k++)
        for (int j = 0; j < 16; j++) {
            int d = k - j;
            if (d >= 0 && d < W) t->btile[k / 4][j * 4 + k % 4] = t->key[d];
        }
    /* AVX-512: lane k of group g holds bytes y[b + 4g + k .. +3] */
    for (int g = 0; g < 12; g++) {
        for (int k = 0; k < 16; k++)
            for (int j = 0; j < 4; j++) t->idx[g][k * 4 + j] = (uint8_t)(4 * g + k + j);
        uint32_t q = 0;
        for (int j = 0; j < 4; j++) q |= (uint32_t)(uint8_t)t->key[4 * g + j] << (8 * j);
        t->keyq[g] = (int32_t)q;
    }
}

/* ------------------------------ CPU support ------------------------------ */
static int avx512_ok, amx_ok;
static pthread_once_t detect_once = PTHREAD_ONCE_INIT;

static void detect(void) {
    __builtin_cpu_init();
    avx512_ok = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512vnni");
    unsigned a, b, c, d;
    if (avx512_ok && __get_cpuid_count(7, 0, &a, &b, &c, &d) && (d & (1u << 24)) && (d & (1u << 25))) {
        /* ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA: ask the kernel for tile state */
        amx_ok = syscall(SYS_arch_prctl, 0x1023, 18) == 0;
    }
}
static int has_avx512(void) { pthread_once(&detect_once, detect); return avx512_ok; }
static int has_amx(void) { pthread_once(&detect_once, detect); return amx_ok; }

int tilecdc_impl_available(tilecdc_impl impl) {
    switch (impl) {
    case TILECDC_IMPL_SCALAR: return 1;
    case TILECDC_IMPL_AVX512: return has_avx512();
    case TILECDC_IMPL_AMX: return has_amx();
    case TILECDC_IMPL_AUTO: return 1;
    }
    return 0;
}

const char *tilecdc_impl_name(tilecdc_impl impl) {
    switch (impl) {
    case TILECDC_IMPL_SCALAR: return "scalar";
    case TILECDC_IMPL_AVX512: return "avx512";
    case TILECDC_IMPL_AMX: return "amx";
    default: return "auto";
    }
}

/* ------------------------------ S-box ------------------------------ */
static void sbox_scalar(const tilecdc *t, const uint8_t *src, uint8_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = t->sbox[src[i]];
}

__attribute__((target("avx512f,avx512bw,avx512vbmi")))
static void sbox_avx512(const tilecdc *t, const uint8_t *src, uint8_t *dst, size_t n) {   /* n % 64 == 0 */
    __m512i t0 = _mm512_load_si512(t->sbox), t1 = _mm512_load_si512(t->sbox + 64);
    __m512i t2 = _mm512_load_si512(t->sbox + 128), t3 = _mm512_load_si512(t->sbox + 192);
    for (size_t i = 0; i < n; i += 64) {
        __m512i x = _mm512_loadu_si512(src + i);
        __m512i lo = _mm512_permutex2var_epi8(t0, x, t1), hi = _mm512_permutex2var_epi8(t2, x, t3);
        _mm512_storeu_si512(dst + i, _mm512_mask_blend_epi8(_mm512_movepi8_mask(x), lo, hi));
    }
}

/* ------------------------------ hash blocks ------------------------------ */
/* y holds BLKBYTES substituted bytes starting at hash index i0; set bit b of
   S/L when hash i0+b passes the strict/loose test. */
static void block_scalar(const tilecdc *t, const uint8_t *y, uint64_t S[16], uint64_t L[16]) {
    memset(S, 0, 16 * 8); memset(L, 0, 16 * 8);
    for (int b = 0; b < BLOCK; b++) {
        int32_t h = 0;
        for (int d = 0; d < W; d++) h += (int32_t)t->key[d] * (int32_t)y[b + d];
        uint32_t m = (uint32_t)h * MIXC;
        if (!(m & t->smask)) S[b >> 6] |= 1ull << (b & 63);
        if (!(m & t->lmask)) L[b >> 6] |= 1ull << (b & 63);
    }
}

__attribute__((target("avx512f,avx512bw,avx512vbmi,avx512vnni")))
static void block_avx512(const tilecdc *t, const uint8_t *y, uint64_t S[16], uint64_t L[16]) {
    __m512i idx[12], kq[12];
    for (int g = 0; g < 12; g++) { idx[g] = _mm512_load_si512(t->idx[g]); kq[g] = _mm512_set1_epi32(t->keyq[g]); }
    const __m512i mul = _mm512_set1_epi32((int)MIXC), sm = _mm512_set1_epi32((int)t->smask), lm = _mm512_set1_epi32((int)t->lmask);
    for (int w = 0; w < 16; w++) {
        uint64_t s64 = 0, l64 = 0;
        for (int u = 0; u < 4; u++) {
            int b = w * 64 + u * 16;
            __m512i base = _mm512_loadu_si512(y + b);
            __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
            for (int g = 0; g < 12; g += 2) {
                acc0 = _mm512_dpbusd_epi32(acc0, _mm512_permutexvar_epi8(idx[g], base), kq[g]);
                acc1 = _mm512_dpbusd_epi32(acc1, _mm512_permutexvar_epi8(idx[g + 1], base), kq[g + 1]);
            }
            __m512i m = _mm512_mullo_epi32(_mm512_add_epi32(acc0, acc1), mul);
            s64 |= (uint64_t)_mm512_testn_epi32_mask(m, sm) << (16 * u);
            l64 |= (uint64_t)_mm512_testn_epi32_mask(m, lm) << (16 * u);
        }
        S[w] = s64; L[w] = l64;
    }
}

typedef struct { uint8_t palette, start; uint8_t r[14]; uint16_t colsb[16]; uint8_t rows[16]; } __attribute__((packed)) tilecfg;

/* Configure tiles for TileCDC, remembering any configuration the calling thread
   had, and put it back afterwards. Tile register contents are not preserved:
   do not call TileCDC while holding live data in tile registers. */
__attribute__((target("amx-tile")))
static void amx_configure(tilecfg *saved) {
    tilecfg c;
    memset(saved, 0, sizeof *saved);
    _tile_storeconfig(saved);
    memset(&c, 0, sizeof c);
    c.palette = 1;
    for (int i = 0; i < 8; i++) { c.colsb[i] = 64; c.rows[i] = 16; }
    _tile_loadconfig(&c);
}
__attribute__((target("amx-tile")))
static void amx_release(const tilecfg *saved) {
    if (saved->palette) _tile_loadconfig(saved);
    else _tile_release();
}

__attribute__((target("amx-tile,amx-int8,avx512f,avx512bw")))
static void block_amx(const tilecdc *t, const uint8_t *y, uint64_t S[16], uint64_t L[16]) {
    int32_t C[4][16][16] __attribute__((aligned(64)));
    _tile_loadd(7, t->btile, 64);
    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
    /* stride 16 < row length 64: row r of each A tile = y[256q + 16r .. +64) */
    _tile_loadd(4, y, 16);       _tile_dpbusd(0, 4, 7);
    _tile_loadd(5, y + 256, 16); _tile_dpbusd(1, 5, 7);
    _tile_loadd(6, y + 512, 16); _tile_dpbusd(2, 6, 7);
    _tile_loadd(4, y + 768, 16); _tile_dpbusd(3, 4, 7);
    _tile_stored(0, C[0], 64); _tile_stored(1, C[1], 64); _tile_stored(2, C[2], 64); _tile_stored(3, C[3], 64);
    const int32_t *h = &C[0][0][0];   /* row-major: h[b] = hash i0 + b */
    const __m512i mul = _mm512_set1_epi32((int)MIXC), sm = _mm512_set1_epi32((int)t->smask), lm = _mm512_set1_epi32((int)t->lmask);
    for (int w = 0; w < 16; w++) {
        uint64_t s64 = 0, l64 = 0;
        for (int u = 0; u < 4; u++) {
            __m512i m = _mm512_mullo_epi32(_mm512_load_si512(h + w * 64 + u * 16), mul);
            s64 |= (uint64_t)_mm512_testn_epi32_mask(m, sm) << (16 * u);
            l64 |= (uint64_t)_mm512_testn_epi32_mask(m, lm) << (16 * u);
        }
        S[w] = s64; L[w] = l64;
    }
}

/* ------------------------------ cut search ------------------------------ */
static size_t cut_core(tilecdc *t, const uint8_t *p, size_t n) {
    if (n <= t->min) return n;
    size_t end = n < t->max ? n : t->max, mid = n < t->avg ? n : t->avg;
    block_fn blk = t->impl == TILECDC_IMPL_AMX ? block_amx : t->impl == TILECDC_IMPL_AVX512 ? block_avx512 : block_scalar;
    int vec = t->impl != TILECDC_IMPL_SCALAR;
    uint8_t raw[BLKBYTES] __attribute__((aligned(64)));
    uint8_t y[BLKBYTES] __attribute__((aligned(64)));
    uint64_t S[16], L[16];
    for (size_t i0 = t->min + 1 - W;; i0 += BLOCK) {
        size_t c0 = i0 + W;                        /* chunk end tested by bit 0 */
        if (c0 > end) return end;
        /* substitute the bytes of this block; never read past n */
        const uint8_t *src = p + i0;
        size_t avail = n - i0;
        if (avail < BLKBYTES) { memcpy(raw, src, avail); memset(raw + avail, 0, BLKBYTES - avail); src = raw; }
        if (vec) sbox_avx512(t, src, y, BLKBYTES);
        else sbox_scalar(t, src, y, BLKBYTES);
        blk(t, y, S, L);
        for (int w = 0; w < 16; w++) {
            size_t cw = c0 + (size_t)w * 64;       /* chunk end for bit 0 of this word */
            if (cw > end) return end;
            uint64_t sel_s = 0, sel_l = 0;
            /* bits whose end c = cw + b lies in (min, mid] are strict, (mid, end] loose */
            size_t ns = mid >= cw ? mid - cw + 1 : 0;               /* number of strict-region bits */
            size_t ne = end >= cw ? end - cw + 1 : 0;               /* bits up to end */
            uint64_t ms = ns >= 64 ? ~0ull : ((1ull << ns) - 1);
            uint64_t me = ne >= 64 ? ~0ull : ((1ull << ne) - 1);
            sel_s = S[w] & ms;
            if (sel_s) return cw + (size_t)__builtin_ctzll(sel_s);
            sel_l = L[w] & me & ~ms;
            if (sel_l) {
                /* a strict hit later in this word cannot precede a loose hit
                   only if the strict region already ended: guaranteed by ~ms */
                return cw + (size_t)__builtin_ctzll(sel_l);
            }
        }
    }
}

size_t tilecdc_cut(tilecdc *t, const uint8_t *data, size_t len) {
    if (t->impl == TILECDC_IMPL_AMX && len > t->min) {
        tilecfg saved;
        amx_configure(&saved);
        size_t r = cut_core(t, data, len);
        amx_release(&saved);
        return r;
    }
    return cut_core(t, data, len);
}

size_t tilecdc_chunk_all(tilecdc *t, const uint8_t *data, size_t len, size_t *ends, size_t cap) {
    size_t n = 0, s = 0;
    tilecfg saved;
    if (t->impl == TILECDC_IMPL_AMX) amx_configure(&saved);
    while (s < len) {
        s += cut_core(t, data + s, len - s);
        if (n < cap) ends[n] = s;
        n++;
    }
    if (t->impl == TILECDC_IMPL_AMX) amx_release(&saved);
    return n;
}

/* ------------------------------ setup ------------------------------ */
void tilecdc_default_params(tilecdc_params *p) {
    p->min_size = 2048;
    p->avg_size = 8192;
    p->max_size = 65536;
    p->seed = 0;
    p->norm = 1;
    p->impl = TILECDC_IMPL_AUTO;
}

tilecdc *tilecdc_new(const tilecdc_params *params) {
    tilecdc_params p;
    if (params) p = *params; else tilecdc_default_params(&p);
    if (p.min_size < 64 || p.avg_size <= p.min_size || p.max_size < p.avg_size) { errno = EINVAL; return NULL; }
    if (p.avg_size > TILECDC_MAX_AVG || p.norm < 0 || p.norm > 4) { errno = EINVAL; return NULL; }
    int lg = 63 - __builtin_clzll((unsigned long long)p.avg_size);
    int sb = lg + p.norm, lb = lg - p.norm - 1;
    if (sb > 31 || lb < 1) { errno = EINVAL; return NULL; }
    tilecdc_impl impl = p.impl;
    if (impl == TILECDC_IMPL_AUTO) impl = has_amx() ? TILECDC_IMPL_AMX : has_avx512() ? TILECDC_IMPL_AVX512 : TILECDC_IMPL_SCALAR;
    else if (!tilecdc_impl_available(impl)) { errno = ENOTSUP; return NULL; }
    tilecdc *t = aligned_alloc(64, (sizeof(tilecdc) + 63) & ~(size_t)63);
    if (!t) return NULL;
    memset(t, 0, sizeof *t);
    t->min = p.min_size; t->avg = p.avg_size; t->max = p.max_size;
    t->smask = 0xFFFFFFFFu << (32 - sb);
    t->lmask = 0xFFFFFFFFu << (32 - lb);
    t->impl = impl;
    derive(t, p.seed);
    return t;
}

void tilecdc_free(tilecdc *t) { free(t); }
tilecdc_impl tilecdc_get_impl(const tilecdc *t) { return t->impl; }
