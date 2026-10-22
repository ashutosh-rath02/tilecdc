/*
 * TileCDC benchmarks (single source of every number in the paper).
 *
 *   bench compare <dataset> <files...>   throughput, chunk statistics, dedup ratio:
 *                                        FastCDC 2020 (C), TileCDC scalar/avx512/amx
 *   bench sweep <files...>               average-size sweep 4..64 KiB
 *   bench edits <file>                   chunk reuse after random insertions / deletions
 *   bench breakdown                      AMX cost breakdown per stage (cached block)
 *   bench threads <files...>             1 vs 2 threads (one file per thread)
 *
 * Throughput = input bytes / best-of-5 wall time for chunking only (the data is
 * in memory; fingerprinting is excluded and reported separately).
 */
#include "../src/tilecdc.c"
#include "fastcdc_ref.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

typedef struct { uint8_t *p; size_t n; } buf_t;
static buf_t load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); size_t n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc(n + 64);
    if (fread(p, 1, n, f) != n) { perror("read"); exit(1); }
    fclose(f);
    return (buf_t){p, n};
}

/* ---------------- chunk fingerprint set for dedup ratio ---------------- */
static uint64_t fp64(const uint8_t *p, size_t n) {
    uint64_t h = 0x243F6A8885A308D3ull ^ n, k;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { memcpy(&k, p + i, 8); k *= 0x87c37b91114253d5ull; k = (k << 31) | (k >> 33); k *= 0x4cf5ad432745937full; h ^= k; h = ((h << 27) | (h >> 37)) * 5 + 0x52dce729; }
    for (; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull; h ^= h >> 33;
    return h ? h : 1;
}
typedef struct { uint64_t *k; size_t cap; } set_t;
static set_t set_new(size_t cap) { set_t s = {calloc(cap, 8), cap}; return s; }
static int set_add(set_t *s, uint64_t key) {
    size_t m = s->cap - 1, i = (key * 0x9E3779B97F4A7C15ull >> 17) & m;
    while (s->k[i]) { if (s->k[i] == key) return 0; i = (i + 1) & m; }
    s->k[i] = key;
    return 1;
}

/* ---------------- methods ---------------- */
enum { M_FASTCDC, M_SCALAR, M_AVX512, M_AMX, NM };
static const char *MNAME[NM] = {"fastcdc-c", "tilecdc-scalar", "tilecdc-avx512", "tilecdc-amx"};

typedef struct { int m; fastcdc_t fc; tilecdc *t; } chunker_t;
static int make_chunker(chunker_t *c, int m, size_t min, size_t avg, size_t max) {
    c->m = m;
    if (m == M_FASTCDC) { fastcdc_init(&c->fc, min, avg, max, 2); c->t = NULL; return 1; }
    tilecdc_params p; tilecdc_default_params(&p);
    p.min_size = min; p.avg_size = avg; p.max_size = max;
    p.impl = m == M_SCALAR ? TILECDC_IMPL_SCALAR : m == M_AVX512 ? TILECDC_IMPL_AVX512 : TILECDC_IMPL_AMX;
    c->t = tilecdc_new(&p);
    return c->t != NULL;
}
static size_t run(chunker_t *c, const uint8_t *p, size_t n, size_t *ends, size_t cap) {
    if (c->m == M_FASTCDC) {
        size_t k = 0, s = 0;
        while (s < n) { s += fastcdc_cut(&c->fc, p + s, n - s); if (k < cap) ends[k] = s; k++; }
        return k;
    }
    return tilecdc_chunk_all(c->t, p, n, ends, cap);
}

typedef struct { double gbps, avg, sd, dedup; size_t chunks; double p5, p95; } result_t;
static int cmp_sz(const void *a, const void *b) { size_t x = *(const size_t *)a, y = *(const size_t *)b; return (x > y) - (x < y); }

static result_t measure(int m, buf_t *B, int nb, size_t min, size_t avg, size_t max, int reps) {
    result_t r = {0};
    chunker_t c;
    if (!make_chunker(&c, m, min, avg, max)) { r.gbps = -1; return r; }
    size_t total = 0, maxn = 0;
    for (int i = 0; i < nb; i++) { total += B[i].n; if (B[i].n > maxn) maxn = B[i].n; }
    size_t cap = maxn / 64 + 16;
    size_t *ends = malloc(cap * sizeof(size_t));
    size_t *sizes = malloc((total / 64 + 16 * nb) * sizeof(size_t));
    set_t S = set_new(1u << 24);
    size_t ns = 0, uniq = 0;
    double tsum = 0;
    for (int i = 0; i < nb; i++) {
        double best = 1e30;
        size_t k = 0;
        for (int rep = 0; rep < reps; rep++) {
            double t0 = now();
            k = run(&c, B[i].p, B[i].n, ends, cap);
            double t = now() - t0;
            if (t < best) best = t;
        }
        tsum += best;
        size_t s = 0;
        for (size_t j = 0; j < k; j++) {
            size_t len = ends[j] - s;
            sizes[ns++] = len;
            if (set_add(&S, fp64(B[i].p + s, len))) uniq += len;
            s = ends[j];
        }
    }
    double sum = 0, sum2 = 0;
    for (size_t j = 0; j < ns; j++) { sum += sizes[j]; sum2 += (double)sizes[j] * sizes[j]; }
    qsort(sizes, ns, sizeof(size_t), cmp_sz);
    r.gbps = total / tsum / 1e9;
    r.chunks = ns;
    r.avg = sum / ns;
    r.sd = sqrt(sum2 / ns - r.avg * r.avg);
    r.dedup = (double)total / uniq;
    r.p5 = sizes[ns * 5 / 100];
    r.p95 = sizes[ns * 95 / 100];
    free(ends); free(sizes); free(S.k);
    if (c.t) tilecdc_free(c.t);
    return r;
}

static void cmd_compare(const char *name, buf_t *B, int nb) {
    size_t total = 0;
    for (int i = 0; i < nb; i++) total += B[i].n;
    printf("# dataset,method,bytes,gbps,chunks,avg_kib,sd_kib,p5_kib,p95_kib,dedup\n");
    for (int m = 0; m < NM; m++) {
        result_t r = measure(m, B, nb, 2048, 8192, 65536, m == M_SCALAR ? 1 : 5);
        if (r.gbps < 0) continue;
        printf("%s,%s,%zu,%.3f,%zu,%.3f,%.3f,%.2f,%.2f,%.4f\n", name, MNAME[m], total, r.gbps, r.chunks, r.avg / 1024, r.sd / 1024, r.p5 / 1024, r.p95 / 1024, r.dedup);
        fflush(stdout);
    }
    /* fingerprinting cost for context: our 64-bit fp over every byte */
    double t0 = now(); uint64_t x = 0;
    for (int i = 0; i < nb; i++) x ^= fp64(B[i].p, B[i].n);
    printf("# fingerprint-64 pass: %.2f GB/s (%llx)\n", total / (now() - t0) / 1e9, (unsigned long long)(x & 0xF));
}

static void cmd_sweep(buf_t *B, int nb) {
    printf("# avg_target_kib,method,gbps,avg_kib,sd_kib,dedup\n");
    size_t avgs[] = {4096, 8192, 16384, 32768};
    for (int a = 0; a < 4; a++) {
        size_t avg = avgs[a], min = avg / 4, max = avg * 8;
        int ms[] = {M_FASTCDC, M_AVX512, M_AMX};
        for (int q = 0; q < 3; q++) {
            result_t r = measure(ms[q], B, nb, min, avg, max, 3);
            if (r.gbps < 0) continue;
            printf("%zu,%s,%.3f,%.3f,%.3f,%.4f\n", avg / 1024, MNAME[ms[q]], r.gbps, r.avg / 1024, r.sd / 1024, r.dedup);
            fflush(stdout);
        }
    }
}

/* Fraction of the edited file's bytes that fall in chunks also present in the
   original, averaged over 5 seeds. TileCDC runs twice: at default parameters
   and with avg_size scaled so its mean chunk size matches FastCDC's on this file. */
static double mean_chunk(chunker_t *c, buf_t A, size_t *ends, size_t cap) { return (double)A.n / run(c, A.p, A.n, ends, cap); }
static double reuse(chunker_t *c, buf_t A, const uint8_t *q, size_t dst, size_t *ends, size_t cap) {
    set_t S = set_new(1u << 22);
    size_t k = run(c, A.p, A.n, ends, cap), s = 0;
    for (size_t j = 0; j < k; j++) { set_add(&S, fp64(A.p + s, ends[j] - s)); s = ends[j]; }
    k = run(c, q, dst, ends, cap); s = 0;
    size_t reused = 0;
    for (size_t j = 0; j < k; j++) { size_t len = ends[j] - s; if (!set_add(&S, fp64(q + s, len))) reused += len; s = ends[j]; }
    free(S.k);
    return 100.0 * reused / dst;
}
static void cmd_edits(buf_t A) {
    size_t cap = A.n / 64 + 4096;
    size_t *ends = malloc(cap * sizeof(size_t));
    chunker_t fc, td, tm;
    make_chunker(&fc, M_FASTCDC, 2048, 8192, 65536);
    make_chunker(&td, M_AMX, 2048, 8192, 65536);
    double mf = mean_chunk(&fc, A, ends, cap), mt = mean_chunk(&td, A, ends, cap);
    /* search avg_size (same min/max) for the mean closest to FastCDC's */
    size_t best_avg = 8192; double best_err = 1e9;
    for (size_t avg = 8192; avg <= 16383; avg += 128) {
        chunker_t c; make_chunker(&c, M_AMX, 2048, avg, 65536);
        double e = fabs(mean_chunk(&c, A, ends, cap) - mf);
        if (e < best_err) { best_err = e; best_avg = avg; }
        tilecdc_free(c.t);
    }
    make_chunker(&tm, M_AMX, 2048, best_avg, 65536);
    printf("# mean chunk: fastcdc-c %.0f, tilecdc default %.0f, tilecdc matched (avg_size=%zu) %.0f\n", mf, mt, best_avg, mean_chunk(&tm, A, ends, cap));
    printf("# edit_kind,edits,fastcdc_c,tilecdc_default,tilecdc_matched (mean of 5 seeds)\n");
    int counts[] = {10, 100, 1000};
    for (int kind = 0; kind < 2; kind++)
        for (int ci = 0; ci < 3; ci++) {
            int e = counts[ci];
            double r[3] = {0, 0, 0};
            for (int seed = 0; seed < 5; seed++) {
                uint8_t *q = malloc(A.n + (size_t)e * 128 + 64);
                size_t *pos = malloc(e * sizeof(size_t));
                rs = 1234 + (uint64_t)e * 7 + kind + 1000003ull * seed;
                for (int i = 0; i < e; i++) pos[i] = rnd() % (A.n - 256);
                qsort(pos, e, sizeof(size_t), cmp_sz);
                size_t src = 0, dst = 0;
                for (int i = 0; i < e; i++) {
                    if (pos[i] < src) continue;
                    memcpy(q + dst, A.p + src, pos[i] - src); dst += pos[i] - src; src = pos[i];
                    int k = 1 + (int)(rnd() % 100);
                    if (kind == 0) for (int j = 0; j < k; j++) q[dst++] = (uint8_t)rnd();   /* insert */
                    else src += (size_t)k;                                                   /* delete */
                }
                memcpy(q + dst, A.p + src, A.n - src); dst += A.n - src;
                r[0] += reuse(&fc, A, q, dst, ends, cap) / 5;
                r[1] += reuse(&td, A, q, dst, ends, cap) / 5;
                r[2] += reuse(&tm, A, q, dst, ends, cap) / 5;
                free(q); free(pos);
            }
            printf("%s,%d,%.3f,%.3f,%.3f\n", kind ? "delete" : "insert", e, r[0], r[1], r[2]);
        }
    tilecdc_free(td.t); tilecdc_free(tm.t); free(ends);
}

__attribute__((target("amx-tile,amx-int8")))
static void amx_tmul_part(const tilecdc *t, const uint8_t *y, int32_t C[4][16][16], int all_stores) {
    _tile_loadd(7, t->btile, 64);
    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
    _tile_loadd(4, y, 16);       _tile_dpbusd(0, 4, 7);
    _tile_loadd(5, y + 256, 16); _tile_dpbusd(1, 5, 7);
    _tile_loadd(6, y + 512, 16); _tile_dpbusd(2, 6, 7);
    _tile_loadd(4, y + 768, 16); _tile_dpbusd(3, 4, 7);
    if (all_stores) { _tile_stored(0, C[0], 64); _tile_stored(1, C[1], 64); _tile_stored(2, C[2], 64); }
    _tile_stored(3, C[3], 64);   /* keep the products live */
}
/* per-stage AMX cost on a cache-resident block, in ns per input byte */
static void cmd_breakdown(void) {
    tilecdc_params p; tilecdc_default_params(&p); p.impl = TILECDC_IMPL_AMX;
    tilecdc *t = tilecdc_new(&p);
    if (!t) { printf("# AMX unavailable\n"); return; }
    uint8_t raw[BLKBYTES] __attribute__((aligned(64))), y[BLKBYTES] __attribute__((aligned(64)));
    for (int i = 0; i < BLKBYTES; i++) raw[i] = (uint8_t)rnd();
    uint64_t S[16], L[16];
    const int R = 200000;
    tilecfg saved;
    amx_configure(&saved);
    double t0 = now();
    for (int r = 0; r < R; r++) { sbox_avx512(t, raw, y, BLKBYTES); __asm__ volatile("" ::"r"(y) : "memory"); }
    double t_sbox = now() - t0;
    t0 = now();
    for (int r = 0; r < R; r++) { block_amx(t, y, S, L); __asm__ volatile("" ::"r"(S) : "memory"); }
    double t_amx = now() - t0;
    static int32_t Ct[4][16][16] __attribute__((aligned(64)));
    t0 = now();
    for (int r = 0; r < R; r++) { amx_tmul_part(t, y, Ct, 0); __asm__ volatile("" ::"r"(Ct) : "memory"); }
    double t_tmul = now() - t0;
    t0 = now();
    for (int r = 0; r < R; r++) { amx_tmul_part(t, y, Ct, 1); __asm__ volatile("" ::"r"(Ct) : "memory"); }
    double t_store = now() - t0;
    amx_release(&saved);
    t0 = now();
    for (int r = 0; r < R; r++) { block_avx512(t, y, S, L); __asm__ volatile("" ::"r"(S) : "memory"); }
    double t_avx = now() - t0;
    t0 = now();
    for (int r = 0; r < R / 20; r++) { block_scalar(t, y, S, L); __asm__ volatile("" ::"r"(S) : "memory"); }
    double t_sc = (now() - t0) * 20;
    double per = (double)R * BLOCK;
    printf("# stage,ns_per_byte,gbps_equiv\n");
    printf("sbox_avx512,%.4f,%.2f\n", t_sbox / per * 1e9, per / t_sbox / 1e9);
    printf("hash+mix+test_amx,%.4f,%.2f\n", t_amx / per * 1e9, per / t_amx / 1e9);
    printf("amx_loads+tdpbusd+1store,%.4f,%.2f\n", t_tmul / per * 1e9, per / t_tmul / 1e9);
    printf("amx_loads+tdpbusd+stores,%.4f,%.2f\n", t_store / per * 1e9, per / t_store / 1e9);
    printf("hash+mix+test_avx512,%.4f,%.2f\n", t_avx / per * 1e9, per / t_avx / 1e9);
    printf("hash+mix+test_scalar,%.4f,%.2f\n", t_sc / per * 1e9, per / t_sc / 1e9);
    tilecdc_free(t);
}

typedef struct { int m; buf_t b; double secs; } targ_t;
static void *thr(void *a) {
    targ_t *x = a;
    chunker_t c;
    make_chunker(&c, x->m, 2048, 8192, 65536);
    size_t *ends = malloc((x->b.n / 64 + 16) * sizeof(size_t));
    double t0 = now();
    run(&c, x->b.p, x->b.n, ends, x->b.n / 64 + 16);
    x->secs = now() - t0;
    free(ends);
    return NULL;
}
static void cmd_threads(buf_t *B, int nb) {
    printf("# method,threads,aggregate_gbps\n");
    int ms[] = {M_FASTCDC, M_AMX};
    for (int mi = 0; mi < 2; mi++)
        for (int th = 1; th <= 2; th++) {
            double best = 1e30; size_t bytes = 0;
            for (int rep = 0; rep < 3; rep++) {
                targ_t a[2]; pthread_t id[2];
                double t0 = now();
                bytes = 0;
                for (int i = 0; i < th; i++) { a[i].m = ms[mi]; a[i].b = B[i % nb]; bytes += B[i % nb].n; pthread_create(&id[i], NULL, thr, &a[i]); }
                for (int i = 0; i < th; i++) pthread_join(id[i], NULL);
                double t = now() - t0;
                if (t < best) best = t;
            }
            printf("%s,%d,%.3f\n", MNAME[ms[mi]], th, bytes / best / 1e9);
        }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench compare|sweep|edits|breakdown|threads ...\n"); return 2; }
    const char *cmd = argv[1];
    if (!strcmp(cmd, "breakdown")) { cmd_breakdown(); return 0; }
    int first = !strcmp(cmd, "compare") ? 3 : 2;
    int nb = argc - first;
    if (nb < 1) { fprintf(stderr, "no files\n"); return 2; }
    buf_t *B = malloc(nb * sizeof(buf_t));
    for (int i = 0; i < nb; i++) B[i] = load(argv[first + i]);
    if (!strcmp(cmd, "compare")) cmd_compare(argv[2], B, nb);
    else if (!strcmp(cmd, "sweep")) cmd_sweep(B, nb);
    else if (!strcmp(cmd, "edits")) cmd_edits(B[0]);
    else if (!strcmp(cmd, "threads")) cmd_threads(B, nb);
    return 0;
}
