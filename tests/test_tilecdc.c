/*
 * Correctness tests for TileCDC.
 *  1. An independent brute-force implementation written from the definition in
 *     tilecdc.h (no blocks, no bitmaps) is the oracle.
 *  2. scalar, avx512 and amx must equal the oracle on every tested buffer, for
 *     several parameter sets and seeds, including every length near min, avg
 *     and max, and data with no entropy at all.
 *  3. Streaming use (feeding at most max_size bytes at a time) must equal
 *     whole-buffer chunking.
 * usage: test_tilecdc [extra files...]
 */
#include "tilecdc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FAIL(...) do { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)

static uint64_t rs = 42;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

/* ---------------- oracle, straight from the definition ---------------- */
typedef struct { uint8_t sbox[256]; int8_t key[48]; int sb, lb; size_t min, avg, max; } oracle_t;
static uint64_t sm64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static void oracle_init(oracle_t *o, const tilecdc_params *p) {
    uint64_t s = p->seed ^ 0x54696C6543444321ull;
    for (int i = 0; i < 256; i++) o->sbox[i] = (uint8_t)i;
    for (int i = 255; i > 0; i--) { int j = (int)(sm64(&s) % (uint64_t)(i + 1)); uint8_t x = o->sbox[i]; o->sbox[i] = o->sbox[j]; o->sbox[j] = x; }
    for (int d = 0; d < 48; d++) { int v; do { v = (int)(sm64(&s) & 0xFF) - 128; } while (v == 0 || v == -128); o->key[d] = (int8_t)v; }
    int lg = 63 - __builtin_clzll(p->avg_size);
    o->sb = lg + p->norm; o->lb = lg - p->norm - 1;
    o->min = p->min_size; o->avg = p->avg_size; o->max = p->max_size;
}
static int allowed(const oracle_t *o, const uint8_t *x, size_t c, int bits) {
    int32_t h = 0;
    for (int d = 0; d < 48; d++) h += o->key[d] * (int32_t)o->sbox[x[c - 48 + d]];
    uint32_t m = (uint32_t)h * 0x9E3779B1u;
    return (m >> (32 - bits)) == 0;
}
static size_t oracle_cut(const oracle_t *o, const uint8_t *x, size_t n) {
    if (n <= o->min) return n;
    size_t end = n < o->max ? n : o->max, mid = n < o->avg ? n : o->avg;
    for (size_t c = o->min + 1; c <= mid; c++) if (allowed(o, x, c, o->sb)) return c;
    for (size_t c = mid + 1; c <= end; c++) if (allowed(o, x, c, o->lb)) return c;
    return end;
}

/* ------------------------------ helpers ------------------------------ */
static size_t chunk_oracle(const oracle_t *o, const uint8_t *x, size_t n, size_t *ends) {
    size_t k = 0, s = 0;
    while (s < n) { s += oracle_cut(o, x + s, n - s); ends[k++] = s; }
    return k;
}
static size_t chunk_stream(tilecdc *t, size_t max, const uint8_t *x, size_t n, size_t *ends) {
    /* feed windows of at most max bytes, like a reader with a max-size buffer */
    size_t k = 0, s = 0;
    while (s < n) { size_t win = n - s < max ? n - s : max; s += tilecdc_cut(t, x + s, win); ends[k++] = s; }
    return k;
}

static long checks;
static void compare_all(const char *what, const tilecdc_params *base, const uint8_t *x, size_t n, int use_oracle) {
    size_t cap = n / 64 + 16;
    size_t *ref = malloc(cap * sizeof(size_t)), *got = malloc(cap * sizeof(size_t));
    size_t nref;
    oracle_t o;
    oracle_init(&o, base);
    tilecdc_params p = *base;
    p.impl = TILECDC_IMPL_SCALAR;
    tilecdc *sc = tilecdc_new(&p);
    if (use_oracle) nref = chunk_oracle(&o, x, n, ref);
    else nref = tilecdc_chunk_all(sc, x, n, ref, cap);
    tilecdc_impl impls[3] = {TILECDC_IMPL_SCALAR, TILECDC_IMPL_AVX512, TILECDC_IMPL_AMX};
    for (int i = 0; i < 3; i++) {
        if (!tilecdc_impl_available(impls[i])) continue;
        p.impl = impls[i];
        tilecdc *t = tilecdc_new(&p);
        if (!t) FAIL("%s: cannot create %s", what, tilecdc_impl_name(impls[i]));
        size_t ng = tilecdc_chunk_all(t, x, n, got, cap);
        if (ng != nref || memcmp(ref, got, nref * sizeof(size_t)))
            FAIL("%s: %s differs (%zu vs %zu chunks) seed=%llu min=%zu avg=%zu max=%zu n=%zu", what, tilecdc_impl_name(impls[i]),
                 ng, nref, (unsigned long long)p.seed, p.min_size, p.avg_size, p.max_size, n);
        ng = chunk_stream(t, p.max_size, x, n, got);
        if (ng != nref || memcmp(ref, got, nref * sizeof(size_t))) FAIL("%s: %s streaming differs", what, tilecdc_impl_name(impls[i]));
        checks += 2;
        tilecdc_free(t);
    }
    tilecdc_free(sc);
    free(ref); free(got);
}

static uint8_t *make(int kind, size_t n) {
    uint8_t *x = malloc(n + 1);
    for (size_t i = 0; i < n; i++) {
        switch (kind) {
        case 0: x[i] = (uint8_t)rnd(); break;                                           /* random */
        case 1: x[i] = 0; break;                                                         /* zeros */
        case 2: x[i] = "ACGT"[rnd() & 3]; break;                                         /* DNA */
        case 3: x[i] = (uint8_t)("the quick brown fox jumps over the lazy dog "[i % 44]); break;  /* periodic text */
        case 4: x[i] = (uint8_t)(rnd() % 10 < 9 ? 'a' + rnd() % 26 : ' '); break;        /* word-like */
        default: x[i] = (uint8_t)((i / 4096) & 1 ? rnd() : 0); break;                    /* alternating regions */
        }
    }
    return x;
}

int main(int argc, char **argv) {
    printf("implementations available: scalar=1 avx512=%d amx=%d\n", tilecdc_impl_available(TILECDC_IMPL_AVX512), tilecdc_impl_available(TILECDC_IMPL_AMX));
    tilecdc_params sets[4];
    for (int i = 0; i < 4; i++) tilecdc_default_params(&sets[i]);
    sets[1].min_size = 512;  sets[1].avg_size = 1024;  sets[1].max_size = 8192;  sets[1].seed = 7;
    sets[2].min_size = 4096; sets[2].avg_size = 16384; sets[2].max_size = 131072; sets[2].seed = 123456789;
    sets[3].min_size = 64;   sets[3].avg_size = 256;   sets[3].max_size = 2048;  sets[3].seed = 99; sets[3].norm = 1;

    /* 1. oracle on all data kinds and parameter sets */
    for (int s = 0; s < 4; s++)
        for (int k = 0; k < 6; k++) {
            size_t n = 1 << 20;
            uint8_t *x = make(k, n);
            char what[64]; snprintf(what, sizeof what, "oracle set%d kind%d", s, k);
            compare_all(what, &sets[s], x, n, 1);
            free(x);
        }
    printf("ok: 4 parameter sets x 6 data kinds x 1 MiB equal the oracle (whole and streaming)\n");

    /* 2. every length around min/avg/max and every small length */
    uint8_t *x = make(0, 200000);
    for (int s = 0; s < 4; s++) {
        oracle_t o; oracle_init(&o, &sets[s]);
        tilecdc_params p = sets[s];
        tilecdc_impl impls[3] = {TILECDC_IMPL_SCALAR, TILECDC_IMPL_AVX512, TILECDC_IMPL_AMX};
        tilecdc *t[3];
        for (int i = 0; i < 3; i++) { p.impl = impls[i]; t[i] = tilecdc_impl_available(impls[i]) ? tilecdc_new(&p) : NULL; }
        size_t pts[] = {0, p.min_size, p.avg_size, p.max_size};
        for (int q = 0; q < 4; q++)
            for (long dn = -70; dn <= 70; dn++) {
                long n = (long)pts[q] + dn;
                if (n < 0 || n > 200000) continue;
                for (size_t off = 0; off < 3; off++) {          /* also misaligned starts */
                    size_t want = oracle_cut(&o, x + off * 17, (size_t)n);
                    for (int i = 0; i < 3; i++) if (t[i]) {
                        size_t got = tilecdc_cut(t[i], x + off * 17, (size_t)n);
                        if (got != want) FAIL("edge: %s set%d n=%ld off=%zu got %zu want %zu", tilecdc_impl_name(impls[i]), s, n, off, got, want);
                        checks++;
                    }
                }
            }
        for (int i = 0; i < 3; i++) if (t[i]) tilecdc_free(t[i]);
    }
    free(x);
    printf("ok: every length within 70 bytes of 0/min/avg/max equals the oracle, 3 alignments\n");

    /* 3. long random buffers, many seeds (implementations vs scalar) */
    for (int r = 0; r < 20; r++) {
        tilecdc_params p; tilecdc_default_params(&p);
        p.seed = rnd();
        size_t n = (8 << 20) + (rnd() % 100000);
        uint8_t *y = make(r % 6, n);
        compare_all("random seeds", &p, y, n, 0);
        free(y);
    }
    printf("ok: 20 seeds x ~8 MiB, all implementations identical\n");

    /* 4. extra files */
    for (int a = 1; a < argc; a++) {
        FILE *f = fopen(argv[a], "rb");
        if (!f) FAIL("open %s", argv[a]);
        fseek(f, 0, SEEK_END); size_t n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t *y = malloc(n + 1);
        if (fread(y, 1, n, f) != n) FAIL("read %s", argv[a]);
        fclose(f);
        tilecdc_params p; tilecdc_default_params(&p);
        compare_all(argv[a], &p, y, n, 0);
        free(y);
        printf("ok: %s (%zu bytes) identical across implementations\n", argv[a], n);
    }
    printf("ALL TESTS PASSED (%ld comparisons)\n", checks);
    return 0;
}
