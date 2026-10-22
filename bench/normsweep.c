/* dedup ratio vs mean chunk size for TileCDC normalization levels 0-3 (used to choose the default level) */
#include "tilecdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t fp64(const uint8_t *p, size_t n) {
    uint64_t h = 0x243F6A8885A308D3ull ^ n, k; size_t i = 0;
    for (; i + 8 <= n; i += 8) { memcpy(&k, p + i, 8); k *= 0x87c37b91114253d5ull; k = (k << 31) | (k >> 33); k *= 0x4cf5ad432745937full; h ^= k; h = ((h << 27) | (h >> 37)) * 5 + 0x52dce729; }
    for (; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33; return h ? h : 1;
}
int main(int argc, char **argv) {
    const char *name = argv[1];
    int nf = argc - 2; uint8_t **P = malloc(nf * sizeof *P); size_t *N = malloc(nf * sizeof *N), total = 0;
    for (int i = 0; i < nf; i++) { FILE *f = fopen(argv[i + 2], "rb"); fseek(f, 0, SEEK_END); N[i] = ftell(f); fseek(f, 0, SEEK_SET); P[i] = malloc(N[i]); if (fread(P[i], 1, N[i], f) != N[i]) return 1; fclose(f); total += N[i]; }
    size_t avgs[] = {4096, 8192, 16384, 32768};
    for (int norm = 0; norm <= 3; norm++) for (int a = 0; a < 4; a++) {
        tilecdc_params p; tilecdc_default_params(&p);
        p.avg_size = avgs[a]; p.min_size = avgs[a] / 4; p.max_size = avgs[a] * 8; p.norm = norm;
        tilecdc *t = tilecdc_new(&p); if (!t) continue;
        size_t cap = 1 << 24; uint64_t *set = calloc(cap, 8); size_t uniq = 0, chunks = 0;
        size_t *ends = malloc((total / 64 + 16) * sizeof(size_t));
        for (int i = 0; i < nf; i++) {
            size_t k = tilecdc_chunk_all(t, P[i], N[i], ends, total / 64 + 16), s = 0;
            for (size_t j = 0; j < k; j++) { size_t len = ends[j] - s; uint64_t h = fp64(P[i] + s, len); size_t x = (h * 0x9E3779B97F4A7C15ull >> 17) & (cap - 1);
                int fresh = 1; while (set[x]) { if (set[x] == h) { fresh = 0; break; } x = (x + 1) & (cap - 1); } if (fresh) { set[x] = h; uniq += len; } s = ends[j]; }
            chunks += k;
        }
        printf("%s,%d,%zu,%.3f,%.4f\n", name, norm, avgs[a] / 1024, (double)total / chunks / 1024, (double)total / uniq);
        free(set); free(ends); tilecdc_free(t);
    }
    return 0;
}
