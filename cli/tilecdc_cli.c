/* tilecdc: chunk files and print chunk boundaries or a summary.
 *   tilecdc [-s] [--avg N] [--seed N] [--impl amx|avx512|scalar] file...
 *   default output: one line per chunk "file offset length"
 *   -s: summary per file (chunks, average size, throughput, implementation)
 */
#include "tilecdc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    tilecdc_params p;
    tilecdc_default_params(&p);
    int summary = 0, nfiles = 0;
    char **files = calloc((size_t)argc, sizeof(char *));
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-s")) summary = 1;
        else if (!strcmp(argv[i], "--avg") && i + 1 < argc) { p.avg_size = strtoull(argv[++i], 0, 0); p.min_size = p.avg_size / 4; p.max_size = p.avg_size * 8; }
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) p.seed = strtoull(argv[++i], 0, 0);
        else if (!strcmp(argv[i], "--impl") && i + 1 < argc) {
            const char *s = argv[++i];
            p.impl = !strcmp(s, "amx") ? TILECDC_IMPL_AMX : !strcmp(s, "avx512") ? TILECDC_IMPL_AVX512 : !strcmp(s, "scalar") ? TILECDC_IMPL_SCALAR : TILECDC_IMPL_AUTO;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("usage: %s [-s] [--avg N] [--seed N] [--impl amx|avx512|scalar] file...\n", argv[0]);
            return 0;
        } else files[nfiles++] = argv[i];
    }
    tilecdc *t = tilecdc_new(&p);
    if (!t) { fprintf(stderr, "tilecdc: invalid parameters or implementation unavailable\n"); return 1; }
    size_t cap = 1 << 16, bufsz = p.max_size * 16;
    uint8_t *buf = malloc(bufsz);
    for (int f = 0; f < nfiles; f++) {
        FILE *fp = fopen(files[f], "rb");
        if (!fp) { perror(files[f]); continue; }
        size_t have = 0, base = 0, chunks = 0;
        int eof = 0;
        double t0 = now(), tc = 0;
        for (;;) {
            if (!eof && have < bufsz) {
                size_t r = fread(buf + have, 1, bufsz - have, fp);
                have += r;
                if (r == 0) eof = 1;
            }
            if (have == 0) break;
            size_t used = 0;
            /* cut while a full max-size window is available (or at end of file) */
            double c0 = now();
            while (have - used > 0 && (eof || have - used >= p.max_size)) {
                size_t len = tilecdc_cut(t, buf + used, have - used);
                if (!summary) printf("%s %zu %zu\n", files[f], base + used, len);
                used += len;
                chunks++;
            }
            tc += now() - c0;
            memmove(buf, buf + used, have - used);
            have -= used;
            base += used;
            if (eof && have == 0) break;
        }
        fclose(fp);
        (void)cap;
        if (summary)
            printf("%s: %zu bytes, %zu chunks, avg %.0f B, chunking %.2f GB/s (%s), total %.2f s\n", files[f], base, chunks,
                   chunks ? (double)base / chunks : 0, tc > 0 ? base / tc / 1e9 : 0, tilecdc_impl_name(tilecdc_get_impl(t)), now() - t0);
    }
    tilecdc_free(t);
    free(buf); free(files);
    return 0;
}
