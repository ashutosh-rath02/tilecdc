/* Mean chunk size on 1 GiB of pseudo-random data for 6 seeds per target average,
   including 64 KiB (above TILECDC_MAX_AVG, measured through the internal API by
   lifting the cap) to document why the cap exists. */
#include "../src/tilecdc.c"
#include <stdio.h>
static tilecdc *mk(size_t avg, uint64_t seed) {   /* like tilecdc_new without the size cap */
    tilecdc_params p; tilecdc_default_params(&p);
    p.avg_size = avg; p.min_size = avg / 4; p.max_size = avg * 8; p.seed = seed;
    int lg = 63 - __builtin_clzll(avg);
    tilecdc *t = aligned_alloc(64, (sizeof(tilecdc) + 63) & ~(size_t)63);
    memset(t, 0, sizeof *t);
    t->min = p.min_size; t->avg = p.avg_size; t->max = p.max_size;
    t->smask = 0xFFFFFFFFu << (32 - (lg + p.norm)); t->lmask = 0xFFFFFFFFu << (32 - (lg - p.norm - 1));
    t->impl = has_amx() ? TILECDC_IMPL_AMX : has_avx512() ? TILECDC_IMPL_AVX512 : TILECDC_IMPL_SCALAR;
    derive(t, seed);
    return t;
}
int main(void) {
    size_t n = (size_t)1 << 30;
    uint8_t *p = malloc(n);
    uint64_t s = 7;
    for (size_t i = 0; i < n; i += 8) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; memcpy(p + i, &s, 8); }
    size_t avgs[] = {8192, 16384, 32768, 65536};
    size_t *e = malloc(n / 1024 * sizeof(size_t));
    printf("avg_target_kib,seed,mean_kib\n");
    for (int a = 0; a < 4; a++)
        for (uint64_t seed = 0; seed < 6; seed++) {
            tilecdc *t = mk(avgs[a], seed * 977 + 1);
            size_t k = tilecdc_chunk_all(t, p, n, e, n / 1024);
            printf("%zu,%llu,%.2f\n", avgs[a] / 1024, (unsigned long long)(seed * 977 + 1), (double)n / k / 1024);
            free(t);
        }
    return 0;
}
