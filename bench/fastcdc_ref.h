/* FastCDC 2020 reference (Xia et al., TPDS 2020): Gear hash with normalized
   chunking (level 2) and min-size skipping, one byte per step. For an 8 KiB
   average it uses the paper's masks (MaskS 0x0000d9f003530000, 15 bits;
   MaskL 0x0000d90003530000, 11 bits); other sizes spread the bits over 16..47. */
#ifndef FASTCDC_REF_H
#define FASTCDC_REF_H
#include <stddef.h>
#include <stdint.h>

typedef struct { uint64_t gear[256]; uint64_t ms, ml; size_t min, avg, max; } fastcdc_t;

static uint64_t fcdc_spread_mask(int n) {
    uint64_t m = 0;
    for (int i = 0; i < n; i++) m |= 1ull << (16 + (i * 32) / n);
    return m;
}
static void fastcdc_init(fastcdc_t *f, size_t min, size_t avg, size_t max, int norm) {
    uint64_t s = 0x2545F4914F6CDD1Dull;
    for (int i = 0; i < 256; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; f->gear[i] = s; }
    int lg = 63 - __builtin_clzll(avg);
    f->ms = fcdc_spread_mask(lg + norm);
    f->ml = fcdc_spread_mask(lg - norm);
    if (lg == 13 && norm == 2) { f->ms = 0x0000d9f003530000ull; f->ml = 0x0000d90003530000ull; }
    f->min = min; f->avg = avg; f->max = max;
}
static size_t fastcdc_cut(const fastcdc_t *f, const uint8_t *p, size_t n) {
    if (n <= f->min) return n;
    size_t end = n < f->max ? n : f->max, mid = n < f->avg ? n : f->avg;
    uint64_t h = 0;
    size_t i = f->min;
    for (; i < mid; i++) { h = (h << 1) + f->gear[p[i]]; if (!(h & f->ms)) return i + 1; }
    for (; i < end; i++) { h = (h << 1) + f->gear[p[i]]; if (!(h & f->ml)) return i + 1; }
    return end;
}
#endif
