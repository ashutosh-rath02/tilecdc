/*
 * TileCDC: content-defined chunking on CPU matrix units.
 *
 * TileCDC computes a windowed fingerprint at every byte offset with Intel AMX
 * tile instructions: a tile load whose row stride (16 bytes) is shorter than
 * its rows (64 bytes) forms a Hankel matrix of the byte stream without
 * copying, and one TDPBUSD against a constant Toeplitz key tile yields 256
 * window hashes. Boundaries follow FastCDC-style normalized chunking.
 *
 * The chunking function is defined independently of the hardware (below), and
 * the AMX, AVX-512 and scalar implementations return identical boundaries.
 *
 * Definition (seed-dependent constants KEY[48] in [-127,127]\{0} and a byte
 * permutation SBOX, both derived from the seed with splitmix64):
 *   y[j]  = SBOX[x[j]]
 *   h(i)  = sum_{d=0}^{47} KEY[d] * y[i+d]               (exact, int32)
 *   m(i)  = (uint32) h(i) * 0x9E3779B1                     (mod 2^32)
 *   a cut is allowed after byte c-1 (chunk end c) under b bits when the top b
 *   bits of m(c-48) are zero.
 *   For a buffer of n bytes: if n <= min the chunk is the whole buffer;
 *   otherwise the chunk ends at the first allowed c in (min, min(n,avg)] under
 *   the strict bits, else the first in (min(n,avg), min(n,max)] under the
 *   loose bits, else at min(n,max).
 *   strict bits = floor(log2(avg)) + norm, loose = floor(log2(avg)) - norm - 1.
 */
#ifndef TILECDC_H
#define TILECDC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TILECDC_WINDOW 48
/* Largest supported average chunk size. The window hash carries about 17 bits
   of entropy, so masks of 17+ bits quantize: at a 64 KiB target the mean chunk
   size varied from 55 to 74 KiB across seeds (32 KiB: 32 to 37 KiB). */
#define TILECDC_MAX_AVG 32768
#define TILECDC_VERSION "0.1.2"

typedef enum { TILECDC_IMPL_AUTO = 0, TILECDC_IMPL_SCALAR = 1, TILECDC_IMPL_AVX512 = 2, TILECDC_IMPL_AMX = 3 } tilecdc_impl;

typedef struct tilecdc tilecdc;

typedef struct tilecdc_params {
    size_t min_size;   /* default 2048; must be >= 64 (one 48-byte window plus slack) */
    size_t avg_size;   /* default 8192; at most TILECDC_MAX_AVG */
    size_t max_size;   /* default 65536 */
    uint64_t seed;     /* default 0; changes KEY and SBOX */
    int norm;          /* normalization level 0..4, default 1 */
    tilecdc_impl impl; /* default AUTO: AMX if usable, else AVX-512, else scalar */
} tilecdc_params;

/* Fill p with defaults. */
void tilecdc_default_params(tilecdc_params *p);

/* Create a chunker; NULL params means defaults. Returns NULL with errno
   EINVAL for invalid parameters, ENOTSUP when a forced implementation is
   unavailable. Thread safety: a chunker holds no mutable state and may be
   used from several threads at once. */
tilecdc *tilecdc_new(const tilecdc_params *params);
void tilecdc_free(tilecdc *t);

/* Length of the first chunk of data[0..len). In a stream, pass at least
   max_size bytes unless the stream has ended (the last chunk may be short). */
size_t tilecdc_cut(tilecdc *t, const uint8_t *data, size_t len);

/* Chunk a whole buffer: writes chunk end offsets into ends (capacity cap)
   and returns the number of chunks (which may exceed cap: only cap are written). */
size_t tilecdc_chunk_all(tilecdc *t, const uint8_t *data, size_t len, size_t *ends, size_t cap);

/* Which implementation this chunker uses, and what the CPU supports. */
tilecdc_impl tilecdc_get_impl(const tilecdc *t);
const char *tilecdc_impl_name(tilecdc_impl impl);
int tilecdc_impl_available(tilecdc_impl impl);

#ifdef __cplusplus
}
#endif
#endif
