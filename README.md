# TileCDC

**Content-defined chunking on CPU matrix units.** TileCDC splits data into deduplication-friendly chunks 2.2 to 2.6 times faster than FastCDC on one core, with a deduplication ratio within 1% of FastCDC's. It computes a window hash at every byte offset at once on Intel AMX, with an AVX-512 path for CPUs without AMX and a portable C path. All three return bit-identical boundaries.

```python
import tilecdc

chunker = tilecdc.Chunker()                       # 2 KiB min, 8 KiB average, 64 KiB max
lengths = chunker.chunk(data)                     # chunk a bytes-like object
for offset, length in chunker.iter_file("backup.tar"):
    ...
```

```c
#include "tilecdc.h"
tilecdc *t = tilecdc_new(NULL);                   /* defaults; picks AMX, AVX-512 or scalar */
size_t first = tilecdc_cut(t, data, len);         /* FastCDC-style: length of the first chunk */
```

## The idea

Rolling hashes such as Gear (used by FastCDC) compute the hash at byte *i* from the hash at byte *i - 1*. That dependency chain caps a scalar chunker near 2 GB/s.

TileCDC instead computes an independent 48-byte window hash at every offset. The hash is linear in the window, so all of the hashes together form a matrix product: the Hankel matrix of the byte stream times a Toeplitz key matrix. The Hankel matrix costs nothing to build. An AMX tile load with a 16-byte row stride and 64-byte rows makes row *r* of the tile the window starting at byte 16*r*, with no copying. One `TDPBUSD` instruction then yields 256 window hashes.

Before the multiply, each byte goes through a random permutation; after it, one multiplication mixes the result. These make the linear hash a well-distributed boundary test. Boundaries then follow FastCDC's rules: min-size skipping, normalized chunking and a max cut.

The exact function is specified in [include/tilecdc.h](include/tilecdc.h). The paper is [paper/tilecdc.pdf](paper/tilecdc.pdf).

## Results

Medians of 5 runs on one vCPU of a 2.1 GHz Intel Xeon VM (AMX, AVX-512) at 2/8/64 KiB; run-to-run spread on this shared VM reached 34%. SRC is the source releases of Django and NumPy (421 MB). BIN is unpacked NumPy, SciPy and pandas wheels (632 MB). Raw data is in [results/](results/).

| GB/s, one core | SRC | BIN | random |
|---|---:|---:|---:|
| fastcdc crate 3.2 (Rust, v2020, target-cpu=native) | 1.69 | 1.84 | 1.84 |
| FastCDC algorithm in C (one byte per step, -march=native) | 1.85 | 1.88 | 2.05 |
| **TileCDC, AVX-512** | 2.40 | 2.54 | 2.74 |
| **TileCDC, AMX** | **4.65** | **4.82** | **4.52** |

- **Deduplication:** at equal mean chunk size (8.8 to 37 KiB), TileCDC's ratio is within -0.9% to +0.3% of FastCDC's, 0.21% lower on average. See Figure 3 of the paper.
- **Edits:** after random insertions or deletions, TileCDC at a matched mean chunk size keeps the same share of the file in previously stored chunks as FastCDC (within 0.4 points).
- **Two threads:** 8.3 GB/s against 3.3 GB/s for FastCDC (C).
- **Bit-identical boundaries:** AMX, AVX-512 and scalar agree with an independent oracle in every test: 17,994 comparisons in `make test`, 18,078 including the dataset files.

Numbers on your machine will differ. `scripts/fetch_data.sh` rebuilds the datasets and `bench/run_all.sh` reproduces every number.

## When to use it, and when not

- **It speeds up the chunking step only.** Deduplication also fingerprints every chunk (SHA-256 or BLAKE3, about 1 to 3 GB/s per core), so an end-to-end pipeline gains less than the chunker alone.
- **Average chunk size is capped at 32 KiB.** The window hash carries about 17 bits of entropy; beyond 32 KiB the mean chunk size starts to depend on the seed (55 to 74 KiB for a 64 KiB target across six seeds).
- **AMX is only on Intel Xeon** (Sapphire Rapids and later). It is available in the major clouds, but not on laptops, AMD or most ARM servers. The AVX-512 path, 1.3x FastCDC, covers recent AMD and Intel servers.
- **Tile registers:** TileCDC saves and restores the calling thread's AMX tile configuration, but not tile contents. Do not call it while holding live data in tile registers.
- **Hashless chunkers are faster still.** VectorCDC and SeqCDC report higher throughput, at some cost in deduplication on some data. TileCDC keeps a hash-based boundary test. We have not compared against SS-CDC, RapidCDC or SIMD Gear variants.
- **Boundaries differ from FastCDC.** Switching an existing repository to TileCDC splits deduplication between old and new data, as any chunker change does.
- **Not a keyed or secret chunker.** The seed changes boundaries, but TileCDC makes no claim to resist chunk-length fingerprinting attacks.

## Build

```sh
make            # build/libtilecdc.{a,so}, build/tilecdc (CLI), tests, benchmarks
make test       # correctness suite
./build/tilecdc -s file...        # chunk summary;  without -s: "file offset length" per chunk

cd python && python3 -m build     # wheel and sdist (x86-64 Linux)
pip install dist/tilecdc-*.tar.gz
python -m tilecdc -s file...
```

Requires x86-64 Linux and GCC 11+ or Clang 14+. AMX needs Linux 5.16+ and a Sapphire Rapids or newer Xeon. The AVX-512 path needs VBMI and VNNI (Ice Lake, Zen 4 and later). Everything else falls back to scalar code.

## API

| C | Python | |
|---|---|---|
| `tilecdc_new(&params)` | `Chunker(min_size, avg_size, max_size, seed, norm, impl)` | create |
| `tilecdc_cut(t, p, n)` | `c.cut(buf)` | length of the first chunk |
| `tilecdc_chunk_all(t, p, n, ends, cap)` | `c.chunk(buf)` | all chunks of a buffer |
| | `c.iter_file(path, with_data=False)` | stream a file |
| `tilecdc_get_impl(t)` | `c.impl`, `tilecdc.available_impls()` | implementation in use |

## Layout

```
include/tilecdc.h     public API and the normative definition
src/tilecdc.c         AMX, AVX-512 and scalar implementations
tests/                oracle-checked correctness suite
cli/                  tilecdc command-line tool
python/               CPython package (tilecdc)
bench/                benchmarks, FastCDC reference, Rust fastcdc baseline, run_all.sh
results/              raw benchmark output used in the paper
scripts/              dataset download (with checksums)
.github/workflows/    CI, and manylinux wheel builds + PyPI publishing on tags
paper/                LaTeX source, figure generator, PDF
```

## Citation

```
Ashutosh Rath. TileCDC: Content-Defined Chunking on CPU Matrix Units. 2026.
https://github.com/ashutosh-rath02/tilecdc
```

## License

Apache 2.0. Developed with the assistance of Claude (Anthropic); see the paper's acknowledgments.
