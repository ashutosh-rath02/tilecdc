"""python -m tilecdc [-s] [--avg N] [--seed N] [--impl X] FILE...

Prints "file offset length" per chunk, or a per-file summary with -s.
"""
import argparse
import sys
import time

from . import Chunker, available_impls


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tilecdc", description="Content-defined chunking on CPU matrix units")
    ap.add_argument("files", nargs="+")
    ap.add_argument("-s", "--summary", action="store_true", help="print a summary per file instead of every chunk")
    ap.add_argument("--avg", type=int, default=8192, help="average chunk size (min = avg/4, max = avg*8)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--impl", default="auto", choices=["auto", "amx", "avx512", "scalar"])
    a = ap.parse_args(argv)
    c = Chunker(min_size=a.avg // 4, avg_size=a.avg, max_size=a.avg * 8, seed=a.seed, impl=a.impl)
    out = sys.stdout
    for path in a.files:
        t0 = time.perf_counter()
        n = total = 0
        for off, length in c.iter_file(path):
            n += 1
            total += length
            if not a.summary:
                out.write(f"{path} {off} {length}\n")
        dt = time.perf_counter() - t0
        if a.summary:
            avg = total / n if n else 0
            out.write(f"{path}: {total} bytes, {n} chunks, avg {avg:.0f} B, {total / dt / 1e9:.2f} GB/s incl. I/O ({c.impl}; available: {', '.join(available_impls())})\n")


if __name__ == "__main__":
    main()
