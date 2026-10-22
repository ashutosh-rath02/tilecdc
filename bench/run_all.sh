#!/bin/sh
# Reproduce every number in the paper (needs data/ from scripts/fetch_data.sh).
# Output: results/*.csv. Single-thread runs are pinned to CPU 0.
set -e
cd "$(dirname "$0")/.."
make -s all
(cd bench/rust_fastcdc && cargo build --release -q)
B="taskset -c 0 ./build/bench"
R="taskset -c 0 ./bench/rust_fastcdc/target/release/fastcdc_bench"
{ lscpu; echo; free -g; echo; uname -a; echo; gcc --version | head -1; rustc --version; } > results/machine.txt
./build/test_tilecdc data/src/*.tar data/bin/*.tar > results/tests.txt
$B breakdown > results/breakdown.csv
: > results/compare_runs.csv; : > results/compare_rust_runs.csv
for run in 1 2 3 4 5; do   # five independent runs, each best of 5 (scalar: 1)
  { $B compare SRC data/src/*.tar; $B compare BIN data/bin/*.tar; $B compare RAND data/random.bin; } | sed "s/^/$run,/" >> results/compare_runs.csv
  { echo "$run,SRC,$($R 2048 8192 65536 data/src/*.tar)"; echo "$run,BIN,$($R 2048 8192 65536 data/bin/*.tar)"; echo "$run,RAND,$($R 2048 8192 65536 data/random.bin)"; } >> results/compare_rust_runs.csv
done
{ $B sweep data/src/*.tar | sed 's/^/SRC,/'; $B sweep data/bin/*.tar | sed 's/^/BIN,/'; } > results/sweep.csv
for a in 4096 8192 16384 32768; do
  echo "SRC,$((a/1024)),$($R $((a/4)) $a $((a*8)) data/src/*.tar)"; echo "BIN,$((a/1024)),$($R $((a/4)) $a $((a*8)) data/bin/*.tar)"
done > results/sweep_rust.csv
{ echo "dataset,norm,avg_target_kib,avg_kib,dedup"; ./build/normsweep SRC data/src/*.tar; ./build/normsweep BIN data/bin/*.tar; } > results/normsweep.csv
{ $B edits data/src/numpy-2.3.0.tar | sed 's/^/numpy-src,/'; $B edits data/bin/scipy-1.16.0-*.tar | sed 's/^/scipy-bin,/'; } > results/edits.csv
taskset -c 0 ./build/seedvar > results/seedvar.csv
./build/bench threads data/src/numpy-2.2.0.tar data/src/numpy-2.3.0.tar > results/threads.csv
echo done
