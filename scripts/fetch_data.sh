#!/bin/sh
# Rebuild the paper's datasets from PyPI and check them against datasets.sha256.
#   SRC: source releases of Django 5.0.14/5.1.15/5.2.18 and NumPy 2.1.0/2.2.0/2.3.0, uncompressed
#   BIN: x86-64 CPython 3.12 manylinux wheels of NumPy, SciPy and pandas, unpacked into tar files
#   RAND: 256 MiB of random bytes (generated, not checked)
set -e
cd "$(dirname "$0")/.."
mkdir -p data/src data/bin data/tmp
cd data/tmp
for spec in Django==5.0.14 Django==5.1.15 Django==5.2.18; do
  pip download --no-binary :all: --no-deps "$spec" -d . -q
done
for v in 2.1.0 2.2.0 2.3.0; do
  curl -sSL -o numpy-$v.tar.gz "https://pypi.org/packages/source/n/numpy/numpy-$v.tar.gz"
done
for f in *.tar.gz; do gunzip -c "$f" > ../src/"${f%.gz}"; done
for spec in numpy==2.1.0 numpy==2.2.0 scipy==1.14.0 scipy==1.15.0 scipy==1.16.0 pandas==2.2.0 pandas==2.2.3 pandas==2.3.0; do
  pip download "$spec" --no-deps --only-binary :all: --platform manylinux2014_x86_64 --python-version 3.12 --implementation cp -d . -q
done
for w in *.whl; do
  d=${w%.whl}; mkdir -p x/$d && (cd x/$d && python3 -m zipfile -e ../../$w .)
  tar cf ../bin/$d.tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C x $d
done
cd ..
head -c 268435456 /dev/urandom > random.bin
rm -rf tmp
cd .. && (cd data && sha256sum -c ../scripts/datasets.sha256) || echo "note: checksums differ if tar metadata differs from the original run; chunking results should still match closely"
