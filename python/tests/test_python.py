"""Tests for the Python package: identical boundaries across implementations,
streaming equals in-memory chunking, and argument validation."""
import io
import os
import random

import pytest

import tilecdc

DATA = random.Random(1).randbytes(6 << 20) + b"\x00" * 200_000 + b"abc" * 100_000


def test_available():
    impls = tilecdc.available_impls()
    assert "scalar" in impls


@pytest.mark.parametrize("impl", ["avx512", "amx"])
def test_impls_identical(impl):
    if impl not in tilecdc.available_impls():
        pytest.skip(f"{impl} not available")
    ref = tilecdc.Chunker(impl="scalar").chunk(DATA[: 1 << 20])
    assert tilecdc.Chunker(impl=impl).chunk(DATA[: 1 << 20]) == ref


def test_cover_and_bounds():
    c = tilecdc.Chunker()
    lengths = c.chunk(DATA)
    assert sum(lengths) == len(DATA)
    assert all(l <= c.max_size for l in lengths)
    assert all(l > c.min_size for l in lengths[:-1])


def test_stream_equals_memory():
    c = tilecdc.Chunker()
    mem = c.chunk(DATA)
    for bs in (c.max_size, c.max_size + 1, 1 << 20, 3 << 20):
        streamed = [l for _, l in c._iter_stream(io.BytesIO(DATA), False, bs)]
        assert streamed == mem, bs


def test_iter_file_with_data(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(DATA)
    c = tilecdc.Chunker(avg_size=16384, min_size=4096, max_size=131072)
    joined = b"".join(d for _, _, d in c.iter_file(str(p), with_data=True))
    assert joined == DATA


def test_seed_changes_boundaries():
    a = tilecdc.Chunker(seed=1).chunk(DATA[: 1 << 20])
    b = tilecdc.Chunker(seed=2).chunk(DATA[: 1 << 20])
    assert a != b


def test_small_inputs():
    c = tilecdc.Chunker()
    assert c.chunk(b"") == []
    assert c.chunk(b"x") == [1]
    assert c.chunk(b"y" * 2048) == [2048]


def test_bad_args():
    with pytest.raises(ValueError):
        tilecdc.Chunker(min_size=10)
    with pytest.raises(ValueError):
        tilecdc.Chunker(impl="gpu")


def test_uninitialised_and_failed_reinit_do_not_crash():
    c = tilecdc.Chunker.__new__(tilecdc.Chunker)
    with pytest.raises(RuntimeError):
        c.chunk(b"x")
    d = tilecdc.Chunker()
    before = d.chunk(DATA[:100000])
    with pytest.raises(ValueError):
        d.__init__(min_size=10)
    assert d.chunk(DATA[:100000]) == before   # old chunker still intact


def test_small_buffer_size_terminates():
    c = tilecdc.Chunker()
    out = [l for _, l in c.iter_file(io.BytesIO(DATA), buffer_size=1000)]
    assert out == c.chunk(DATA)


def test_avg_cap_and_seed_validation():
    with pytest.raises(ValueError):
        tilecdc.Chunker(min_size=16384, avg_size=65536, max_size=524288)
    with pytest.raises(ValueError):
        tilecdc.Chunker(seed=-1)
    assert tilecdc.Chunker(seed=2**64 - 1).chunk(b"a" * 5000)


def test_amx_tile_config_loaded_from_readonly_memory():
    """Regression test for 0.1.0, whose manylinux wheel crashed on AMX CPUs: the
    compiler dropped the stores that built the tile configuration on the stack.
    The configuration must be loaded from a constant (RIP-relative operand).
    Runs anywhere objdump exists, so CI catches it without AMX hardware."""
    import shutil
    import subprocess

    from tilecdc import _tilecdc

    import os

    objdump = shutil.which("objdump")
    if not objdump:
        if os.environ.get("CIBUILDWHEEL"):
            pytest.fail("objdump missing in the wheel build environment; this check must run before publishing")
        pytest.skip("objdump not available")
    asm = subprocess.run([objdump, "-d", "--no-show-raw-insn", _tilecdc.__file__], capture_output=True, text=True).stdout
    loads, inside = [], False
    for line in asm.splitlines():
        if line.endswith(">:"):
            inside = "<amx_configure" in line
        elif inside and "ldtilecfg" in line:
            loads.append(line.strip())
    assert loads, "amx_configure not found in the extension"
    assert all("(%rip)" in l for l in loads), loads
