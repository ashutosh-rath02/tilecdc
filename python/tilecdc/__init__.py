"""TileCDC: fast content-defined chunking on CPU matrix units.

    import tilecdc

    c = tilecdc.Chunker()                 # 2 KiB / 8 KiB / 64 KiB
    lengths = c.chunk(data)               # chunk an in-memory buffer
    for off, length in c.iter_file("big.tar"):
        ...

Boundaries are identical on every CPU: AMX, AVX-512 and the portable scalar
path implement the same function, so chunks dedup across machines.
"""
import os

from ._tilecdc import MAX_AVG_SIZE, Chunker as _Chunker, available_impls, __version__

__all__ = ["Chunker", "available_impls", "chunk", "MAX_AVG_SIZE", "__version__"]


class Chunker(_Chunker):
    """Chunker(min_size=2048, avg_size=8192, max_size=65536, seed=0, norm=1, impl='auto')

    impl is 'auto' (fastest available), 'amx', 'avx512' or 'scalar'.
    The seed changes where boundaries fall; use the same seed everywhere
    data should deduplicate. avg_size may be at most MAX_AVG_SIZE (32 KiB).
    """

    def iter_file(self, file, with_data=False, buffer_size=None):
        """Yield (offset, length) or (offset, length, bytes) for each chunk of
        a path or a binary file object, reading it in large blocks."""
        close = False
        if isinstance(file, (str, bytes, os.PathLike)):
            file = open(file, "rb")
            close = True
        try:
            # the buffer must hold at least two maximum-size chunks to make progress
            size = max(buffer_size or 4 << 20, 16 * self.max_size if buffer_size is None else 2 * self.max_size)
            yield from self._iter_stream(file, with_data, size)
        finally:
            if close:
                file.close()

    def _iter_stream(self, f, with_data, bufsize):
        buf = bytearray()
        base = 0
        eof = False
        while True:
            if not eof and len(buf) < bufsize:
                block = f.read(bufsize - len(buf))
                if not block:
                    eof = True
                else:
                    buf += block
            if not buf:
                return
            view = memoryview(buf)
            used = 0
            # One C call chunks the whole buffer. A chunk that starts at s is
            # final once bytes up to s + max_size are visible (or at end of file),
            # because its cut never depends on later bytes.
            limit = len(buf) if eof else len(buf) - self.max_size
            for n in self.chunk(view):
                if used > limit or (not eof and used + n > len(buf)):
                    break
                if not eof and used + self.max_size > len(buf):
                    break
                if with_data:
                    yield base + used, n, bytes(view[used:used + n])
                else:
                    yield base + used, n
                used += n
            view.release()
            del buf[:used]
            base += used
            if eof and not buf:
                return


def chunk(data, **kwargs):
    """Chunk a bytes-like object; returns the list of chunk lengths."""
    return Chunker(**kwargs).chunk(data)
