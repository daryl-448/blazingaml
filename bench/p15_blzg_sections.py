"""Compare two .blzg files section by section (SPEC v3 P15: v3 keeps every v2 column section).

  python3 bench/p15_blzg_sections.py <a.blzg> <b.blzg>

Walks both files' layouts (src/io/blzg.hpp) and compares the header's m, n and CSV SHA-256, then
the bytes of each column section and string table. A version 3 file's dataset section, which a
version 2 file lacks, is printed, not compared. Exits 1 if anything differs.
"""
import struct
import sys

ARRAYS = [("src", 4), ("dst", 4), ("ts", 4), ("label", 1), ("amount_paid", 8), ("amount_received", 8),
          ("pay_ccy", 1), ("recv_ccy", 1), ("format", 1), ("file_row", 4)]
TABLES = ["ccy_dict", "format_dict", "node_bank", "node_account"]


def align(x):
    return (x + 63) // 64 * 64


def table(raw, off):
    """A string table at the next aligned offset: its bytes, its end, and its strings."""
    off = align(off)
    count = struct.unpack_from("<Q", raw, off)[0]
    offsets = struct.unpack_from(f"<{count + 1}Q", raw, off + 8)
    chars = off + 8 + 8 * (count + 1)
    end = chars + offsets[-1]
    return raw[off:end], end, lambda: [raw[chars + offsets[i]:chars + offsets[i + 1]].decode() for i in range(count)]


def sections(path):
    raw = open(path, "rb").read()
    magic, version, m, n = struct.unpack_from("<4sIQQ", raw, 0)
    assert magic == b"BLZG", path
    out = {"header m, n, csv sha256": raw[8:56]}
    off = 64
    for name, size in ARRAYS:
        off = align(off)
        out[name] = raw[off:off + size * m]
        off += size * m
    for name in TABLES:
        out[name], off, _ = table(raw, off)
    extra = None
    if version >= 3:
        off = align(off)
        epoch, b_fv, b_test = struct.unpack_from("<qQQ", raw, off)
        sha = raw[off + 24:off + 56].hex()
        _, off, strings = table(raw, off + 56)
        extra = {"epoch": epoch, "b_fv": b_fv, "b_test": b_test, "descriptor_sha256": sha,
                 "id, epoch date, schema, split": strings()}
    assert off == len(raw), f"{path}: {len(raw) - off} trailing bytes"
    return version, out, extra


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    (va, a, xa), (vb, b, xb) = sections(sys.argv[1]), sections(sys.argv[2])
    print(f"versions {va} and {vb}")
    bad = 0
    for name in a:
        same = a[name] == b[name]
        bad += not same
        print(f"{name:24s} {len(a[name]):>11d} B  {'equal' if same else 'DIFFERENT'}")
    for path, x in ((sys.argv[1], xa), (sys.argv[2], xb)):
        if x:
            print(f"dataset section of {path}: {x}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
