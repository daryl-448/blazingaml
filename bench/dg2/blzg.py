"""Read a .blzg cache (SPEC §3.6, version 3: SPEC v3 P15) with numpy only.

Layout (src/io/blzg.hpp): a 64-byte header ("BLZG", u32 version, u64 m, u64 n, sha256[32] of the
CSV, u64 file size), then 64-byte aligned sections: src u32[m], dst u32[m], ts i32[m], label u8[m],
amount_paid f64[m], amount_received f64[m], pay_ccy u8[m], recv_ccy u8[m], format u8[m],
file_row u32[m], the string tables ccy_dict, format_dict, node_bank[n], node_account[n]
(u64 count, u64 offsets[count + 1], chars), and the dataset section: i64 epoch, u64 b_fv,
u64 b_test, sha256[32] of the dataset descriptor, then the string table [id, epoch date, schema,
split rule]. Arrays are read-only views of a memory map.
"""
import numpy as np

ARRAYS = [("src", np.uint32), ("dst", np.uint32), ("ts", np.int32), ("label", np.uint8),
          ("amount_paid", np.float64), ("amount_received", np.float64), ("pay_ccy", np.uint8),
          ("recv_ccy", np.uint8), ("format", np.uint8), ("file_row", np.uint32)]
TABLES = ["ccy_dict", "format_dict", "node_bank", "node_account", "dataset"]


class Blzg:
    def __init__(self, path):
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        if bytes(raw[:4]) != b"BLZG":
            raise ValueError(f"{path}: not a .blzg file")
        version = int(np.frombuffer(raw[4:8], np.uint32)[0])
        if version != 3:
            raise ValueError(f"{path}: .blzg version {version}; re-run `blz prepare --dataset ...` (this reader needs 3)")
        self.m = m = int(np.frombuffer(raw[8:16], np.uint64)[0])
        self.n = int(np.frombuffer(raw[16:24], np.uint64)[0])
        self.csv_sha256 = bytes(raw[24:56]).hex()
        if int(np.frombuffer(raw[56:64], np.uint64)[0]) != raw.shape[0]:
            raise ValueError(f"{path}: size mismatch (truncated?)")
        off = 64
        align = lambda x: (x + 63) // 64 * 64
        for name, dt in ARRAYS:
            off = align(off)
            size = np.dtype(dt).itemsize * m
            setattr(self, name, np.frombuffer(raw[off:off + size], dt))
            off += size
        self._tables = {}
        for name in TABLES:
            off = align(off)
            if name == "dataset":  # its fixed part, then its string table
                self.epoch, self.b_fv, self.b_test = (int(x) for x in np.frombuffer(raw[off:off + 24], "<i8"))
                self.descriptor_sha256 = bytes(raw[off + 24:off + 56]).hex()
                off = align(off + 56)
            count = int(np.frombuffer(raw[off:off + 8], np.uint64)[0])
            offsets = np.frombuffer(raw[off + 8:off + 8 + 8 * (count + 1)], np.uint64)
            chars = off + 8 + 8 * (count + 1)
            self._tables[name] = (raw, chars, offsets)
            off = chars + int(offsets[-1])
        if off != raw.shape[0]:
            raise ValueError(f"{path}: trailing bytes")

    def strings(self, name):
        """A string table as a list of str."""
        raw, chars, offsets = self._tables[name]
        blob = bytes(raw[chars:chars + int(offsets[-1])]).decode()
        o = offsets.tolist()
        return [blob[o[i]:o[i + 1]] for i in range(len(o) - 1)]

    def dataset(self):
        """The dataset section: id, epoch (unix seconds) and its date, schema, split rule."""
        id_, date, schema, split = self.strings("dataset")
        return {"id": id_, "epoch": self.epoch, "epoch_date": date, "schema": schema, "split": split,
                "b_fv": self.b_fv, "b_test": self.b_test, "descriptor_sha256": self.descriptor_sha256}

    def splits(self):
        """{fit, val, test, train}: eid ranges [a, b), as `blz prepare` stored them (SPEC v3 §6.3)."""
        return {"fit": (0, self.b_fv), "val": (self.b_fv, self.b_test), "test": (self.b_test, self.m),
                "train": (0, self.b_test)}
