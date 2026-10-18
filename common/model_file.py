# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Hany El Imam
"""
Model file of the SSTV ML denoiser ("SSTVDNN1"), the file that
src/sstv/sstv_dnn.c reads, on the computer and in the 1KO125 firmware of the
Xiegu X6100.  A format of our own, as simple as possible: a header, then the
numbers of every layer one after the other.  Little-endian:

    "SSTVDNN1"                     8 bytes, magic
    layers, flags                  2 x uint32; flags bit 0 = residual
                                   (output = input - network)
    then for every layer:
      cin, cout, k(=3), relu       4 x uint32
      weights                      cout*cin*3*3 float32, [cout][cin][ky][kx]
      bias                         cout float32

The same format is read in C by sstv_dnn_load() (src/sstv/sstv_dnn.c).
3_train/train.py writes it with write_model().
"""
import struct
import numpy as np


def write_model(path, layers, residual=True):
    """layers: list of (weight[cout][cin][3][3], bias[cout], relu)"""
    with open(path, "wb") as f:
        f.write(b"SSTVDNN1")
        f.write(struct.pack("<II", len(layers), 1 if residual else 0))
        for w, b, relu in layers:
            w = np.ascontiguousarray(w, dtype="<f4")
            b = np.ascontiguousarray(b, dtype="<f4")
            cout, cin, kh, kw = w.shape
            assert kh == 3 and kw == 3 and b.shape == (cout,)
            f.write(struct.pack("<IIII", cin, cout, 3, 1 if relu else 0))
            f.write(w.tobytes())
            f.write(b.tobytes())


def read_model(path):
    """Returns (layers, residual), layers as for write_model()"""
    with open(path, "rb") as f:
        assert f.read(8) == b"SSTVDNN1"
        n, flags = struct.unpack("<II", f.read(8))
        layers = []
        for _ in range(n):
            cin, cout, k, act = struct.unpack("<IIII", f.read(16))
            w = np.frombuffer(f.read(4 * cout * cin * 9), dtype="<f4").reshape(cout, cin, 3, 3)
            b = np.frombuffer(f.read(4 * cout), dtype="<f4")
            layers.append((w, b, act == 1))
        return layers, bool(flags & 1)
