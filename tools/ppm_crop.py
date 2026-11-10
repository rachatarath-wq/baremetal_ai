#!/usr/bin/env python3
"""ppm_crop.py — crop a top-left region of a QEMU screendump (P6 PPM) and
re-save as PPM, so the (small) framebuffer text can be zoomed into for review.

Usage:
    tools/venv/bin/python tools/ppm_crop.py in.ppm out.ppm <width> <height>
"""
import sys


def crop(inpath, outpath, w, h):
    data = open(inpath, "rb").read()

    # parse the P6 header: P6 <width> <height> <maxval> + one whitespace byte
    toks = []
    i = 0
    ws = b" \n\r\t"
    while len(toks) < 4:
        while i < len(data) and data[i:i + 1] in ws:
            i += 1
        j = i
        while j < len(data) and data[j:j + 1] not in ws:
            j += 1
        toks.append(data[i:j].decode())
        i = j
    magic, width, height, maxval = toks[0], int(toks[1]), int(toks[2]), int(toks[3])
    while i < len(data) and data[i:i + 1] in ws:
        i += 1
    raw = data[i:]

    bpp = 3
    stride = width * bpp
    w = min(w, width)
    h = min(h, height)
    out = bytearray()
    for y in range(h):
        out += raw[y * stride: y * stride + w * bpp]

    header = f"P6\n{w} {h}\n{maxval}\n".encode()
    open(outpath, "wb").write(header + out)
    print(f"{w}x{h} cropped from {width}x{height} -> {outpath}")


if __name__ == "__main__":
    crop(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
