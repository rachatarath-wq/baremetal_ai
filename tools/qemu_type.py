#!/usr/bin/env python3
"""
qemu_type.py — type Thai (and ASCII) text into a running QEMU guest via QMP
`input-send-event`, then optionally capture the framebuffer.

The guest app (run.c) maps US keyboard keys through the Kedmanee layout when in
THAI mode (toggled with F1), so this script reverse-maps Thai code points back
to the US keys a real Kedmanee keyboard would produce.  ASCII characters must be
typed in ENGLISH mode, so `--th` / `--en` segments are processed in order and an
F1 is emitted automatically whenever the mode changes.

Usage:
    tools/venv/bin/python tools/qemu_type.py /tmp/qemu_bm.sock \
        --th "พ่อเราชื่ออะไร" --shot /tmp/shot.ppm

    # mixed-mode (the ':' and '=' must be typed in EN mode):
    tools/venv/bin/python tools/qemu_type.py /tmp/qemu_bm.sock \
        --th "จำไว้" --en ": " --th "ดาวพุธมีสีอะไร" --en " = " --th "สีเหลือง" \
        --shot /tmp/shot.ppm
"""
import argparse
import json
import os
import re
import socket
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_kedmanee():
    src = open(os.path.join(ROOT, "run.c"), encoding="utf-8").read()
    m = re.search(r"static const UINT32 gKedmanee\[128\] = \{(.*?)\};", src, re.S)
    if not m:
        sys.exit("gKedmanee table not found in run.c")
    body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)  # drop the /* 0xNN */ comments
    vals = [int(t, 16) for t in re.findall(r"0x[0-9A-Fa-f]+", body)]
    assert len(vals) == 128, f"expected 128 entries, got {len(vals)}"
    return vals


def reverse_map(ked):
    rev = {}
    for i, cp in enumerate(ked):
        if cp and cp not in rev:
            rev[cp] = i
    return rev


def us_layout():
    """ASCII code -> (qcode, needs_shift) for a US keyboard."""
    m = {}
    for c in range(ord("a"), ord("z") + 1):
        m[c] = (chr(c), False)
    for c in range(ord("A"), ord("Z") + 1):
        m[c] = (chr(c + 32), True)
    for c in range(ord("0"), ord("9") + 1):
        m[c] = (chr(c), False)
    m.update({
        0x20: ("spc", False), 0x21: ("1", True), 0x22: ("apostrophe", True),
        0x23: ("3", True), 0x24: ("4", True), 0x25: ("5", True), 0x26: ("7", True),
        0x27: ("apostrophe", False), 0x28: ("9", True), 0x29: ("0", True),
        0x2A: ("8", True), 0x2B: ("equal", True), 0x2C: ("comma", False),
        0x2D: ("minus", False), 0x2E: ("dot", False), 0x2F: ("slash", False),
        0x3A: ("semicolon", True), 0x3B: ("semicolon", False), 0x3C: ("comma", True),
        0x3D: ("equal", False), 0x3E: ("dot", True), 0x3F: ("slash", True),
        0x40: ("2", True), 0x5B: ("bracket_left", False), 0x5C: ("backslash", False),
        0x5D: ("bracket_right", False), 0x5E: ("6", True), 0x5F: ("minus", True),
        0x60: ("grave_accent", False), 0x7B: ("bracket_left", True),
        0x7C: ("backslash", True), 0x7D: ("bracket_right", True), 0x7E: ("grave_accent", True),
    })
    return m


def text_to_keys(text, mode, rev, layout):
    """Translate `text` into a list of (qcode, shift) key presses for `mode`."""
    keys = []
    for ch in text:
        cp = ord(ch)
        if mode == "th":
            idx = rev.get(cp)
            if idx is None:
                sys.exit(f"no Kedmanee mapping for U+{cp:04X} {ch!r}")
            keys.append(layout[idx])
        else:  # en: type ASCII directly
            if cp >= 0x80:
                sys.exit(f"non-ASCII {ch!r} in --en segment")
            keys.append(layout.get(cp, ("spc", False)))
    return keys


class QMP:
    def __init__(self, sock):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(sock)
        self.f = self.s.makefile("rwb", buffering=0)
        self.f.readline()  # QMP greeting
        self.cmd("qmp_capabilities")

    def cmd(self, name, **args):
        obj = {"execute": name}
        if args:
            obj["arguments"] = args
        self.f.write((json.dumps(obj) + "\n").encode())
        line = self.f.readline()
        if not line:
            return {"error": "connection closed"}
        return json.loads(line.decode(errors="replace"))

    @staticmethod
    def _ev(qcode, down):
        return {"type": "key",
                "data": {"key": {"type": "qcode", "data": qcode}, "down": down}}

    def key(self, qcode, delay):
        self.cmd("input-send-event", events=[self._ev(qcode, True), self._ev(qcode, False)])
        time.sleep(delay)

    def _down(self, qcode):
        self.cmd("input-send-event", events=[self._ev(qcode, True)])

    def _up(self, qcode):
        self.cmd("input-send-event", events=[self._ev(qcode, False)])

    def keys(self, keys, delay):
        for qcode, shift in keys:
            if shift:
                # hold shift *while* the key is pressed so the guest sees the
                # shifted UnicodeChar (e.g. 'T' for Thai ธ์, ':' for ASCII ':')
                self._down("shift")
                time.sleep(delay)
            self.key(qcode, delay)
            if shift:
                self._up("shift")
                time.sleep(delay)

    def close(self):
        self.s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sock")
    ap.add_argument("--th", action="append", default=[], help="Thai segment")
    ap.add_argument("--en", action="append", default=[], help="ASCII/English segment")
    ap.add_argument("--segments", default=None,
                    help="'mode:text,mode:text,...' e.g. 'th:จำไว้,en: = ,th:สีเหลือง'")
    ap.add_argument("--shot", default=None, help="screendump path (PPM)")
    ap.add_argument("--delay", type=float, default=0.03, help="s per key")
    ap.add_argument("--enter", action="store_true", default=True, help="press Enter (default)")
    ap.add_argument("--no-enter", dest="enter", action="store_false")
    ap.add_argument("--start-mode", default="en", choices=["en", "th"],
                    help="guest's current mode (it persists between invocations)")
    args = ap.parse_args()

    # build the ordered (mode, text) segment list from --segments and/or --th/--en
    segments = []
    if args.segments:
        for part in args.segments.split(","):
            mode, _, text = part.partition(":")
            segments.append((mode.strip(), text))
    # --th/--en repeatable flags append after any --segments (interleaved order lost)
    for mode, texts in (("th", args.th), ("en", args.en)):
        for t in texts:
            segments.append((mode, t))

    if not segments:
        sys.exit("nothing to type (pass --segments or --th/--en)")

    rev = reverse_map(parse_kedmanee())
    layout = us_layout()

    qmp = QMP(args.sock)
    current = args.start_mode  # the guest's mode persists across invocations
    for mode, text in segments:
        if mode != current:
            qmp.key("f1", 0.15)
            current = mode
        qmp.keys(text_to_keys(text, mode, rev, layout), args.delay)

    if args.enter:
        qmp.key("ret", 0.15)

    if args.shot:
        r = qmp.cmd("screendump", filename=args.shot)
        print("screendump:", r)
    qmp.close()


if __name__ == "__main__":
    main()
