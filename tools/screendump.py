#!/usr/bin/env python3
"""
screendump.py -- capture the guest framebuffer from a running QEMU via QMP.

Usage:
    python3 tools/screendump.py <qmp-unix-socket> <output-file>

Example (after `make qemu-headless` is running in another terminal):
    python3 tools/screendump.py /tmp/qemu_bm.sock /tmp/shot.ppm

Note: QEMU writes the image in PPM (Netpbm) format regardless of the file
extension, unless it was built with PNG support.  Convert before viewing:

    python3 -c "from PIL import Image; \
                Image.open('/tmp/shot.ppm').convert('RGB').save('/tmp/shot.png')"
"""

import json
import socket
import sys


def qmp_command(sock_path, command):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    f = s.makefile("rwb", buffering=0)

    f.readline()  # discard the {"QMP": {...}} greeting

    # QMP handshake, then the actual command.
    f.write(b'{"execute":"qmp_capabilities"}\n')
    f.readline()

    f.write((json.dumps(command) + "\n").encode())
    reply = f.readline()
    s.close()
    return reply.decode(errors="replace")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sock, outfile = sys.argv[1], sys.argv[2]
    reply = qmp_command(sock, {
        "execute": "screendump",
        "arguments": {"filename": outfile},
    })
    print(reply)
    if '"return"' not in reply:
        sys.exit("error: screendump failed: %s" % reply)
    print("saved %s" % outfile)


if __name__ == "__main__":
    main()
