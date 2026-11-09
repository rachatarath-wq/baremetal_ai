qemu-system-x86_64 -machine q35 -m 1G -net none -bios OVMF.fd \
  -device ich9-ahci,id=ahci \
  -drive file=fat:rw:/tmp/baremetal_test,format=raw,if=none,id=disk0 \
  -device ide-hd,drive=disk0,bus=ahci.0 \
  -device qemu-xhci,id=xhci -device usb-kbd,bus=xhci.0 \
  -monitor tcp:127.0.0.1:45454,server,nowait \
  -qmp tcp:127.0.0.1:45455,server,nowait \
  -vnc 127.0.0.1:1
