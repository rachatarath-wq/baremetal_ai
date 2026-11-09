# ---------------------------------------------------------------------------
#  bare-metal llama2.c as a UEFI application  (BOOTX64.EFI)
#
#  Cross-compile on macOS with mingw-w64 + gnu-efi.  Produces a PE/COFF image
#  directly (UEFI loads PE/COFF natively), so no ELF -> objcopy dance is needed.
#
#  Usage:
#      make            # builds build/BOOTX64.EFI
#      make install    # copies it to fat_usb/EFI/BOOT/BOOTX64.EFI
#      make clean
# ---------------------------------------------------------------------------

CC      = x86_64-w64-mingw32-gcc
GNUEFI  = gnu-efi
BUILD   = build

INCDIR  = -I$(GNUEFI)/inc -I$(GNUEFI)/inc/protocol

# -DGNU_EFI_USE_MS_ABI : mingw already uses the Microsoft x64 ABI, so EFIAPI and
#                        uefi_call_wrapper collapse to direct calls (no SysV<->MS
#                        thunk, no callwrap.c).  Do NOT pass -fshort-wchar: mingw's
#                        wchar_t is already 16-bit, matching CHAR16.
# Do NOT pass -mavx/-march=native: UEFI does not enable AVX (XCR0), so stay on the
# x86-64 SSE2 baseline.  -ffreestanding + -nostdlib keep the image free of libc.
CFLAGS  = -O3 -Wall -ffreestanding -fno-stack-protector -fno-builtin \
          -DGNU_EFI_USE_MS_ABI $(INCDIR)

# gnu-efi library objects.  Excluded on purpose:
#   entry.c     -> ELF _start, not needed for a PE/COFF entry point
#   callwrap.c  -> SysV->MS calling-convention thunk, not needed under MS ABI
#   vm.c        -> runtime virtual-memory mapping, not needed by this app
GNUEFI_OBJS = $(addprefix $(BUILD)/, \
	boxdraw.o cmdline.o console.o crc.o data.o debug.o dpath.o \
	error.o event.o exit.o guid.o hand.o hw.o init.o lock.o misc.o \
	pause.o print.o smbios.o sread.o str.o \
	initplat.o math.o \
	efirtlib.o rtdata.o rtlock.o rtstr.o)

# let make find the gnu-efi .c files across its three source dirs
vpath %.c $(GNUEFI)/lib $(GNUEFI)/lib/x86_64 $(GNUEFI)/lib/runtime

OBJS   = $(GNUEFI_OBJS) $(BUILD)/run.o
TARGET = $(BUILD)/BOOTX64.EFI

# PE subsystem 10 = EFI_APPLICATION, entry point = efi_main.
# --dynamicbase emits a .reloc section so firmware can relocate the image.
LDFLAGS = -nostdlib -Wl,--subsystem,10 -Wl,--entry,efi_main -Wl,--dynamicbase
LIBS    = -lgcc

all: $(TARGET)

$(BUILD)/%.o: %.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/run.o: run.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) $(LIBS) -o $@

$(BUILD):
	mkdir -p $(BUILD)

install: $(TARGET)
	cp $(TARGET) fat_usb/EFI/BOOT/BOOTX64.EFI

clean:
	rm -rf $(BUILD)

.PHONY: all install clean
