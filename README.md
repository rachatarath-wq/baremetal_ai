# Bare-metal llama2.c as a UEFI payload (BOOTX64.EFI)

Port of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) `run.c` to run
entirely bare-metal — no operating system, no libc. The result is a single PE/COFF UEFI
application that firmware loads and runs directly.

- **Cross-compiler:** `x86_64-w64-mingw32-gcc` (mingw-w64), running on macOS
- **UEFI lib:** `gnu-efi`
- **Output:** `build/BOOTX64.EFI` (PE32+, subsystem `0x0a` EFI application, `.reloc` present)

---

## Files

| File | Purpose |
|------|---------|
| `run.c` | The full bare-metal port (no `<stdio.h>`, `<stdlib.h>`, `<math.h>`, `<string.h>`, `<time.h>`, `<unistd.h>`; only `<efi.h>` + `<efilib.h>`) |
| `Makefile` | mingw-w64 + gnu-efi cross-compile recipe |
| `build/BOOTX64.EFI` | Built image (also copied to `fat_usb/EFI/BOOT/BOOTX64.EFI`) |
| `stories15M.bin` | Model checkpoint (from llama2.c) |
| `tokenizer.bin` | Tokenizer (from llama2.c) |

---

## The four deliverables

### 1. Custom math (freestanding, no `-lm`)

All implemented directly in `run.c`:

| Function | Technique |
|----------|-----------|
| `expf(x)` | Horner-evaluated Taylor series for the integer part + IEEE-754 bit-trick for the fractional part (`efi_pow2` scales `x / ln 2` into the exponent field). Range-clamped to avoid overflow/denormals. |
| `sqrtf(x)` | Newton–Raphson, seeded with `(bits >> 1) + 0x1FC00000`. |
| `sinf(x)` / `cosf(x)` | `efi_reduce_pi` range-reduction into `[-π/2, π/2]`, then Taylor series. |

`powf` is **removed entirely**. The only RoPE frequency that used it —

```c
freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
```

— becomes

```c
freq = expf(-(head_dim / (float)head_size) * 9.210340371976184f);  /* ln(10000) */
```

**Constraint:** no AVX. `-mavx` / `-march=native` are forbidden because UEFI never enables
XCR0, so AVX instructions would fault. The whole image stays on the x86-64 SSE2 baseline.

---

### 2. UEFI file reading

`read_file()` reads `stories15M.bin` and `tokenizer.bin` from the FAT32 root into RAM:

1. `HandleProtocol(ImageHandle, LOADED_IMAGE_PROTOCOL)` → `li->DeviceHandle`
2. `HandleProtocol(li->DeviceHandle, SIMPLE_FILE_SYSTEM_PROTOCOL)` → `EFI_FILE_IO_INTERFACE`
3. `OpenVolume` → root `EFI_FILE_HANDLE`
4. `Open(root, &f, L"\\stories15M.bin", EFI_FILE_MODE_READ, 0)`
5. `GetInfo(f, EFI_FILE_INFO_ID, ...)` → file size
6. `AllocatePages` (26 MB checkpoint) or `AllocatePool` (tokenizer) → `Read` in a loop to EOF

The GUIDs are **local `EFI_GUID` initializers** (not references to gnu-efi's `lib/guid.c`
symbols). Protocol call argument counts:

```
Open=5   Close=1   Read=3   GetInfo=4
HandleProtocol=3   OpenVolume=2   AllocatePages=4   FreePages=2
```

---

### 3. Main loop + memory allocation

- `efi_main(EFI_HANDLE, EFI_SYSTEM_TABLE*)` replaces `main()`, calls `InitializeLib(...)` first.
- `malloc_run_state` / `free_run_state` use `AllocateZeroPool` / `FreePool` (small `RunState` buffers).
- Checkpoint weights are page-allocated via `AllocatePages` and freed with `FreePages`
  (page count `= (size + 4095) / 4096`, matching the allocation in `read_file`).
- Tokenizer buffer uses `AllocatePool` / `FreePool`.
- The transformer build, `memory_map_weights`, `forward`, and sampler logic are
  algorithmically identical to Karpathy's, but use `memcpy`-based buffer reads instead of
  `mmap`/`fread`, and `efi_qsort`/`efi_bsearch` instead of `qsort`/`bsearch`.
- `decode()` hand-parses `<0xNN>` byte escapes (no `sscanf`); `encode()` uses
  `strcpy`/`strcat` (no `sprintf`); output via `Print(L"%a", piece)`.
- `generate()` accepts a `char *prompt` and feeds it to `encode()` as the starting tokens.
- **Interactive prompting:** `read_line(SystemTable->ConIn, buf, max)` waits on
  `ConIn->ReadKeyStroke`, echoes printable ASCII, handles Backspace (`\b \b` erase), and
  returns on Enter. `efi_main` loops asking `"Enter prompt (empty to quit): "`, generates,
  and repeats until an empty line.
- **Thai input (Kedmanee):** `F1` toggles an EN/TH mode; in TH mode each US key maps to a
  Thai codepoint via `gKedmanee[128]`, encoded to UTF-8 and rendered through
  `fb_put_codepoint` (with zero-width combining-mark handling). Backspace deletes a whole
  multi-byte codepoint. See [`USAGE_THAI.md`](USAGE_THAI.md) for the full key table,
  small-model setup, and QEMU testing.

---

### 4. Makefile / compile command

```makefile
CC      = x86_64-w64-mingw32-gcc
CFLAGS  = -O3 -Wall -ffreestanding -fno-stack-protector -fno-builtin \
          -DGNU_EFI_USE_MS_ABI -Ignu-efi/inc -Ignu-efi/inc/protocol
LDFLAGS = -nostdlib -Wl,--subsystem,10 -Wl,--entry,efi_main -Wl,--dynamicbase
LIBS    = -lgcc
```

Key decisions (all required for a clean mingw + gnu-efi build):

- **`-DGNU_EFI_USE_MS_ABI`** — mingw already uses the MS x64 ABI, so `EFIAPI` collapses to
  `__attribute__((ms_abi))` and `uefi_call_wrapper` to a direct call. This is why
  `callwrap.c` is **excluded** (it is a SysV→MS thunk).
- **No `-fshort-wchar`** — mingw's `wchar_t` is already 16-bit, so `L"..."` matches `CHAR16`.
- **`entry.c` excluded** — it is an ELF `_start`; PE/COFF uses `--entry,efi_main` instead.
- **`--dynamicbase`** emits `.reloc` so firmware can rebase the image.
- gnu-efi objects: `lib/*.c` (minus `entry.c`, `callwrap.c`), `lib/x86_64/{initplat,math}.c`,
  and `lib/runtime/{efirtlib,rtdata,rtlock,rtstr}.c` (minus `vm.c`).

---

## Build & install

```bash
make            # builds build/BOOTX64.EFI
make install    # copies it to fat_usb/EFI/BOOT/BOOTX64.EFI
make clean
```

---

## Deploy to hardware

Run `/createboot <disk>` in Claude Code (or `sudo ./create_usb.sh <disk>`): the script
formats the USB as FAT32, copies `BOOTX64.EFI`, the model `.bin`, `tokenizer.bin` and
`kb.txt`, then ejects it. Boot the USB in **UEFI mode**.

---

## Games

Besides chatting with the model, `BOOTX64.EFI` ships five classic games drawn directly to
the framebuffer (no sprite library — every pixel is written by hand):

| Game | Command | Controls |
|------|---------|----------|
| Pac-Man | `/games pacman [auto]` | Arrows / WASD to steer · Q/ESC to quit · `auto` lets the AI play |
| Snake | `/games snake` | Arrows / WASD to turn · first press starts · Q/ESC to quit |
| XO (Tic-tac-toe) | `/games xo` / `tic` | Arrows to move + Enter to place X (vs. AI) |
| Pong | `/games pong` | W/S or ↑/↓ to move the paddle (vs. AI) |
| Tetris | `/games tetris` | ←/→ move · ↑ rotate · ↓ soft drop · Space hard drop · Q/ESC quit |

Type `/games` with no argument to open a picker dropdown (↑/↓ to move, Enter to start, ESC
to cancel). See [`GUIDE_THAI.md`](GUIDE_THAI.md) for the Thai guide.

---

## Verified image properties

```
file format:      pei-x86-64
architecture:     i386:x86-64
Magic:            020b (PE32+)
Subsystem:        0000000a (EFI application)
.reloc:           present (via --dynamicbase)
import table:     empty (no DLL dependencies)
```

---

## Gotcha: gnu-efi `Print` format widths

`%u` reads a **32-bit** `UINT32`, not `UINTN`. For 64-bit values use `%lu` / `%ld`.

| Specifier | Reads | Example |
|-----------|-------|---------|
| `%a` | `CHAR8*` (ASCII) | `Print(L"%a", ascii_str)` |
| `%s` | `CHAR16*` (wide) | `Print(L"%s", L"wide")` |
| `%d` / `%u` | 32-bit `INT32` / `UINT32` | `Print(L"%d", 42)` |
| `%ld` / `%lu` | 64-bit `INT64` / `UINT64` | `Print(L"%lu", checkpoint_size)` |
| `%r` | `EFI_STATUS` | `Print(L"status=%r", st)` |
| `%p` | pointer | `Print(L"%p", ptr)` |
| `%c` | `UINTN` | |
| `%g` | `EFI_GUID*` | |

---

## Next steps (not yet done)

- Optional: `AllocatePages` with `AllocateMaxAddress` below 4 GiB if running on 32-bit-only
  firmware (not needed for x64).
- Clean up the temporary `hello_efi.c` smoke-test file.
