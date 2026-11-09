#include <efi.h>
#include <efilib.h>

EFI_STATUS
efi_main(EFI_HANDLE ImageHandle, EFI_SYSTEM_TABLE *SystemTable)
{
    InitializeLib(ImageHandle, SystemTable);

    Print(L"Hello UEFI! int=%d ascii=%a wide=%s\n", 42, "ascii-str", L"wide-str");
    Print(L"float=%f g=%g\n", 1.25, 2.5);

    // pool alloc
    VOID *p = AllocatePool(64);
    if (p) { Print(L"pool ok %p\n", p); FreePool(p); }

    VOID *z = AllocateZeroPool(64);
    if (z) { Print(L"zeropool ok %p\n", z); FreePool(z); }

    // page alloc
    EFI_PHYSICAL_ADDRESS addr = 0;
    EFI_STATUS st = uefi_call_wrapper(BS->AllocatePages, 4,
                                      AllocateAnyPages, EfiLoaderData, 1, &addr);
    if (!EFI_ERROR(st)) {
        Print(L"pages ok %p\n", (void *)(UINTN)addr);
        uefi_call_wrapper(BS->FreePages, 2, addr, 1);
    }

    // file system
    EFI_LOADED_IMAGE *li = NULL;
    EFI_GUID lip = LOADED_IMAGE_PROTOCOL;
    EFI_GUID fsg = SIMPLE_FILE_SYSTEM_PROTOCOL;
    EFI_FILE_IO_INTERFACE *io = NULL;
    EFI_FILE_HANDLE root = NULL;

    st = uefi_call_wrapper(BS->HandleProtocol, 3, ImageHandle, &lip, (VOID **)&li);
    Print(L"HandleProtocol(loaded) = %r\n", st);
    if (!EFI_ERROR(st)) {
        st = uefi_call_wrapper(BS->HandleProtocol, 3, li->DeviceHandle, &fsg, (VOID **)&io);
        Print(L"HandleProtocol(sfs) = %r\n", st);
        if (!EFI_ERROR(st)) {
            st = uefi_call_wrapper(io->OpenVolume, 2, io, &root);
            Print(L"OpenVolume = %r\n", st);
            if (!EFI_ERROR(st)) {
                EFI_FILE_HANDLE f = NULL;
                st = uefi_call_wrapper(root->Open, 5, root, &f, L"\\tokenizer.bin", EFI_FILE_MODE_READ, 0);
                Print(L"Open(tokenizer.bin) = %r\n", st);
                if (!EFI_ERROR(st)) {
                    CHAR8 buf[64];
                    UINTN sz = sizeof(buf);
                    st = uefi_call_wrapper(f->Read, 3, f, &sz, buf);
                    Print(L"Read = %r size=%u\n", st, (UINTN)sz);
                    uefi_call_wrapper(f->Close, 1, f);
                }
            }
        }
    }

    return EFI_SUCCESS;
}
