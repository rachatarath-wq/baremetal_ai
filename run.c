/*
 * Inference for a Llama-2 Transformer model in pure C, running bare-metal
 * as a UEFI application (BOOTX64.EFI).
 *
 * Ported from Andrej Karpathy's llama2.c (run.c) for the UEFI environment:
 *   - No libc: only <efi.h> / <efilib.h> from gnu-efi are included.
 *   - efi_main() replaces main().
 *   - malloc/calloc/free -> AllocatePool / AllocateZeroPool / FreePool
 *     (AllocatePages for the large checkpoint weights).
 *   - fopen/fread/mmap -> EFI_SIMPLE_FILE_SYSTEM_PROTOCOL + EFI_FILE_PROTOCOL,
 *     reading stories15M.bin and tokenizer.bin from the root of the boot volume.
 *   - printf -> Print(L"...") (gnu-efi format: %a = ASCII, %s = CHAR16).
 *   - Freestanding expf/sqrtf/sinf/cosf replace <math.h>.
 */

#include <efi.h>
#include <efilib.h>

/* High-resolution Thai/ASCII renderer (draw_utf8_char() + font.h glyph data). */
#include "hrfont.h"

/* ----------------------------------------------------------------------------
 * Freestanding libc replacements (no <string.h> / <stdlib.h> / <ctype.h>).
 *
 * NOTE: gnu-efi already provides memset() and memcpy() (lib/init.c), so they are
 * only declared here, not redefined.
 * ------------------------------------------------------------------------- */

void *memcpy(void *dest, const void *src, UINTN n);
void *memset(void *s, int c, UINTN n);

/* framebuffer console primitives (defined in the GOP section below) */
static void fb_putc(char c);
static void fb_puts(const char *s);
static void fb_printf(const char *fmt, ...);

static UINTN strlen(const char *s) {
    UINTN n = 0;
    while (s[n]) n++;
    return n;
}

static int strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return (unsigned char)*a - (unsigned char)*b;
}

static int strncmp(const char *a, const char *b, UINTN n) {
    while (n && *a && *a == *b) { a++; b++; n--; }
    if (n == 0) return 0;
    return (unsigned char)*a - (unsigned char)*b;
}

static char *strcpy(char *dest, const char *src) {
    char *r = dest;
    while ((*dest++ = *src++) != '\0') ;
    return r;
}

static char *strcat(char *dest, const char *src) {
    char *r = dest;
    while (*dest) dest++;
    while ((*dest++ = *src++) != '\0') ;
    return r;
}

int memcmp(const void *a, const void *b, UINTN n) {
    const unsigned char *x = (const unsigned char *)a;
    const unsigned char *y = (const unsigned char *)b;
    for (UINTN i = 0; i < n; i++) {
        if (x[i] != y[i]) return (int)x[i] - (int)y[i];
    }
    return 0;
}

void *memmove(void *dest, const void *src, UINTN n) {
    unsigned char *d = (unsigned char *)dest;
    const unsigned char *s = (const unsigned char *)src;
    if (d == s) return dest;
    if (d < s) {
        for (UINTN i = 0; i < n; i++) d[i] = s[i];
    } else {
        for (UINTN i = n; i-- > 0; ) d[i] = s[i];
    }
    return dest;
}

static int isprint(int c) { return (c >= 0x20 && c <= 0x7e); }

static int isspace(int c) {
    return (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f');
}

/* ----------------------------------------------------------------------------
 * Freestanding single-precision math (no <math.h>, no FPU runtime library).
 * The only transcendental calls llama2.c makes are expf (softmax, SiLU),
 * sqrtf (RMSNorm, attention scaling) and sinf/cosf (RoPE).  powf(10000, y) is
 * rewritten as expf(-y * ln(10000)) so logf/powf are not needed.
 * ------------------------------------------------------------------------- */

#define FLT_MAX_VAL 3.402823466e38f

static float efi_pow2(int n) {
    /* 2^n via IEEE-754 exponent-field bit manipulation, n in [-126, 127]. */
    if (n < -126) return 0.0f;
    if (n > 127)  return FLT_MAX_VAL;
    union { unsigned int u; float f; } x;
    x.u = (unsigned int)(n + 127) << 23;
    return x.f;
}

float expf(float x) {
    /* e^x = 2^n * e^r with n = round(x/ln2), r = x - n*ln2, |r| <= ln2/2. */
    if (x > 88.722839f) return FLT_MAX_VAL;  /* overflow */
    if (x < -87.336544f) return 0.0f;        /* underflow */

    int n = (int)(x * 1.4426950408889634f + (x >= 0.0f ? 0.5f : -0.5f));
    float r = x - (float)n * 0.6931471805599453f;

    /* Taylor series for e^r, evaluated with Horner (8 terms). */
    float p = 1.0f / 5040.0f;
    p = p * r + 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;

    return p * efi_pow2(n);
}

float sqrtf(float x) {
    if (x <= 0.0f) return x;   /* 0 -> 0, negatives -> NaN; callers use x > 0 */

    /* Good initial guess from the exponent, then Newton-Raphson refinement. */
    union { float f; unsigned int u; } v;
    v.f = x;
    v.u = (v.u >> 1) + 0x1FC00000;
    float y = v.f;
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    return y;
}

static float efi_floor(float x) {
    int i = (int)x;
    float f = (float)i;
    return (x < f) ? f - 1.0f : f;
}

static float efi_reduce_pi(float x) {
    /* reduce x into [-pi, pi] so the Taylor series stays accurate */
    return x - efi_floor(x / 6.28318530717958647692f + 0.5f) * 6.28318530717958647692f;
}

float sinf(float x) {
    x = efi_reduce_pi(x);
    float x2 = x * x;
    float term = x;
    float sum = x;                 /* sin(x) = x - x^3/3! + x^5/5! - ...  */
    for (int k = 1; k <= 7; k++) {
        term *= -x2 / (float)((2 * k) * (2 * k + 1));
        sum += term;
    }
    return sum;
}

float cosf(float x) {
    return sinf(x + 1.57079632679489661923f);   /* cos(x) = sin(x + pi/2) */
}

/* ----------------------------------------------------------------------------
 * UEFI file I/O (replaces fopen/fread/fseek/mmap).
 * ------------------------------------------------------------------------- */

static EFI_FILE_HANDLE open_root_volume(EFI_HANDLE ImageHandle) {
    EFI_STATUS status;
    EFI_LOADED_IMAGE *loaded_image = NULL;
    EFI_FILE_IO_INTERFACE *io = NULL;
    EFI_FILE_HANDLE root = NULL;

    EFI_GUID lipGuid = LOADED_IMAGE_PROTOCOL;
    EFI_GUID fsGuid  = SIMPLE_FILE_SYSTEM_PROTOCOL;

    status = uefi_call_wrapper(BS->HandleProtocol, 3, ImageHandle,
                               &lipGuid, (VOID **)&loaded_image);
    if (EFI_ERROR(status) || loaded_image == NULL) return NULL;

    status = uefi_call_wrapper(BS->HandleProtocol, 3, loaded_image->DeviceHandle,
                               &fsGuid, (VOID **)&io);
    if (EFI_ERROR(status) || io == NULL) return NULL;

    status = uefi_call_wrapper(io->OpenVolume, 2, io, &root);
    if (EFI_ERROR(status)) return NULL;

    return root;
}

/* Read the whole file `name` (relative to `root`) into a freshly allocated
 * buffer.  Returns EFI_SUCCESS on success and sets *out / *out_size.
 * want_pages selects AllocatePages (page-aligned, for the 60 MB weights)
 * versus AllocatePool (for the ~400 KB tokenizer). */
static EFI_STATUS read_file(EFI_FILE_HANDLE root, CHAR16 *name,
                            int want_pages, VOID **out, UINTN *out_size) {
    EFI_STATUS status;
    EFI_FILE_HANDLE file = NULL;
    EFI_FILE_INFO *info = NULL;
    UINTN info_size = SIZE_OF_EFI_FILE_INFO + 256;
    VOID *buffer = NULL;
    UINTN file_size = 0;

    EFI_GUID fiGuid = EFI_FILE_INFO_ID;

    *out = NULL;
    *out_size = 0;

    status = uefi_call_wrapper(root->Open, 5, root, &file, name,
                               EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(status)) goto fail;

    /* query file size */
    info = AllocateZeroPool(info_size);
    if (info == NULL) { status = EFI_OUT_OF_RESOURCES; goto fail; }
    status = uefi_call_wrapper(file->GetInfo, 4, file, &fiGuid, &info_size, info);
    if (status == EFI_BUFFER_TOO_SMALL) {
        FreePool(info);
        info = AllocateZeroPool(info_size);
        if (info == NULL) { status = EFI_OUT_OF_RESOURCES; goto fail; }
        status = uefi_call_wrapper(file->GetInfo, 4, file, &fiGuid, &info_size, info);
    }
    if (EFI_ERROR(status)) goto fail;
    file_size = (UINTN)info->FileSize;

    /* allocate the destination buffer */
    if (want_pages) {
        EFI_PHYSICAL_ADDRESS addr = 0;
        UINTN pages = EFI_SIZE_TO_PAGES(file_size);

        /* A 1 GB+ model needs a single contiguous block of RAM.  Try
         * AllocateAnyPages first (firmware may place it anywhere).  Some
         * firmware cannot find a large contiguous block above 4 GiB, so on
         * failure retry with AllocateMaxAddress pinned below 4 GiB. */
        status = uefi_call_wrapper(BS->AllocatePages, 4, AllocateAnyPages,
                                   EfiLoaderData, pages, &addr);
        if (EFI_ERROR(status)) {
            addr = 0xFFFFFFFF;   /* AllocateMaxAddress: highest usable address */
            status = uefi_call_wrapper(BS->AllocatePages, 4, AllocateMaxAddress,
                                       EfiLoaderData, pages, &addr);
        }
        if (EFI_ERROR(status)) goto fail;
        buffer = (VOID *)(UINTN)addr;
    } else {
        buffer = AllocatePool(file_size);
        if (buffer == NULL) { status = EFI_OUT_OF_RESOURCES; goto fail; }
    }

    /* read the whole file */
    UINTN remaining = file_size;
    UINT8 *dst = (UINT8 *)buffer;
    while (remaining > 0) {
        UINTN chunk = remaining;
        status = uefi_call_wrapper(file->Read, 3, file, &chunk, dst);
        if (EFI_ERROR(status) || chunk == 0) goto fail;
        dst += chunk;
        remaining -= chunk;
    }

    uefi_call_wrapper(file->Close, 1, file);
    FreePool(info);
    *out = buffer;
    *out_size = file_size;
    return EFI_SUCCESS;

fail:
    if (info != NULL) FreePool(info);
    if (file != NULL) uefi_call_wrapper(file->Close, 1, file);
    if (buffer != NULL) {
        if (want_pages) uefi_call_wrapper(BS->FreePages, 2, (UINTN)buffer,
                                          EFI_SIZE_TO_PAGES(file_size));
        else FreePool(buffer);
    }
    return status;
}

/* ----------------------------------------------------------------------------
 * Transformer model (structs and weight layout identical to llama2.c)
 * ------------------------------------------------------------------------- */

typedef struct {
    int dim;          // transformer dimension
    int hidden_dim;   // for ffn layers
    int n_layers;     // number of layers
    int n_heads;      // number of query heads
    int n_kv_heads;   // number of key/value heads (can be < query heads: multiquery)
    int vocab_size;   // vocabulary size, usually 256 (byte-level)
    int seq_len;      // max sequence length
} Config;

/* Q8_0 quantized tensor: int8 values plus one fp32 scale per group.  The group
 * size is the global gQGroupSize, read from the checkpoint header. */
typedef struct {
    int8_t* q;    // quantized int8 values
    float*  s;    // per-group scale factors
} QuantizedTensor;

/* group size for Q8_0 quantization, read from the checkpoint header */
static int gQGroupSize = 0;

typedef struct {
    QuantizedTensor *q_tokens;        // (vocab_size, dim) quantized embedding
    float* token_embedding_table;     // (vocab_size, dim) dequantized embedding
    float* rms_att_weight;            // (layer, dim) fp32 rmsnorm weights
    float* rms_ffn_weight;            // (layer, dim) fp32 rmsnorm weights
    QuantizedTensor *wq;              // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk;              // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv;              // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo;              // (layer, n_heads * head_size, dim)
    QuantizedTensor *w1;              // (layer, hidden_dim, dim)
    QuantizedTensor *w2;              // (layer, dim, hidden_dim)
    QuantizedTensor *w3;              // (layer, hidden_dim, dim)
    float* rms_final_weight;          // (dim,) fp32 rmsnorm weights
    QuantizedTensor *wcls;            // classifier weights (or == q_tokens)
} TransformerWeights;

typedef struct {
    float *x;          // activation at current time stamp (dim,)
    float *xb;         // same, but inside a residual branch (dim,)
    float *xb2;        // an additional buffer just for convenience (dim,)
    float *hb;         // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2;        // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x/xb activations (dim,)
    QuantizedTensor hq; // quantized hb activations (hidden_dim,)
    float *q;          // query (dim,)
    float *k;          // key (dim,) -- alias into key_cache
    float *v;          // value (dim,) -- alias into value_cache
    float *att;        // buffer for scores/attention values (n_heads, seq_len)
    float *logits;     // output logits
    float* key_cache;   // (layer, seq_len, kv_dim)
    float* value_cache; // (layer, seq_len, kv_dim)
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    VOID *data;        // the checkpoint buffer (page-aligned), freed on shutdown
    UINTN data_size;   // size of the checkpoint buffer in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = AllocateZeroPool(p->dim * sizeof(float));
    s->xb = AllocateZeroPool(p->dim * sizeof(float));
    s->xb2 = AllocateZeroPool(p->dim * sizeof(float));
    s->hb = AllocateZeroPool(p->hidden_dim * sizeof(float));
    s->hb2 = AllocateZeroPool(p->hidden_dim * sizeof(float));
    s->xq.q = AllocateZeroPool(p->dim * sizeof(int8_t));
    s->xq.s = AllocateZeroPool(p->dim * sizeof(float));
    s->hq.q = AllocateZeroPool(p->hidden_dim * sizeof(int8_t));
    s->hq.s = AllocateZeroPool(p->hidden_dim * sizeof(float));
    s->q = AllocateZeroPool(p->dim * sizeof(float));
    s->key_cache = AllocateZeroPool(p->n_layers * p->seq_len * kv_dim * sizeof(float));
    s->value_cache = AllocateZeroPool(p->n_layers * p->seq_len * kv_dim * sizeof(float));
    s->att = AllocateZeroPool(p->n_heads * p->seq_len * sizeof(float));
    s->logits = AllocateZeroPool(p->vocab_size * sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2
     || !s->xq.q || !s->xq.s || !s->hq.q || !s->hq.s || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fb_puts("error: failed to allocate run state buffers\n");
        return;
    }
}

void free_run_state(RunState* s) {
    if (s->x) FreePool(s->x);
    if (s->xb) FreePool(s->xb);
    if (s->xb2) FreePool(s->xb2);
    if (s->hb) FreePool(s->hb);
    if (s->hb2) FreePool(s->hb2);
    if (s->xq.q) FreePool(s->xq.q);
    if (s->xq.s) FreePool(s->xq.s);
    if (s->hq.q) FreePool(s->hq.q);
    if (s->hq.s) FreePool(s->hq.s);
    if (s->q) FreePool(s->q);
    if (s->att) FreePool(s->att);
    if (s->logits) FreePool(s->logits);
    if (s->key_cache) FreePool(s->key_cache);
    if (s->value_cache) FreePool(s->value_cache);
}

/* Dequantize a Q8_0 tensor into a fp32 buffer. */
void dequantize(QuantizedTensor *qx, float *x, int n) {
    int gs = gQGroupSize;
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / gs];
    }
}

/* Quantize a fp32 activation vector into a QuantizedTensor (symmetric int8,
 * one scale per group of gQGroupSize). */
void quantize(QuantizedTensor *qx, float *x, int n) {
    int gs = gQGroupSize;
    int num_groups = n / gs;
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0f;
        for (int i = 0; i < gs; i++) {
            float val = x[group * gs + i];
            if (val < 0.0f) val = -val;
            if (val > wmax) wmax = val;
        }
        float scale = wmax / 127.0f;
        qx->s[group] = scale;
        for (int i = 0; i < gs; i++) {
            float qv = x[group * gs + i] / scale;
            qx->q[group * gs + i] = (int8_t)(qv >= 0.0f ? qv + 0.5f : qv - 0.5f);
        }
    }
}

/* Map n quantized tensors (each `size_each` int8 elements) out of *ptr,
 * advancing *ptr past each tensor's int8 data + scale array. */
QuantizedTensor *init_quantized_tensors(VOID **ptr, int n, int size_each) {
    UINT8 *p = (UINT8 *)*ptr;
    QuantizedTensor *res = AllocatePool(n * sizeof(QuantizedTensor));
    for (int i = 0; i < n; i++) {
        res[i].q = (int8_t *)p;
        p += size_each;
        res[i].s = (float *)p;
        p += (size_each / gQGroupSize) * sizeof(float);
    }
    *ptr = (VOID *)p;
    return res;
}

void memory_map_weights(TransformerWeights *w, Config* p, VOID* ptr, UINT8 shared_classifier) {
    int head_size = p->dim / p->n_heads;
    // fp32 rmsnorm weights come first
    float* fptr = (float*)ptr;
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    ptr = (VOID*)fptr;
    // token embedding (quantized) + a dequantized fp32 copy for fast gather
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    w->token_embedding_table = AllocatePool(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);
    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

/* Build a Transformer from an in-memory checkpoint buffer (already read from
 * disk).  Returns 0 on success, -1 on a bad header. */
int build_transformer(Transformer *t, VOID *data, UINTN size) {
    UINT8 *p = (UINT8 *)data;

    // header: uint32 magic, int32 version, Config, uint8 shared, int32 group,
    // padded to 256 bytes (llama2.c export.py --version 2, Q8_0).
    UINT32 magic = 0;
    memcpy(&magic, p, sizeof(UINT32)); p += sizeof(UINT32);
    if (magic != 0x616b3432) { fb_puts("error: bad model magic (not Q8_0 v2)\n"); return -1; }
    int version = 0;
    memcpy(&version, p, sizeof(int)); p += sizeof(int);
    if (version != 2) { fb_puts("error: bad model version (need 2)\n"); return -1; }

    Config *config = &t->config;
    memcpy(config, p, sizeof(Config)); p += sizeof(Config);

    UINT8 shared_classifier = 0;
    memcpy(&shared_classifier, p, sizeof(UINT8)); p += sizeof(UINT8);
    int group_size = 0;
    memcpy(&group_size, p, sizeof(int)); p += sizeof(int);
    gQGroupSize = group_size;

    VOID *weights_ptr = (VOID *)((UINT8 *)data + 256);
    memory_map_weights(&t->weights, config, weights_ptr, shared_classifier);

    t->data = data;
    t->data_size = size;
    malloc_run_state(&t->state, config);
    return 0;
}

void free_transformer(Transformer* t) {
    free_run_state(&t->state);
    FreePool(t->weights.q_tokens);
    FreePool(t->weights.token_embedding_table);
    FreePool(t->weights.wq);
    FreePool(t->weights.wk);
    FreePool(t->weights.wv);
    FreePool(t->weights.wo);
    FreePool(t->weights.w1);
    FreePool(t->weights.w2);
    FreePool(t->weights.w3);
    if (t->weights.wcls != t->weights.q_tokens) FreePool(t->weights.wcls);
    if (t->data != NULL) {
        uefi_call_wrapper(BS->FreePages, 2, (UINTN)t->data, EFI_SIZE_TO_PAGES(t->data_size));
        t->data = NULL;
    }
}

/* ----------------------------------------------------------------------------
 * neural net blocks; the dynamics of the Transformer
 * ------------------------------------------------------------------------- */

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,), both inputs Q8_0 quantized
    int gs = gQGroupSize;
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;
        for (int j = 0; j <= n - gs; j += gs) {
            for (int k = 0; k < gs; k++) {
                ival += ((int32_t)x->q[j + k]) * ((int32_t)w->q[in + j + k]);
            }
            val += (float)ival * w->s[(in + j) / gs] * x->s[j / gs];
            ival = 0;
        }
        xout[i] = val;
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));

    // forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position (quantize activations once, reuse xq)
        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head.
        // powf(10000, y) == expf(y * ln(10000)), so 1/pow(10000,y) = expf(-y*ln(10000)).
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = expf(-(head_dim / (float)head_size) * 9.210340371976184f);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;      // query vector for this head
            float* att = s->att + h * p->seq_len; // attention scores for this head
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf((float)head_size);
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // self.w2(F.silu(self.w1(x)) * self.w3(x))
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x) = x * sigmoid(x)
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i]; // elementwise multiply with w3(x)
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

/* ----------------------------------------------------------------------------
 * The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens.
 * The tokenizer file is parsed from a memory buffer instead of via fread().
 * ------------------------------------------------------------------------- */

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, VOID *data, int vocab_size) {
    UINT8 *p = (UINT8 *)data;

    t->vocab_size = vocab_size;
    t->vocab = AllocatePool(vocab_size * sizeof(char*));
    t->vocab_scores = AllocatePool(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily

    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    int max_token_length;
    memcpy(&max_token_length, p, sizeof(int)); p += sizeof(int);
    t->max_token_length = (unsigned int)max_token_length;

    int len;
    for (int i = 0; i < vocab_size; i++) {
        memcpy(&t->vocab_scores[i], p, sizeof(float)); p += sizeof(float);
        memcpy(&len, p, sizeof(int)); p += sizeof(int);
        t->vocab[i] = AllocatePool(len + 1);
        memcpy(t->vocab[i], p, len);
        t->vocab[i][len] = '\0'; // add the string terminating token
        p += len;
    }
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { if (t->vocab[i]) FreePool(t->vocab[i]); }
    if (t->vocab) FreePool(t->vocab);
    if (t->vocab_scores) FreePool(t->vocab_scores);
    if (t->sorted_vocab) FreePool(t->sorted_vocab);
}

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // some tokens designate raw bytes, and look like '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x') {
        byte_val = (unsigned char)((hex_digit(piece[3]) << 4) | hex_digit(piece[4]));
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

/* Generation output sink.  generate() emits decoded token pieces through here:
 * text mode prints to the framebuffer, the GUI captures into gGenBuf so it can
 * route the answer into the chat log. */
static char gGenBuf[8192];
static int  gGenLen = 0;
static int  gGenCapture = 0;   /* 0 = fb_puts, 1 = accumulate into gGenBuf */

static void gen_emit(const char *piece) {
    if (gGenCapture) {
        for (const char *p = piece; *p && gGenLen < (int)sizeof(gGenBuf) - 1; p++)
            gGenBuf[gGenLen++] = *p;
        gGenBuf[gGenLen] = '\0';
    } else {
        fb_puts(piece);
    }
}

static void safe_print(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars
    // or whitespace, because some bytes are control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = (unsigned char)piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    gen_emit(piece);
}

/* Simple freestanding qsort/bsearch (element types are <= 32 bytes here). */

static void qs_impl(unsigned char *a, int lo, int hi, int size,
                    int (*cmp)(const void *, const void *)) {
    if (lo >= hi) return;
    unsigned char pivot[32];
    unsigned char tmp[32];
    memcpy(pivot, a + (size_t)((lo + hi) / 2) * size, size);
    int i = lo, j = hi;
    while (i <= j) {
        while (cmp(a + (size_t)i * size, pivot) < 0) i++;
        while (cmp(a + (size_t)j * size, pivot) > 0) j--;
        if (i <= j) {
            memcpy(tmp, a + (size_t)i * size, size);
            memcpy(a + (size_t)i * size, a + (size_t)j * size, size);
            memcpy(a + (size_t)j * size, tmp, size);
            i++; j--;
        }
    }
    qs_impl(a, lo, j, size, cmp);
    qs_impl(a, i, hi, size, cmp);
}

static void efi_qsort(void *base, int n, int size, int (*cmp)(const void *, const void *)) {
    if (n < 2) return;
    qs_impl((unsigned char *)base, 0, n - 1, size, cmp);
}

static void *efi_bsearch(const void *key, const void *base, int n, int size,
                         int (*cmp)(const void *, const void *)) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int c = cmp(key, (const char *)base + (size_t)mid * size);
        if (c == 0) return (void *)((char *)base + (size_t)mid * size);
        if (c < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return NULL;
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = efi_bsearch(&tok, sorted_vocab, vocab_size,
                                  sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fb_puts("error: cannot encode NULL text\n"); return; }

    if (t->sorted_vocab == NULL) {
        // lazily allocate and sort the vocabulary
        t->sorted_vocab = AllocatePool(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        efi_qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // temporary buffer for merge candidates of two consecutive tokens
    char* str_buffer = AllocatePool((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;

    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default: prepend a dummy prefix token if text != ""
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // UTF-8 handling: process the raw byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {
        // reset the buffer on a new UTF-8 codepoint (i.e. not a continuation byte)
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        // if the next byte is a continuation byte, keep appending (up to 4 bytes)
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // full codepoint read; look it up
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: encode each byte as a token
            // (+3 because the first 3 vocab elements are <unk>, <s>, </s>)
            for (int i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according to vocab_scores
    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            strcpy(str_buffer, t->vocab[tokens[i]]);
            strcat(str_buffer, t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // no more pairs to merge
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    FreePool(str_buffer);
}

/* ----------------------------------------------------------------------------
 * The Sampler, which takes logits and returns a sampled token
 * ------------------------------------------------------------------------- */

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1)
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) return i;
    }
    return n - 1; // in case of rounding errors
}

static int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p (nucleus) sampling: sample from the smallest set of tokens that
    // exceed probability topp.
    int n0 = 0;
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    efi_qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) return probindex[i].index;
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = AllocatePool(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    if (sampler->probindex) FreePool(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (unsigned int)((*state * 0x2545F4914F6CDD1Dull) >> 32);
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

/* ----------------------------------------------------------------------------
 * utilities: wall-clock time in milliseconds (for the tok/s report)
 * ------------------------------------------------------------------------- */

UINT64 time_in_ms(void) {
    EFI_TIME t;
    EFI_STATUS status = uefi_call_wrapper(RT->GetTime, 2, &t, NULL);
    if (EFI_ERROR(status)) return 0;
    return (UINT64)((t.Hour * 3600 + t.Minute * 60 + t.Second) * 1000)
         + (UINT64)(t.Nanosecond / 1000000);
}

/* ----------------------------------------------------------------------------
 * Graphics Output Protocol (GOP) framebuffer console.
 *
 * Replaces Simple Text Output (Print): we take the framebuffer directly and
 * draw glyphs with DrawPixel()/DrawChar(), so we can later render a custom
 * non-ASCII (Thai) bitmap font that firmware has no glyphs for.
 * ------------------------------------------------------------------------- */

#define FONT_W 8
#define FONT_H 16

/* Integer pixel scale applied to every glyph cell: 1 = native 8x16, 2 = 16x32,
 * 3 = 24x48, ...  The font bitmaps stay 8x16; DrawGlyph() expands each font
 * pixel into a FONT_SCALE x FONT_SCALE block. */
#define FONT_SCALE 2
#define CELL_W (FONT_W * FONT_SCALE)
#define CELL_H (FONT_H * FONT_SCALE)

/* Printable ASCII range stored in font8x16_ascii[]. */
#define FONT_ASCII_FIRST 0x20
#define FONT_ASCII_LAST  0x7E

/* Thai Unicode block U+0E00..U+0E7F.  Each code point maps onto
 * font8x16_thai[cp - 0x0E00], so index 0 is U+0E00 and index 127 is U+0E7F. */
#define THAI_BLOCK_BASE 0x0E00
#define THAI_BLOCK_END  0x0E7F

static EFI_GRAPHICS_OUTPUT_PROTOCOL *gGop = NULL;
UINT8  *gFbBase = NULL;        /* framebuffer base (device memory); extern in hrfont.h */
static UINTN   gFbSize = 0;           /* bytes, from Mode->FrameBufferSize */
UINT32  gFbWidth = 0;         /* extern in hrfont.h */
UINT32  gFbHeight = 0;        /* extern in hrfont.h */
UINT32  gFbScanLine = 0;      /* PixelsPerScanLine (>= width); extern in hrfont.h */
static EFI_GRAPHICS_PIXEL_FORMAT gFbFormat = PixelBltOnly;
static EFI_PIXEL_BITMASK gFbMask;

/* text cursor, in character cells */
static UINT32 gCols = 0, gRows = 0;
static UINT32 gCursorX = 0, gCursorY = 0;

/* current foreground / background colours (raw framebuffer words) */
static UINT32 gFgColor = 0x00FFFFFF;
UINT32 gBgColor = 0x00000000;        /* extern in hrfont.h */

/* 8x16 bitmap fonts, one byte per row, MSB = leftmost pixel, FONT_H bytes per
 * glyph.  Two independent arrays:
 *
 *   font8x16_ascii[128][16] - indexed by the raw ASCII byte (0x20..0x7E): the
 *                             classic 8x16 VGA ROM glyphs.
 *   font8x16_thai[128][16]  - indexed by (code point - 0x0E00), covering the
 *                             whole Thai block U+0E00..U+0E7F.
 *
 * Thai glyphs follow Thai vertical layout inside the 16-pixel cell:
 *   rows 0..3   = above-base marks (sara i/ii/ue/uee, mai han-akat, tones, ...)
 *   rows 4..11  = base consonant / vowel / digit body
 *   rows 12..15 = below-base marks (sara u/uu, phinthu)
 * A glyph whose rows are all zero renders as a blank cell. */
#if 0 /* legacy 8x16 bitmap fonts — superseded by font.h (see hrf_draw_cp) */
static const unsigned char font8x16_ascii[128][16] = {
    [' '] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    ['!'] = {0x00,0x00,0x18,0x3C,0x3C,0x3C,0x18,0x18,0x18,0x00,0x18,0x18,0x00,0x00,0x00,0x00},
    ['"'] = {0x00,0x66,0x66,0x66,0x24,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    ['#'] = {0x00,0x00,0x00,0x6C,0x6C,0xFE,0x6C,0x6C,0x6C,0xFE,0x6C,0x6C,0x00,0x00,0x00,0x00},
    ['$'] = {0x18,0x18,0x7C,0xC6,0xC2,0xC0,0x7C,0x06,0x86,0xC6,0x7C,0x18,0x18,0x00,0x00,0x00},
    ['%'] = {0x00,0x00,0x00,0x00,0xC2,0xC6,0x0C,0x18,0x30,0x66,0xC6,0x00,0x00,0x00,0x00,0x00},
    ['&'] = {0x00,0x00,0x38,0x6C,0x6C,0x38,0x76,0xDC,0xCC,0xCC,0x76,0x00,0x00,0x00,0x00,0x00},
    ['\'']= {0x00,0x30,0x30,0x30,0x60,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    ['('] = {0x00,0x00,0x0C,0x18,0x30,0x30,0x30,0x30,0x30,0x18,0x0C,0x00,0x00,0x00,0x00,0x00},
    [')'] = {0x00,0x00,0x30,0x18,0x0C,0x0C,0x0C,0x0C,0x0C,0x18,0x30,0x00,0x00,0x00,0x00,0x00},
    ['*'] = {0x00,0x00,0x00,0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    ['+'] = {0x00,0x00,0x00,0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    [','] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x18,0x30,0x00,0x00,0x00},
    ['-'] = {0x00,0x00,0x00,0x00,0x00,0x00,0xFE,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    ['.'] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x00},
    ['/'] = {0x00,0x00,0x00,0x00,0x02,0x06,0x0C,0x18,0x30,0x60,0xC0,0x80,0x00,0x00,0x00,0x00},
    ['0'] = {0x00,0x00,0x38,0x6C,0xC6,0xC6,0xD6,0xD6,0xC6,0xC6,0x6C,0x38,0x00,0x00,0x00,0x00},
    ['1'] = {0x00,0x00,0x18,0x38,0x78,0x18,0x18,0x18,0x18,0x18,0x18,0x7E,0x00,0x00,0x00,0x00},
    ['2'] = {0x00,0x00,0x7C,0xC6,0x06,0x0C,0x18,0x30,0x60,0xC0,0xC6,0xFE,0x00,0x00,0x00,0x00},
    ['3'] = {0x00,0x00,0x7C,0xC6,0x06,0x06,0x3C,0x06,0x06,0x06,0xC6,0x7C,0x00,0x00,0x00,0x00},
    ['4'] = {0x00,0x00,0x0C,0x1C,0x3C,0x6C,0xCC,0xFE,0x0C,0x0C,0x0C,0x1E,0x00,0x00,0x00,0x00},
    ['5'] = {0x00,0x00,0xFE,0xC0,0xC0,0xC0,0xFC,0x06,0x06,0x06,0xC6,0x7C,0x00,0x00,0x00,0x00},
    ['6'] = {0x00,0x00,0x38,0x60,0xC0,0xC0,0xFC,0xC6,0xC6,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00},
    ['7'] = {0x00,0x00,0xFE,0xC6,0x06,0x06,0x0C,0x18,0x30,0x30,0x30,0x30,0x00,0x00,0x00,0x00},
    ['8'] = {0x00,0x00,0x7C,0xC6,0xC6,0xC6,0x7C,0xC6,0xC6,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00},
    ['9'] = {0x00,0x00,0x7C,0xC6,0xC6,0xC6,0x7E,0x06,0x06,0x06,0x0C,0x78,0x00,0x00,0x00,0x00},
    [':'] = {0x00,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x00,0x00},
    [';'] = {0x00,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x18,0x18,0x30,0x00,0x00,0x00,0x00},
    ['<'] = {0x00,0x00,0x00,0x06,0x0C,0x18,0x30,0x60,0x30,0x18,0x0C,0x06,0x00,0x00,0x00,0x00},
    ['='] = {0x00,0x00,0x00,0x00,0x00,0x7E,0x00,0x00,0x7E,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    ['>'] = {0x00,0x00,0x00,0x60,0x30,0x18,0x0C,0x06,0x0C,0x18,0x30,0x60,0x00,0x00,0x00,0x00},
    ['?'] = {0x00,0x00,0x7C,0xC6,0xC6,0x0C,0x18,0x18,0x18,0x00,0x18,0x18,0x00,0x00,0x00,0x00},
    ['@'] = {0x00,0x00,0x00,0x7C,0xC6,0xC6,0xDE,0xDE,0xDE,0xDC,0xC0,0x7C,0x00,0x00,0x00,0x00},
    ['A'] = {0x00,0x00,0x10,0x38,0x6C,0xC6,0xC6,0xFE,0xC6,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00},
    ['B'] = {0x00,0x00,0xFC,0x66,0x66,0x66,0x7C,0x66,0x66,0x66,0x66,0xFC,0x00,0x00,0x00,0x00},
    ['C'] = {0x00,0x00,0x3C,0x66,0xC2,0xC0,0xC0,0xC0,0xC0,0xC2,0x66,0x3C,0x00,0x00,0x00,0x00},
    ['D'] = {0x00,0x00,0xF8,0x6C,0x66,0x66,0x66,0x66,0x66,0x66,0x6C,0xF8,0x00,0x00,0x00,0x00},
    ['E'] = {0x00,0x00,0xFE,0x66,0x62,0x68,0x78,0x68,0x60,0x62,0x66,0xFE,0x00,0x00,0x00,0x00},
    ['F'] = {0x00,0x00,0xFE,0x66,0x62,0x68,0x78,0x68,0x60,0x60,0x60,0xF0,0x00,0x00,0x00,0x00},
    ['G'] = {0x00,0x00,0x3C,0x66,0xC2,0xC0,0xC0,0xDE,0xC6,0xC6,0x66,0x3A,0x00,0x00,0x00,0x00},
    ['H'] = {0x00,0x00,0xC6,0xC6,0xC6,0xC6,0xFE,0xC6,0xC6,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00},
    ['I'] = {0x00,0x00,0x3C,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00},
    ['J'] = {0x00,0x00,0x1E,0x0C,0x0C,0x0C,0x0C,0x0C,0xCC,0xCC,0xCC,0x78,0x00,0x00,0x00,0x00},
    ['K'] = {0x00,0x00,0xE6,0x66,0x6C,0x6C,0x78,0x78,0x6C,0x66,0x66,0xE6,0x00,0x00,0x00,0x00},
    ['L'] = {0x00,0x00,0xF0,0x60,0x60,0x60,0x60,0x60,0x60,0x62,0x66,0xFE,0x00,0x00,0x00,0x00},
    ['M'] = {0x00,0x00,0xC6,0xEE,0xFE,0xFE,0xD6,0xC6,0xC6,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00},
    ['N'] = {0x00,0x00,0xC6,0xE6,0xF6,0xFE,0xDE,0xCE,0xC6,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00},
    ['O'] = {0x00,0x00,0x38,0x6C,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x6C,0x38,0x00,0x00,0x00,0x00},
    ['P'] = {0x00,0x00,0xFC,0x66,0x66,0x66,0x7C,0x60,0x60,0x60,0x60,0xF0,0x00,0x00,0x00,0x00},
    ['Q'] = {0x00,0x00,0x7C,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xD6,0xDE,0x7C,0x0C,0x0E,0x00,0x00},
    ['R'] = {0x00,0x00,0xFC,0x66,0x66,0x66,0x7C,0x6C,0x66,0x66,0x66,0xE6,0x00,0x00,0x00,0x00},
    ['S'] = {0x00,0x00,0x7C,0xC6,0xC6,0x60,0x38,0x0C,0x06,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00},
    ['T'] = {0x00,0x00,0x7E,0x7E,0x5A,0x18,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00},
    ['U'] = {0x00,0x00,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00},
    ['V'] = {0x00,0x00,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x6C,0x38,0x10,0x00,0x00,0x00,0x00},
    ['W'] = {0x00,0x00,0xC6,0xC6,0xC6,0xC6,0xD6,0xD6,0xD6,0xFE,0xEE,0x6C,0x00,0x00,0x00,0x00},
    ['X'] = {0x00,0x00,0xC6,0xC6,0x6C,0x7C,0x38,0x38,0x7C,0x6C,0xC6,0xC6,0x00,0x00,0x00,0x00},
    ['Y'] = {0x00,0x00,0x66,0x66,0x66,0x66,0x3C,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00},
    ['Z'] = {0x00,0x00,0xFE,0xC6,0x86,0x0C,0x18,0x30,0x60,0xC2,0xC6,0xFE,0x00,0x00,0x00,0x00},
    ['['] = {0x00,0x00,0x3C,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x3C,0x00,0x00,0x00,0x00},
    ['\\']= {0x00,0x00,0x00,0x80,0xC0,0xE0,0x70,0x38,0x1C,0x0E,0x06,0x02,0x00,0x00,0x00,0x00},
    [']'] = {0x00,0x00,0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3C,0x00,0x00,0x00,0x00},
    ['^'] = {0x10,0x38,0x6C,0xC6,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    ['_'] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0x00,0x00},
    ['`'] = {0x30,0x18,0x0C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    ['a'] = {0x00,0x00,0x00,0x00,0x00,0x78,0x0C,0x7C,0xCC,0xCC,0xCC,0x76,0x00,0x00,0x00,0x00},
    ['b'] = {0x00,0x00,0xE0,0x60,0x60,0x78,0x6C,0x66,0x66,0x66,0x66,0x7C,0x00,0x00,0x00,0x00},
    ['c'] = {0x00,0x00,0x00,0x00,0x00,0x7C,0xC6,0xC0,0xC0,0xC0,0xC6,0x7C,0x00,0x00,0x00,0x00},
    ['d'] = {0x00,0x00,0x1C,0x0C,0x0C,0x3C,0x6C,0xCC,0xCC,0xCC,0xCC,0x76,0x00,0x00,0x00,0x00},
    ['e'] = {0x00,0x00,0x00,0x00,0x00,0x7C,0xC6,0xFE,0xC0,0xC0,0xC6,0x7C,0x00,0x00,0x00,0x00},
    ['f'] = {0x00,0x00,0x38,0x6C,0x64,0x60,0xF0,0x60,0x60,0x60,0x60,0xF0,0x00,0x00,0x00,0x00},
    ['g'] = {0x00,0x00,0x00,0x00,0x00,0x76,0xCC,0xCC,0xCC,0xCC,0xCC,0x7C,0x0C,0xCC,0x78,0x00},
    ['h'] = {0x00,0x00,0xE0,0x60,0x60,0x6C,0x76,0x66,0x66,0x66,0x66,0xE6,0x00,0x00,0x00,0x00},
    ['i'] = {0x00,0x00,0x18,0x18,0x00,0x38,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00},
    ['j'] = {0x00,0x00,0x06,0x06,0x00,0x0E,0x06,0x06,0x06,0x06,0x06,0x06,0x66,0x66,0x3C,0x00},
    ['k'] = {0x00,0x00,0xE0,0x60,0x60,0x66,0x6C,0x78,0x78,0x6C,0x66,0xE6,0x00,0x00,0x00,0x00},
    ['l'] = {0x00,0x00,0x38,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00},
    ['m'] = {0x00,0x00,0x00,0x00,0x00,0xEC,0xFE,0xD6,0xD6,0xD6,0xD6,0xC6,0x00,0x00,0x00,0x00},
    ['n'] = {0x00,0x00,0x00,0x00,0x00,0xDC,0x66,0x66,0x66,0x66,0x66,0x66,0x00,0x00,0x00,0x00},
    ['o'] = {0x00,0x00,0x00,0x00,0x00,0x7C,0xC6,0xC6,0xC6,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00},
    ['p'] = {0x00,0x00,0x00,0x00,0x00,0xDC,0x66,0x66,0x66,0x66,0x66,0x7C,0x60,0x60,0xF0,0x00},
    ['q'] = {0x00,0x00,0x00,0x00,0x00,0x76,0xCC,0xCC,0xCC,0xCC,0xCC,0x7C,0x0C,0x0C,0x1E,0x00},
    ['r'] = {0x00,0x00,0x00,0x00,0x00,0xDC,0x76,0x66,0x60,0x60,0x60,0xF0,0x00,0x00,0x00,0x00},
    ['s'] = {0x00,0x00,0x00,0x00,0x00,0x7C,0xC6,0x60,0x38,0x0C,0xC6,0x7C,0x00,0x00,0x00,0x00},
    ['t'] = {0x00,0x00,0x10,0x30,0x30,0xFC,0x30,0x30,0x30,0x30,0x36,0x1C,0x00,0x00,0x00,0x00},
    ['u'] = {0x00,0x00,0x00,0x00,0x00,0xCC,0xCC,0xCC,0xCC,0xCC,0xCC,0x76,0x00,0x00,0x00,0x00},
    ['v'] = {0x00,0x00,0x00,0x00,0x00,0xC6,0xC6,0xC6,0xC6,0xC6,0x6C,0x38,0x00,0x00,0x00,0x00},
    ['w'] = {0x00,0x00,0x00,0x00,0x00,0xC6,0xC6,0xD6,0xD6,0xD6,0xFE,0x6C,0x00,0x00,0x00,0x00},
    ['x'] = {0x00,0x00,0x00,0x00,0x00,0xC6,0x6C,0x38,0x38,0x38,0x6C,0xC6,0x00,0x00,0x00,0x00},
    ['y'] = {0x00,0x00,0x00,0x00,0x00,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x7E,0x06,0x0C,0xF8,0x00},
    ['z'] = {0x00,0x00,0x00,0x00,0x00,0xFE,0xCC,0x18,0x30,0x60,0xC6,0xFE,0x00,0x00,0x00,0x00},
    ['{'] = {0x00,0x00,0x0E,0x18,0x18,0x18,0x70,0x18,0x18,0x18,0x18,0x0E,0x00,0x00,0x00,0x00},
    ['|'] = {0x00,0x00,0x18,0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x00},
    ['}'] = {0x00,0x00,0x70,0x18,0x18,0x18,0x0E,0x18,0x18,0x18,0x18,0x70,0x00,0x00,0x00,0x00},
    ['~'] = {0x00,0x00,0x76,0xDC,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
};

/* Thai block U+0E00..U+0E7F, indexed by (cp - 0x0E00).
 *
 * Vowels, tone marks and Thai digits are drawn as faithfully as an 8x16 bitmap
 * allows.  Consonants are hand-drawn approximations: each carries the small
 * head loop (rows 4..5) that makes Thai text recognisably Thai, followed by a
 * simplified body (rows 6..11).  They are readable and distinct, but NOT a
 * professional typeface — for production glyphs, convert a public-domain Thai
 * BDF/PCF font (e.g. the TLWG "tlwgtypewriter" or "garuda" fonts) to this exact
 * byte layout (16 bytes/glyph, MSB = leftmost pixel) and paste it in.
 *
 * Vertical layout is unchanged:
 *   rows 0..3 = above-base marks, rows 4..11 = body, rows 12..15 = below-base. */
static const unsigned char font8x16_thai[128][16] = {
    [0] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ฀ */
    [1] = {0x00,0x00,0x00,0x00, 0x38,0x44,0x64,0x44, 0x44,0x44,0x44,0x00, 0x00,0x00,0x00,0x00}, /* ก */
    [2] = {0x00,0x00,0x00,0x00, 0x74,0x74,0x74,0x34, 0x24,0x24,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ข */
    [3] = {0x00,0x00,0x00,0x74, 0x54,0xB4,0x74,0x24, 0x24,0x7C,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ฃ */
    [4] = {0x00,0x00,0x00,0x00, 0x3C,0x42,0x5A,0x7A, 0x62,0x22,0x22,0x00, 0x00,0x00,0x00,0x00}, /* ค */
    [5] = {0x00,0x00,0x00,0x00, 0x04,0x7E,0x5A,0x5A, 0x62,0x62,0x62,0x00, 0x00,0x00,0x00,0x00}, /* ฅ */
    [6] = {0x00,0x00,0x00,0x74, 0x54,0xB4,0x64,0x74, 0xAC,0x64,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ฆ */
    [7] = {0x00,0x00,0x00,0x00, 0x18,0x28,0x18,0x48, 0x28,0x18,0x18,0x00, 0x00,0x00,0x00,0x00}, /* ง */
    [8] = {0x00,0x00,0x00,0x00, 0xF0,0x10,0xD0,0x50, 0xD0,0x50,0x30,0x00, 0x00,0x00,0x00,0x00}, /* จ */
    [9] = {0x00,0x00,0x00,0x00, 0x78,0x04,0x64,0xA4, 0x6C,0x3A,0x2E,0x00, 0x00,0x00,0x00,0x00}, /* ฉ */
    [10] = {0x00,0x00,0x00,0x06, 0x76,0x7C,0x74,0x34, 0x24,0x24,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ช */
    [11] = {0x00,0x00,0x00,0x06, 0x74,0x5C,0xB4,0x64, 0x24,0x24,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ซ */
    [12] = {0x00,0x00,0x00,0x00, 0x38,0x44,0x64,0x44, 0x6E,0x75,0x6D,0x00, 0x00,0x00,0x00,0x00}, /* ฌ */
    [13] = {0x00,0x00,0x00,0x00, 0x38,0x44,0xE4,0x44, 0x44,0x64,0x67,0x00, 0x0C,0x07,0x00,0x00}, /* ญ */
    [14] = {0x00,0x00,0x00,0x00, 0x1C,0x62,0x22,0x22, 0x62,0xA2,0x62,0x0A, 0x3A,0x36,0x00,0x00}, /* ฎ */
    [15] = {0x00,0x00,0x00,0x00, 0x1C,0x62,0x22,0x22, 0x62,0xA2,0x62,0x02, 0x72,0x6E,0x00,0x00}, /* ฏ */
    [16] = {0x00,0x00,0x00,0x00, 0x78,0xF0,0x08,0x68, 0x68,0x28,0x18,0x58, 0xD8,0xF8,0x00,0x00}, /* ฐ */
    [17] = {0x00,0x00,0x00,0x7B, 0x6D,0x69,0x71,0x11, 0x11,0x11,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ฑ */
    [18] = {0x00,0x00,0x00,0x00, 0x7C,0x42,0x5A,0x6A, 0x5A,0x77,0x66,0x00, 0x00,0x00,0x00,0x00}, /* ฒ */
    [19] = {0x00,0x00,0x00,0x00, 0x38,0x44,0xE4,0x44, 0x44,0x67,0x65,0x00, 0x00,0x00,0x00,0x00}, /* ณ */
    [20] = {0x00,0x00,0x00,0x00, 0x3C,0x62,0x5A,0x6A, 0x5A,0x52,0x22,0x00, 0x00,0x00,0x00,0x00}, /* ด */
    [21] = {0x00,0x00,0x00,0x00, 0x04,0x7E,0x5A,0x6A, 0x5A,0x52,0x22,0x00, 0x00,0x00,0x00,0x00}, /* ต */
    [22] = {0x00,0x00,0x00,0x00, 0x38,0xC4,0x44,0x44, 0x64,0x54,0x64,0x00, 0x00,0x00,0x00,0x00}, /* ถ */
    [23] = {0x00,0x00,0x00,0x00, 0x66,0xAA,0x72,0x32, 0x32,0x22,0x22,0x00, 0x00,0x00,0x00,0x00}, /* ท */
    [24] = {0x00,0x00,0x00,0x00, 0x7C,0x40,0x78,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ธ */
    [25] = {0x00,0x00,0x00,0x00, 0x62,0x62,0x22,0x22, 0x2E,0x36,0x26,0x00, 0x00,0x00,0x00,0x00}, /* น */
    [26] = {0x00,0x00,0x00,0x00, 0x62,0xA2,0x62,0x22, 0x22,0x22,0x7E,0x00, 0x00,0x00,0x00,0x00}, /* บ */
    [27] = {0x00,0x02,0x02,0x02, 0x62,0xA2,0x62,0x22, 0x22,0x22,0x7E,0x00, 0x00,0x00,0x00,0x00}, /* ป */
    [28] = {0x00,0x00,0x00,0x00, 0x64,0x64,0x64,0x44, 0x54,0x6C,0x44,0x00, 0x00,0x00,0x00,0x00}, /* ผ */
    [29] = {0x00,0x04,0x04,0x04, 0x64,0x64,0x64,0x44, 0x54,0x6C,0x44,0x00, 0x00,0x00,0x00,0x00}, /* ฝ */
    [30] = {0x00,0x00,0x00,0x00, 0x6A,0xAA,0x7A,0x36, 0x36,0x36,0x22,0x00, 0x00,0x00,0x00,0x00}, /* พ */
    [31] = {0x00,0x02,0x02,0x02, 0x6A,0xAA,0x7A,0x36, 0x36,0x36,0x22,0x00, 0x00,0x00,0x00,0x00}, /* ฟ */
    [32] = {0x00,0x00,0x00,0x00, 0x1C,0x62,0x22,0x22, 0x62,0xA2,0x62,0x00, 0x00,0x00,0x00,0x00}, /* ภ */
    [33] = {0x00,0x00,0x00,0x00, 0x62,0xA2,0x62,0x22, 0x7A,0xAE,0x66,0x00, 0x00,0x00,0x00,0x00}, /* ม */
    [34] = {0x00,0x00,0x00,0x00, 0x34,0x74,0x74,0x34, 0x24,0x24,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ย */
    [35] = {0x00,0x00,0x00,0x00, 0x7C,0x70,0x08,0x08, 0x18,0x28,0x18,0x00, 0x00,0x00,0x00,0x00}, /* ร */
    [36] = {0x00,0x00,0x00,0x00, 0x38,0xC4,0x44,0x44, 0x64,0x54,0x64,0x04, 0x04,0x00,0x00,0x00}, /* ฤ */
    [37] = {0x00,0x00,0x00,0x00, 0x78,0x04,0x34,0x4C, 0x6C,0x54,0x64,0x00, 0x00,0x00,0x00,0x00}, /* ล */
    [38] = {0x00,0x00,0x00,0x00, 0x1C,0x62,0x22,0x22, 0x62,0xA2,0x62,0x02, 0x02,0x00,0x00,0x00}, /* ฦ */
    [39] = {0x00,0x00,0x00,0x00, 0x70,0x88,0x08,0x08, 0x18,0x28,0x18,0x00, 0x00,0x00,0x00,0x00}, /* ว */
    [40] = {0x00,0x00,0x00,0x02, 0x3E,0x62,0x5A,0x5A, 0x62,0x22,0x22,0x00, 0x00,0x00,0x00,0x00}, /* ศ */
    [41] = {0x00,0x00,0x00,0x00, 0x62,0xA2,0x6A,0x37, 0x2E,0x22,0x7E,0x00, 0x00,0x00,0x00,0x00}, /* ษ */
    [42] = {0x00,0x00,0x00,0x04, 0x7C,0x04,0x34,0x4C, 0x4C,0x64,0x74,0x00, 0x00,0x00,0x00,0x00}, /* ส */
    [43] = {0x00,0x00,0x00,0x00, 0x66,0xAA,0x6E,0x3A, 0x32,0x22,0x22,0x00, 0x00,0x00,0x00,0x00}, /* ห */
    [44] = {0x00,0x0D,0x0A,0x6E, 0xA2,0x62,0x2A,0x36, 0x36,0x22,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ฬ */
    [45] = {0x00,0x00,0x00,0x00, 0x78,0x84,0x64,0x54, 0x64,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* อ */
    [46] = {0x00,0x00,0x00,0x7C, 0x7C,0x04,0x64,0x64, 0x44,0x7C,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ฮ */
    [47] = {0x00,0x00,0x00,0x00, 0x6C,0x7C,0x74,0x04, 0x04,0x04,0x04,0x00, 0x00,0x00,0x00,0x00}, /* ฯ */
    [48] = {0x00,0x00,0x00,0x00, 0x00,0x60,0x68,0x78, 0x60,0x68,0x78,0x00, 0x00,0x00,0x00,0x00}, /* ะ */
    [49] = {0x00,0x64,0x78,0x00, 0x60,0x00,0x00,0x80, 0x04,0x08,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ั */
    [50] = {0x00,0x00,0x00,0x00, 0x70,0x88,0x08,0x08, 0x08,0x08,0x08,0x00, 0x00,0x00,0x00,0x00}, /* า */
    [51] = {0x18,0x18,0x18,0x00, 0x01,0x02,0x00,0x04, 0x80,0x04,0x30,0x00, 0x00,0x00,0x00,0x00}, /* ำ */
    [52] = {0x70,0x48,0xFC,0x00, 0x20,0x00,0x04,0x80, 0x04,0x08,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ิ */
    [53] = {0x3C,0x4C,0x7C,0x00, 0x20,0x00,0x00,0x80, 0x04,0x08,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ี */
    [54] = {0x7C,0x4C,0xF8,0x00, 0x20,0x00,0x04,0x80, 0x04,0x08,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ึ */
    [55] = {0x3C,0x4C,0xFC,0x00, 0x20,0x00,0x00,0x80, 0x04,0x08,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ื */
    [56] = {0x00,0x00,0x00,0x00, 0x00,0x60,0x00,0x84, 0x80,0x04,0x18,0x00, 0x30,0x38,0x08,0x00}, /* ุ */
    [57] = {0x00,0x00,0x00,0x00, 0x20,0x40,0x84,0x80, 0x04,0x18,0x00,0x68, 0x68,0x38,0x00,0x00}, /* ู */
    [58] = {0x00,0x00,0x00,0x00, 0xC0,0x00,0x08,0x00, 0x08,0x30,0x00,0x00, 0x20,0x00,0x00,0x00}, /* ฺ */
    [59] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ฻ */
    [60] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ฼ */
    [61] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ฽ */
    [62] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ฾ */
    [63] = {0x10,0x78,0x54,0x54, 0x54,0x58,0x78,0x56, 0x52,0x52,0x7C,0x10, 0x10,0x00,0x00,0x00}, /* ฿ */
    [64] = {0x00,0x00,0x00,0x00, 0x40,0x40,0x40,0x40, 0x60,0x60,0x60,0x00, 0x00,0x00,0x00,0x00}, /* เ */
    [65] = {0x00,0x00,0x00,0x00, 0x48,0x48,0x48,0x48, 0x6C,0x6C,0x6C,0x00, 0x00,0x00,0x00,0x00}, /* แ */
    [66] = {0x4C,0x30,0x10,0x10, 0x10,0x10,0x10,0x10, 0x18,0x18,0x18,0x00, 0x00,0x00,0x00,0x00}, /* โ */
    [67] = {0x50,0xD0,0x50,0x10, 0x10,0x10,0x10,0x10, 0x18,0x18,0x18,0x00, 0x00,0x00,0x00,0x00}, /* ใ */
    [68] = {0x20,0x20,0x20,0x20, 0x20,0x20,0x20,0x30, 0x30,0x30,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ไ */
    [69] = {0x00,0x00,0x00,0x00, 0x70,0x88,0x08,0x08, 0x08,0x08,0x08,0x08, 0x08,0x08,0x00,0x00}, /* ๅ */
    [70] = {0x00,0x00,0x00,0x00, 0x6C,0x74,0x64,0x64, 0x04,0x04,0x04,0x04, 0x04,0x0C,0x00,0x00}, /* ๆ */
    [71] = {0x44,0x7C,0x00,0x60, 0x00,0x04,0x80,0x04, 0x08,0x00,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ็ */
    [72] = {0x20,0x20,0x20,0x00, 0x00,0xE0,0x10,0x08, 0x00,0x08,0xB0,0x00, 0x00,0x00,0x00,0x00}, /* ่ */
    [73] = {0x70,0x70,0x60,0x00, 0x40,0x80,0x04,0x80, 0x04,0x08,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ้ */
    [74] = {0x00,0x78,0x78,0x70, 0x00,0x20,0x00,0x04, 0x80,0x00,0x10,0x00, 0x00,0x00,0x00,0x00}, /* ๊ */
    [75] = {0x30,0x78,0x30,0x00, 0x00,0x70,0x88,0x04, 0x80,0x84,0x58,0x00, 0x00,0x00,0x00,0x00}, /* ๋ */
    [76] = {0x00,0x30,0x60,0x00, 0x00,0x70,0x88,0x04, 0x80,0x84,0x58,0x00, 0x00,0x00,0x00,0x00}, /* ์ */
    [77] = {0x00,0x30,0x30,0x00, 0x00,0x70,0x88,0x04, 0x80,0x84,0x58,0x00, 0x00,0x00,0x00,0x00}, /* ํ */
    [78] = {0x38,0x20,0x40,0x60, 0x00,0x00,0x04,0x80, 0x04,0x08,0x00,0x00, 0x00,0x00,0x00,0x00}, /* ๎ */
    [79] = {0x00,0x00,0x00,0x00, 0x1C,0x3A,0x7D,0x7D, 0x59,0x22,0x1C,0x00, 0x00,0x00,0x00,0x00}, /* ๏ */
    [80] = {0x00,0x00,0x00,0x00, 0x00,0x3C,0x42,0x42, 0x42,0x66,0x3C,0x00, 0x00,0x00,0x00,0x00}, /* ๐ */
    [81] = {0x00,0x00,0x00,0x00, 0x00,0x3C,0x66,0x42, 0x5A,0x5A,0x34,0x00, 0x00,0x00,0x00,0x00}, /* ๑ */
    [82] = {0x00,0x00,0x00,0x00, 0x80,0x80,0xBC,0xB4, 0xB4,0x84,0xFC,0x00, 0x00,0x00,0x00,0x00}, /* ๒ */
    [83] = {0x00,0x00,0x00,0x00, 0x00,0x00,0x36,0x49, 0x49,0x69,0x69,0x00, 0x00,0x00,0x00,0x00}, /* ๓ */
    [84] = {0x00,0x00,0x00,0x00, 0x02,0x3E,0x40,0x4C, 0x4C,0x48,0x36,0x00, 0x00,0x00,0x00,0x00}, /* ๔ */
    [85] = {0x00,0x00,0x00,0x00, 0x0A,0x3E,0x60,0x4C, 0x4C,0x48,0x3C,0x00, 0x00,0x00,0x00,0x00}, /* ๕ */
    [86] = {0x00,0x00,0x00,0x00, 0x80,0xF0,0x88,0x08, 0xC8,0xC8,0x70,0x00, 0x00,0x00,0x00,0x00}, /* ๖ */
    [87] = {0x00,0x00,0x00,0x00, 0x00,0x34,0x5A,0x42, 0x42,0x73,0x32,0x00, 0x00,0x00,0x00,0x00}, /* ๗ */
    [88] = {0x00,0x00,0x00,0x03, 0x02,0x3E,0x60,0x40, 0x46,0x5E,0x3E,0x00, 0x00,0x00,0x00,0x00}, /* ๘ */
    [89] = {0x00,0x00,0x00,0x00, 0x01,0x35,0x4D,0x46, 0x44,0x54,0x32,0x00, 0x00,0x00,0x00,0x00}, /* ๙ */
    [90] = {0x00,0x00,0x00,0x00, 0x75,0x75,0x4D,0x35, 0x05,0x05,0x0F,0x00, 0x00,0x00,0x00,0x00}, /* ๚ */
    [91] = {0x00,0x00,0x00,0x00, 0x00,0x38,0x7F,0x5F, 0x43,0x64,0x38,0x00, 0x00,0x00,0x00,0x00}, /* ๛ */
    [92] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๜ */
    [93] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๝ */
    [94] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๞ */
    [95] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๟ */
    [96] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๠ */
    [97] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๡ */
    [98] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๢ */
    [99] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๣ */
    [100] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๤ */
    [101] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๥ */
    [102] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๦ */
    [103] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๧ */
    [104] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๨ */
    [105] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๩ */
    [106] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๪ */
    [107] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๫ */
    [108] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๬ */
    [109] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๭ */
    [110] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๮ */
    [111] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๯ */
    [112] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๰ */
    [113] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๱ */
    [114] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๲ */
    [115] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๳ */
    [116] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๴ */
    [117] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๵ */
    [118] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๶ */
    [119] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๷ */
    [120] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๸ */
    [121] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๹ */
    [122] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๺ */
    [123] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๻ */
    [124] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๼ */
    [125] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๽ */
    [126] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๾ */
    [127] = {0x44,0x44,0x44,0x44, 0x44,0x44,0x44,0x44, 0x44,0x44,0x7C,0x00, 0x00,0x00,0x00,0x00}, /* ๿ */
};
#endif /* legacy 8x16 fonts */

/* Build a raw framebuffer pixel word for an 8-bit-per-channel RGB triple,
 * honouring the GOP pixel format (RGB8, BGR8, or arbitrary bit masks). */
static UINT32 fb_make_color(UINT8 r, UINT8 g, UINT8 b) {
    switch (gFbFormat) {
    case PixelRedGreenBlueReserved8BitPerColor:
        return (UINT32)r | ((UINT32)g << 8) | ((UINT32)b << 16);
    case PixelBlueGreenRedReserved8BitPerColor:
        return (UINT32)b | ((UINT32)g << 8) | ((UINT32)r << 16);
    case PixelBitMask:
        return (UINT32)(((UINT64)r * gFbMask.RedMask   / 255) & gFbMask.RedMask)
             | (UINT32)(((UINT64)g * gFbMask.GreenMask / 255) & gFbMask.GreenMask)
             | (UINT32)(((UINT64)b * gFbMask.BlueMask  / 255) & gFbMask.BlueMask);
    default:
        return 0;
    }
}

/* Write a single pixel.  `color` is already in the framebuffer's native format
 * (use fb_make_color() to produce it). */
void DrawPixel(UINT32 x, UINT32 y, UINT32 color) {
    if (gFbBase == NULL) return;
    if (x >= gFbWidth || y >= gFbHeight) return;
    UINT32 *p = (UINT32 *)(gFbBase + (UINTN)y * gFbScanLine * 4 + (UINTN)x * 4);
    *p = color;
}

static void fb_fill(UINT32 x, UINT32 y, UINT32 w, UINT32 h, UINT32 color) {
    if (gFbBase == NULL) return;
    if (x >= gFbWidth || y >= gFbHeight) return;
    if (w > gFbWidth - x)  w = gFbWidth - x;
    if (h > gFbHeight - y) h = gFbHeight - y;
    for (UINT32 row = 0; row < h; row++) {
        UINT32 *p = (UINT32 *)(gFbBase + (UINTN)(y + row) * gFbScanLine * 4
                               + (UINTN)x * 4);
        for (UINT32 col = 0; col < w; col++) p[col] = color;
    }
}

/* Draw the high-resolution glyph for code point `cp` at cell (cx, cy).
 * transparent 0 = opaque (background painted over), 1 = set pixels only (used
 * for zero-width Thai combining marks overlaid on the base consonant). */
static void hrf_draw_cp(UINT32 cp, UINT32 cx, UINT32 cy, int transparent) {
    const HRFONT_glyph_t *g = hrf_lookup(cp);
    if (g == NULL) g = hrf_lookup('?');
    const UINT8 *bmp = HRFONT_bitmap + (UINTN)(g - HRFONT_glyphs) * HRFONT_GLYPH_BYTES;
    hrf_blit(bmp, (int)cx * HRFONT_W, (int)cy * HRFONT_H, gFgColor, gBgColor, transparent);
}

static void fb_scroll(void) {
    UINTN stride = (UINTN)gFbScanLine * 4;
    UINTN move_rows = (UINTN)(gRows - 1) * CELL_H;
    memmove(gFbBase, gFbBase + (UINTN)CELL_H * stride, move_rows * stride);
    fb_fill(0, (UINT32)move_rows, gFbWidth, CELL_H, gBgColor);
}

static void fb_newline(void) {
    gCursorX = 0;
    gCursorY++;
    if (gCursorY >= gRows) {
        fb_scroll();
        gCursorY = gRows - 1;
    }
}

/* Draw the high-res glyph for `cp` at the cursor and advance one cell. */
static void fb_put_glyph_hr(UINT32 cp) {
    if (gFbBase == NULL) return;
    hrf_draw_cp(cp, gCursorX, gCursorY, 0);
    gCursorX++;
    if (gCursorX >= gCols) { gCursorX = 0; fb_newline(); }
}

static void fb_putc(char c) {
    if (gFbBase == NULL) return;
    if (c == '\n') { fb_newline(); return; }
    if (c == '\r') { gCursorX = 0; return; }
    if (c == '\b') {
        if (gCursorX > 0) gCursorX--;
        hrf_draw_cp(' ', gCursorX, gCursorY, 0);
        return;
    }
    if ((unsigned char)c >= HRFONT_ASCII_FIRST && (unsigned char)c <= HRFONT_ASCII_LAST) {
        fb_put_glyph_hr((unsigned char)c);
    }
}

/* Draw an unknown / unrepresentable character as '?'. */
static void fb_put_unknown(void) {
    fb_put_glyph_hr('?');
}

/* True if `cp` is a Thai combining mark that must overlay the preceding
 * consonant (zero-width, non-advancing): U+0E31, U+0E34..0E3A, U+0E47..0E4E. */
static int thai_is_combining_mark(UINT32 cp) {
    if (cp == 0x0E31) return 1;
    if (cp >= 0x0E34 && cp <= 0x0E3A) return 1;
    if (cp >= 0x0E47 && cp <= 0x0E4E) return 1;
    return 0;
}

/* Map a decoded Unicode code point to a glyph and draw it.
 * Returns 1 if it drew something, 0 if the code point is unsupported. */
static int fb_put_codepoint(UINT32 cp) {
    if (cp < 0x80) {
        fb_putc((char)cp);              /* ASCII, incl. \n \r \b */
        return 1;
    }
    if (cp >= HRFONT_THAI_BASE && cp <= HRFONT_THAI_END) {
        if (thai_is_combining_mark(cp)) {
            /* Zero-width: draw over the previous cell (one cell to the left)
             * without touching gCursorX, so the next character still lands in
             * the cell after the consonant.  Transparent blit keeps the base
             * consonant visible underneath the mark. */
            UINT32 target_cx = (gCursorX > 0) ? (gCursorX - 1) : 0;
            hrf_draw_cp(cp, target_cx, gCursorY, /*transparent=*/1);
            return 1;
        }
        fb_put_glyph_hr(cp);
        return 1;
    }
    fb_put_unknown();                   /* unsupported: placeholder */
    return 0;
}

/* UTF-8 decoder state.  Kept static so a multi-byte sequence that is split
 * across two fb_puts() calls (e.g. the tokenizer splitting one Thai character
 * into two tokens) is still decoded correctly. */
static UINT32 gUtf8Accum = 0;   /* decoded code point bits so far */
static int    gUtf8Need  = 0;   /* continuation bytes still expected */

/* Feed one byte to the UTF-8 decoder.  ASCII passes straight through; a
 * multi-byte sequence is buffered until complete and then drawn via
 * fb_put_codepoint(). */
static void fb_put_utf8_byte(unsigned char b) {
    if (b < 0x80) {
        /* ASCII is never a continuation byte: flush any dangling sequence. */
        gUtf8Need = 0;
        fb_putc((char)b);
        return;
    }

    if (gUtf8Need == 0) {
        /* first byte of a new sequence */
        if ((b & 0xE0) == 0xC0) {            /* 110xxxxx : 2-byte */
            gUtf8Accum = b & 0x1F;
            gUtf8Need  = 1;
        } else if ((b & 0xF0) == 0xE0) {     /* 1110xxxx : 3-byte (Thai) */
            gUtf8Accum = b & 0x0F;
            gUtf8Need  = 2;
        } else if ((b & 0xF8) == 0xF0) {     /* 11110xxx : 4-byte */
            gUtf8Accum = b & 0x07;
            gUtf8Need  = 3;
        } else {
            fb_put_unknown();               /* stray continuation / invalid lead */
        }
        return;
    }

    /* we expected a continuation byte 10xxxxxx */
    if ((b & 0xC0) != 0x80) {
        /* invalid: drop the partial sequence, then re-process b as a new start */
        gUtf8Need = 0;
        fb_put_unknown();
        fb_put_utf8_byte(b);
        return;
    }

    gUtf8Accum = (gUtf8Accum << 6) | (b & 0x3F);
    if (--gUtf8Need == 0) {
        fb_put_codepoint(gUtf8Accum);
    }
}

static void fb_puts(const char *s) {
    while (*s) fb_put_utf8_byte((unsigned char)*s++);
}

/* wide strings: ASCII prints directly, non-ASCII code units go through the
 * same code-point mapping as UTF-8 (so CHAR16 Thai renders too). */
static void fb_wputs(const CHAR16 *s) {
    while (*s) {
        CHAR16 c = *s++;
        if (c < 0x80) fb_putc((char)c);
        else fb_put_codepoint((UINT32)c);
    }
}

static void fb_put_u64(UINT64 v) {
    char buf[24];
    int i = 0;
    if (v == 0) { fb_putc('0'); return; }
    while (v) { buf[i++] = (char)('0' + (v % 10)); v /= 10; }
    while (i) fb_putc(buf[--i]);
}

static void fb_put_s64(INT64 v) {
    if (v < 0) { fb_putc('-'); fb_put_u64((UINT64)(-v)); }
    else fb_put_u64((UINT64)v);
}

static void fb_put_hex(UINT64 v, int digits) {
    static const char hex[] = "0123456789abcdef";
    char buf[16];
    if (digits > 16) digits = 16;
    for (int i = 0; i < digits; i++) { buf[digits - 1 - i] = hex[v & 0xf]; v >>= 4; }
    for (int i = 0; i < digits; i++) fb_putc(buf[i]);
}

/* fixed-point float with 2 decimals (enough for the tok/s report) */
static void fb_put_float(double f) {
    if (f < 0.0) { fb_putc('-'); f = -f; }
    UINT64 ip = (UINT64)f;
    UINT64 fp = (UINT64)((f - (double)ip) * 100.0 + 0.5);
    if (fp >= 100) { ip++; fp -= 100; }
    fb_put_u64(ip);
    fb_putc('.');
    fb_putc((char)('0' + (fp / 10) % 10));
    fb_putc((char)('0' + fp % 10));
}

/* Tiny ASCII formatter mirroring gnu-efi Print's format table:
 *   %a = ASCII (char*), %s = wide (CHAR16*), %c = char,
 *   %d/%i = INT32, %u = UINT32, %ld = INT64, %lu = UINT64,
 *   %lx = UINT64 hex, %f = double (2 decimals), %% = literal '%'. */
static void fb_printf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    for (const char *p = fmt; *p; p++) {
        if (*p != '%') { fb_putc(*p); continue; }
        p++;
        switch (*p) {
            case '%': fb_putc('%'); break;
            case 'a': fb_puts(va_arg(ap, char *)); break;
            case 's': fb_wputs(va_arg(ap, CHAR16 *)); break;
            case 'c': fb_putc((char)va_arg(ap, int)); break;
            case 'd':
            case 'i': fb_put_s64((INT64)va_arg(ap, INT32)); break;
            case 'u': fb_put_u64((UINT64)va_arg(ap, UINT32)); break;
            case 'f': fb_put_float(va_arg(ap, double)); break;
            case 'l':
                p++;
                if (*p == 'd')      fb_put_s64(va_arg(ap, INT64));
                else if (*p == 'u') fb_put_u64(va_arg(ap, UINT64));
                else if (*p == 'x') fb_put_hex(va_arg(ap, UINT64), 16);
                else { fb_putc('%'); fb_putc('l'); fb_putc(*p); }
                break;
            default: fb_putc('%'); fb_putc(*p); break;
        }
    }
    va_end(ap);
}

/* Locate the GOP, select the highest-resolution directly-writable 32-bpp mode,
 * switch to it, and record the framebuffer geometry.  Returns EFI_SUCCESS on
 * success; on failure gFbBase stays NULL and Simple Text Output remains usable. */
static EFI_STATUS init_gop(void) {
    EFI_STATUS status;

    status = uefi_call_wrapper(BS->LocateProtocol, 3, &GraphicsOutputProtocol,
                               NULL, (VOID **)&gGop);
    if (EFI_ERROR(status) || gGop == NULL) return EFI_UNSUPPORTED;

    /* pick the biggest mode that we can write directly */
    UINT32 best = 0;
    UINTN  best_px = 0;
    UINT32 n = gGop->Mode->MaxMode;
    for (UINT32 i = 0; i < n; i++) {
        EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *info = NULL;
        UINTN sz = 0;
        status = uefi_call_wrapper(gGop->QueryMode, 4, gGop, i, &sz, &info);
        if (EFI_ERROR(status) || info == NULL) continue;
        UINT32 fmt = info->PixelFormat;
        if (fmt == PixelRedGreenBlueReserved8BitPerColor ||
            fmt == PixelBlueGreenRedReserved8BitPerColor ||
            fmt == PixelBitMask) {
            UINTN px = (UINTN)info->HorizontalResolution
                     * (UINTN)info->VerticalResolution;
            if (px > best_px) { best_px = px; best = i; }
        }
        FreePool(info);
    }

    if (best_px > 0) {
        status = uefi_call_wrapper(gGop->SetMode, 2, gGop, best);
        if (EFI_ERROR(status)) return status;
    }

    /* re-read mode info: SetMode may have swapped the buffer/geometry */
    EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *info = gGop->Mode->Info;
    if (info == NULL) return EFI_UNSUPPORTED;

    gFbFormat   = info->PixelFormat;
    gFbMask     = info->PixelInformation;
    gFbWidth    = info->HorizontalResolution;
    gFbHeight   = info->VerticalResolution;
    gFbScanLine = info->PixelsPerScanLine;
    gFbBase     = (UINT8 *)(UINTN)gGop->Mode->FrameBufferBase;
    gFbSize     = gGop->Mode->FrameBufferSize;

    if (gFbFormat != PixelRedGreenBlueReserved8BitPerColor &&
        gFbFormat != PixelBlueGreenRedReserved8BitPerColor &&
        gFbFormat != PixelBitMask) {
        gFbBase = NULL;   /* PixelBltOnly: framebuffer is not directly writable */
        return EFI_UNSUPPORTED;
    }

    gCols = gFbWidth / CELL_W;
    gRows = gFbHeight / CELL_H;
    if (gCols == 0 || gRows == 0) { gFbBase = NULL; return EFI_UNSUPPORTED; }

    gFgColor = fb_make_color(0xE0, 0xE0, 0xE0);
    gBgColor = fb_make_color(0x00, 0x00, 0x00);
    gCursorX = gCursorY = 0;
    fb_fill(0, 0, gFbWidth, gFbHeight, gBgColor);

    return EFI_SUCCESS;
}

/* ----------------------------------------------------------------------------
 * console keyboard input (replaces getline()/fgets()/readline())
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Thai (Kedmanee) keyboard input
 * ------------------------------------------------------------------------- */

/* EFI scan code for F1, used to toggle English <-> Thai input mode. */
#define KBD_SCAN_F1 0x0B

/* EFI scan codes for the UP / DOWN arrows, used for command-history recall. */
#define KBD_SCAN_UP    0x01
#define KBD_SCAN_DOWN  0x02

/* 1 = Thai input mode (map US keys to Thai via Kedmanee), 0 = English. */
static int gThaiMode = 0;

/* Thai Kedmanee (TIS 820-2538) layout: US QWERTY ASCII -> Thai code point.
 * Indexed by the ASCII code the UEFI console reports in UnicodeChar.
 * 0 = no mapping (pass the key through unchanged).  Values < 0x80 are ASCII
 * characters produced by the Thai shift layer (e.g. 'M' -> '?'). */
static const UINT32 gKedmanee[128] = {
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,  /* 0x00-0x07 */
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,  /* 0x08-0x0F */
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,  /* 0x10-0x17 */
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,  /* 0x18-0x1F */
    0x0000, 0x002B, 0x002E, 0x0E52, 0x0E53, 0x0E54, 0x0E3F, 0x0E07,  /* 0x20-0x27 */
    0x0E56, 0x0E57, 0x0E55, 0x0E59, 0x0E21, 0x0E02, 0x0E43, 0x0E1D,  /* 0x28-0x2F */
    0x0E08, 0x0E45, 0x002F, 0x002D, 0x0E20, 0x0E16, 0x0E38, 0x0E36,  /* 0x30-0x37 */
    0x0E04, 0x0E15, 0x0E0B, 0x0E27, 0x0E12, 0x0E0A, 0x0E2C, 0x0E26,  /* 0x38-0x3F */
    0x0E51, 0x0E24, 0x0E3A, 0x0E09, 0x0E0F, 0x0E0E, 0x0E42, 0x0E0C,  /* 0x40-0x47 */
    0x0E47, 0x0E13, 0x0E4B, 0x0E29, 0x0E28, 0x003F, 0x0E4C, 0x0E2F,  /* 0x48-0x4F */
    0x0E0D, 0x0E50, 0x0E11, 0x0E06, 0x0E18, 0x0E4A, 0x0E2E, 0x0022,  /* 0x50-0x57 */
    0x0029, 0x0E4D, 0x0028, 0x0E1A, 0x0E03, 0x0E25, 0x0E39, 0x0E58,  /* 0x58-0x5F */
    0x005F, 0x0E1F, 0x0E34, 0x0E41, 0x0E01, 0x0E33, 0x0E14, 0x0E40,  /* 0x60-0x67 */
    0x0E49, 0x0E23, 0x0E48, 0x0E32, 0x0E2A, 0x0E17, 0x0E37, 0x0E19,  /* 0x68-0x6F */
    0x0E22, 0x0E46, 0x0E1E, 0x0E2B, 0x0E30, 0x0E35, 0x0E2D, 0x0E44,  /* 0x70-0x77 */
    0x0E1B, 0x0E31, 0x0E1C, 0x0E10, 0x0E05, 0x002C, 0x0025, 0x0000,  /* 0x78-0x7F */
};

/* Append code point `cp` to `buf` as UTF-8 (1..3 bytes), guarding max_len. */
static void append_utf8(char *buf, int *len, int max_len, UINT32 cp) {
    if (cp < 0x80) {
        if (*len < max_len - 1) { buf[(*len)++] = (char)cp; buf[*len] = '\0'; }
    } else if (cp < 0x800) {
        if (*len < max_len - 2) {
            buf[(*len)++] = (char)(0xC0 | (cp >> 6));
            buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
            buf[*len] = '\0';
        }
    } else {
        if (*len < max_len - 3) {
            buf[(*len)++] = (char)(0xE0 | (cp >> 12));
            buf[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
            buf[*len] = '\0';
        }
    }
}

/* ----------------------------------------------------------------------------
 * Blinking text cursor
 * ------------------------------------------------------------------------- */

#define CURSOR_BLINK_MS     500    /* ON/OFF period while idle */
#define CURSOR_POLL_US      10000  /* idle poll interval: 10ms -> 100 Hz */

/* Cursor shape: solid block fills the cell, underscore draws a thick bar across
 * the cell bottom. */
#define CURSOR_STYLE_BLOCK       1
#define CURSOR_STYLE_UNDERSCORE  2
#define CURSOR_STYLE             CURSOR_STYLE_BLOCK
#define CURSOR_UNDERSCORE_H      3     /* bar thickness, px (underscore style) */

/* Blink state: is the cursor currently painted, and how many microseconds have
 * been stalled since the last toggle (see cursor_blink below). */
static int     gCursorVisible = 1;
static UINT32  gBlinkAccumUs  = 0;

/* Paint the cursor shape at pixel (x, y).  `color` is a framebuffer-native
 * colour word (from fb_make_color()); UINT32 == uint32_t in gnu-efi. */
void draw_cursor(int x, int y, uint32_t color) {
    if (gFbBase == NULL) return;
    if (x < 0 || y < 0)  return;
#if CURSOR_STYLE == CURSOR_STYLE_BLOCK
    fb_fill((UINT32)x, (UINT32)y, HRFONT_W, HRFONT_H, (UINT32)color);
#else
    fb_fill((UINT32)x, (UINT32)(y + HRFONT_H - CURSOR_UNDERSCORE_H),
            HRFONT_W, CURSOR_UNDERSCORE_H, (UINT32)color);
#endif
}

/* Restore the cell at pixel (x, y) to the background, erasing the cursor.
 * font_width/font_height are the cell size in pixels (pass HRFONT_W/HRFONT_H). */
void erase_cursor(int x, int y, uint32_t bg_color, int font_width, int font_height) {
    if (gFbBase == NULL) return;
    if (x < 0 || y < 0 || font_width <= 0 || font_height <= 0) return;
    fb_fill((UINT32)x, (UINT32)y,
            (UINT32)font_width, (UINT32)font_height, (UINT32)bg_color);
}

/* --- thin wrappers that use the global cell cursor -------------------------- */

/* Erase the cursor at (gCursorX, gCursorY) and mark it hidden. */
static void cursor_hide(void) {
    erase_cursor((int)(gCursorX * HRFONT_W), (int)(gCursorY * HRFONT_H),
                 gBgColor, HRFONT_W, HRFONT_H);
    gCursorVisible = 0;
}

/* Draw a solid cursor at (gCursorX, gCursorY) and restart the blink timer. */
static void cursor_show(void) {
    draw_cursor((int)(gCursorX * HRFONT_W), (int)(gCursorY * HRFONT_H), gFgColor);
    gCursorVisible = 1;
    gBlinkAccumUs = 0;   /* solid, and the OFF phase starts fresh */
}

/* Advance the blink timer by one poll interval and toggle visibility when a
 * full period has elapsed.  Call once per idle input-loop iteration. */
static void cursor_blink(void) {
    gBlinkAccumUs += CURSOR_POLL_US;
    if (gBlinkAccumUs >= (UINT32)(CURSOR_BLINK_MS * 1000)) {
        gBlinkAccumUs = 0;
        if (gCursorVisible) cursor_hide();
        else                cursor_show();
    }
}

/* ----------------------------------------------------------------------------
 * Command history (bash-style UP / DOWN recall)
 * ------------------------------------------------------------------------- */

#define MAX_HISTORY      10
#define MAX_PROMPT_LEN   512   /* must match the caller's prompt buffer size */

/* The last MAX_HISTORY prompts; gHistory[gHistoryCount-1] is the newest. */
static char gHistory[MAX_HISTORY][MAX_PROMPT_LEN];
static int  gHistoryCount = 0;   /* number of saved entries (0..MAX_HISTORY)  */
static int  gHistoryIndex = -1;  /* -1 = not browsing; else index into gHistory */
static char gDraft[MAX_PROMPT_LEN]; /* in-progress line, saved while browsing  */

/* Erase the typed text currently on screen: paint the background over every
 * cell from the prompt start up to the cursor.  Assumes the prompt is a single
 * row (prompts long enough to wrap are not expected). */
static void prompt_erase(int start_cx) {
    int cy = (int)gCursorY;
    for (int cx = start_cx; cx < (int)gCursorX; cx++)
        hrf_draw_cp(' ', (UINT32)cx, (UINT32)cy, 0);   /* opaque blank -> erase */
}

/* Erase the on-screen text, reset the cursor to the prompt start, copy `text`
 * into buf, and redraw it (advancing the cursor and setting *len). */
static void redraw_prompt(const char *text, char *buf, int *len, int max_len,
                          int start_cx, int start_cy) {
    prompt_erase(start_cx);
    gCursorX = (UINT32)start_cx;
    gCursorY = (UINT32)start_cy;

    UINTN n = strlen(text);
    if (n > (UINTN)(max_len - 1)) n = (UINTN)(max_len - 1);
    memcpy(buf, text, n);
    buf[n] = '\0';
    *len = (int)n;
    fb_puts(buf);                       /* redraw (handles UTF-8 + Thai marks) */
}

/* UP: recall the previous (older) entry.  The first UP stashes the current
 * line as the draft so DOWN can bring it back. */
static void history_prev(char *buf, int *len, int max_len, int start_cx, int start_cy) {
    if (gHistoryCount == 0) return;              /* nothing saved yet */

    if (gHistoryIndex == -1) {
        strcpy(gDraft, buf);                     /* save the line we were typing */
        gHistoryIndex = gHistoryCount - 1;       /* most recent entry */
    } else if (gHistoryIndex > 0) {
        gHistoryIndex--;                         /* step to an older entry */
    } else {
        return;                                  /* already at the oldest */
    }
    redraw_prompt(gHistory[gHistoryIndex], buf, len, max_len, start_cx, start_cy);
}

/* DOWN: recall the next (newer) entry; past the newest, restore the draft and
 * leave browsing mode. */
static void history_next(char *buf, int *len, int max_len, int start_cx, int start_cy) {
    if (gHistoryIndex == -1) return;             /* not browsing */

    if (gHistoryIndex < gHistoryCount - 1) {
        gHistoryIndex++;                         /* newer entry */
        redraw_prompt(gHistory[gHistoryIndex], buf, len, max_len, start_cx, start_cy);
    } else {
        gHistoryIndex = -1;                      /* back at newest -> restore draft */
        redraw_prompt(gDraft, buf, len, max_len, start_cx, start_cy);
    }
}

/* Save a completed (non-empty) line, keeping only the newest MAX_HISTORY. */
static void history_push(const char *s) {
    if (s == NULL || s[0] == '\0') return;

    if (gHistoryCount < MAX_HISTORY) {
        strcpy(gHistory[gHistoryCount++], s);
    } else {
        for (int i = 1; i < MAX_HISTORY; i++)   /* drop oldest, shift left */
            strcpy(gHistory[i - 1], gHistory[i]);
        strcpy(gHistory[MAX_HISTORY - 1], s);
    }
    gHistoryIndex = -1;   /* submitting ends any browsing session */
}

/* Read one line of text from the UEFI console into buf (max `max_len` bytes
 * including the NUL).  Echoes printable characters, handles Backspace, toggles
 * Thai input with F1, and returns when the user presses Enter.  Arrows, and
 * other F-keys / Delete / Esc are ignored.  A solid block cursor blinks at the
 * insertion point while idle.  Returns the number of bytes stored.
 *
 * NOTE: the GUI event loop uses gui_handle_key() instead.  This (and the cursor
 * / command-history helpers below it) are kept for a possible text-mode fallback,
 * so the function is marked unused rather than deleted. */
static int __attribute__((unused)) read_line(SIMPLE_INPUT_INTERFACE *ConIn, char *buf, int max_len) {
    int len = 0;
    buf[0] = '\0';
    gHistoryIndex = -1;   /* start every line outside of history browsing */

    /* remember where the editable prompt begins, so history recall can erase
     * and redraw just the typed text (leaving the "AI prompt ... : " prefix). */
    int start_cx = (int)gCursorX;
    int start_cy = (int)gCursorY;

    cursor_show();   /* solid cursor at the position the prompt left us at */

    for (;;) {
        EFI_INPUT_KEY key;
        EFI_STATUS status = uefi_call_wrapper(ConIn->ReadKeyStroke, 2, ConIn, &key);

        /* ReadKeyStroke is non-blocking: it returns EFI_NOT_READY while the
         * input queue is empty.  While idle we blink the cursor and yield for
         * a short Stall instead of busy-spinning at 100% CPU. */
        if (status == EFI_NOT_READY) {
            cursor_blink();
            uefi_call_wrapper(BS->Stall, 1, CURSOR_POLL_US);
            continue;
        }
        if (EFI_ERROR(status)) continue;

        /* A key arrived: erase the cursor from its current cell so the edit
         * below repaints cleanly.  Every non-Enter path re-draws it solid at
         * the new position (cursor_show), resetting the blink phase. */
        cursor_hide();

        /* F1 toggles English <-> Thai (Kedmanee) input mode. */
        if (key.ScanCode == KBD_SCAN_F1) {
            gThaiMode = !gThaiMode;
            fb_puts(gThaiMode ? "[TH]" : "[EN]");
            cursor_show();
            continue;
        }

        /* UP / DOWN arrow: recall the previous / next prompt from history. */
        if (key.ScanCode == KBD_SCAN_UP) {
            history_prev(buf, &len, max_len, start_cx, start_cy);
            cursor_show();
            continue;
        }
        if (key.ScanCode == KBD_SCAN_DOWN) {
            history_next(buf, &len, max_len, start_cx, start_cy);
            cursor_show();
            continue;
        }

        CHAR16 ch = key.UnicodeChar;

        if (ch == CHAR_CARRIAGE_RETURN) {
            fb_putc('\n');
            if (len > 0) history_push(buf);   /* save non-empty line to history */
            break;                          /* Enter: cursor stays hidden */
        }

        if (ch == CHAR_BACKSPACE) {
            if (len > 0) {
                gHistoryIndex = -1;         /* editing leaves history browsing */
                /* Remove one full UTF-8 character: a Thai char is a 3-byte
                 * sequence, so keep dropping leading continuation bytes too. */
                len--;
                buf[len] = '\0';
                while (len > 0 && ((unsigned char)buf[len-1] & 0xC0) == 0x80) {
                    len--;
                    buf[len] = '\0';
                }
                fb_putc('\b');              /* erase one glyph cell */
            }
            cursor_show();
            continue;
        }

        if (key.ScanCode != 0) {
            cursor_show();                  /* other F-keys / Delete / Esc / Home / End */
            continue;
        }

        /* printable characters only (drop control characters) */
        if (ch < 0x20 || ch >= 0x7f) {
            cursor_show();
            continue;
        }

        gHistoryIndex = -1;                 /* typing leaves history browsing */

        if (gThaiMode) {
            UINT32 cp = gKedmanee[(unsigned char)ch];
            if (cp == 0) cp = (UINT32)ch;   /* key has no Thai equivalent */
            append_utf8(buf, &len, max_len, cp);
            if (cp < 0x80) fb_putc((char)cp);
            else           fb_put_codepoint(cp);
        } else {
            if (len < max_len - 1) {
                buf[len++] = (char)ch;
                buf[len] = '\0';
                fb_putc((char)ch);          /* echo */
            }
        }
        cursor_show();
    }

    return len;
}

/* ----------------------------------------------------------------------------
 * generation loop
 * ------------------------------------------------------------------------- */

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    if (gGenCapture) { gGenLen = 0; gGenBuf[0] = '\0'; }   /* fresh capture */

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = AllocatePool((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fb_puts("error: something is wrong, expected at least 1 prompt token\n");
        return;
    }

    // start the main loop
    UINT64 start = 0;  // used to time our code, only initialized after first iteration
    int next;          // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;       // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1]; // still processing the input prompt
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_print(piece);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    if (!gGenCapture) {
        fb_putc('\n');

        // report achieved tok/s (pos-1 because the timer starts after first iteration)
        if (pos > 1) {
            UINT64 end = time_in_ms();
            fb_printf("achieved tok/s: %f\n",
                      (double)(pos - 1) / (double)(end - start) * 1000.0);
        }
    }

    FreePool(prompt_tokens);
}

/* ----------------------------------------------------------------------------
 * Knowledge-base lookup (LEARNING_SYSTEM_PLAN.md, ชั้นที่ 1)
 *
 * The 15M model can only memorise facts it was trained on, so a brand-new
 * question makes it hallucinate.  kb.txt — a UTF-8, tab-separated
 * "question<TAB>answer" file at the boot volume root — holds the facts as data
 * instead.  kb_lookup() matches a typed prompt against it (exact after
 * whitespace normalisation, then character n-gram Jaccard), and efi_main()
 * prints the stored answer directly when a match is found.
 * ------------------------------------------------------------------------- */

#define KB_MAX_ENTRIES 2048
#define KB_MAX_Q       192    /* max normalised code points per question */

typedef struct {
    int     qn;               /* normalised code-point length */
    UINT32  qcp[KB_MAX_Q];    /* normalised question code points */
    char   *a;                /* answer (points into the kb.txt buffer) */
} KBEntry;

static KBEntry *g_kb = NULL;
static int      g_kb_n = 0;
static VOID    *g_kb_data = NULL;   /* kb.txt buffer, freed at exit */

/* Decode UTF-8 into code points, skipping whitespace and '?'.  Returns the
 * number of code points (truncated at `max`). */
static int utf8_norm(const char *s, UINT32 *cp, int max) {
    int n = 0;
    while (*s && n < max) {
        unsigned char b = (unsigned char)*s;
        UINT32 c;
        int extra;
        if (b < 0x80)              { c = b;       extra = 0; }
        else if ((b & 0xE0) == 0xC0) { c = b & 0x1F; extra = 1; }
        else if ((b & 0xF0) == 0xE0) { c = b & 0x0F; extra = 2; }
        else if ((b & 0xF8) == 0xF0) { c = b & 0x07; extra = 3; }
        else { s++; continue; }    /* invalid lead byte: skip */
        s++;
        for (int i = 0; i < extra && *s; i++) {
            if (((unsigned char)*s & 0xC0) != 0x80) break;
            c = (c << 6) | ((unsigned char)*s & 0x3F);
            s++;
        }
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '?') continue;
        cp[n++] = c;
    }
    return n;
}

/* Character n-gram (trigram, bigram fallback) Jaccard similarity in [0,1]. */
static float ngram_jaccard(const UINT32 *a, int na, const UINT32 *b, int nb) {
    int n = 3;
    if (na < n || nb < n) n = 2;
    if (na < n || nb < n) return 0.0f;
    int ga = na - n + 1;
    int gb = nb - n + 1;
    if (ga <= 0 || gb <= 0) return 0.0f;
    int inter = 0;
    for (int i = 0; i < ga; i++) {
        for (int j = 0; j < gb; j++) {
            int k = 0;
            while (k < n && a[i + k] == b[j + k]) k++;
            if (k == n) { inter++; break; }
        }
    }
    int uni = ga + gb - inter;
    if (uni <= 0) return 0.0f;
    return (float)inter / (float)uni;
}

/* Return the index of the best-matching KB entry for `prompt`, or -1. */
static int kb_lookup(const char *prompt) {
    if (g_kb_n == 0) return -1;
    UINT32 pcp[KB_MAX_Q];
    int pn = utf8_norm(prompt, pcp, KB_MAX_Q);
    if (pn == 0) return -1;

    int best = -1;
    float best_score = 0.0f;
    for (int i = 0; i < g_kb_n; i++) {
        KBEntry *e = &g_kb[i];
        if (pn == e->qn) {
            int same = 1;
            for (int k = 0; k < pn; k++) {
                if (pcp[k] != e->qcp[k]) { same = 0; break; }
            }
            if (same) return i;   /* exact match always wins */
        }
        float s = ngram_jaccard(pcp, pn, e->qcp, e->qn);
        if (s > best_score) { best_score = s; best = i; }
    }
    return (best_score >= 0.60f) ? best : -1;
}

/* Load kb.txt from the boot volume root into g_kb.  Returns the entry count,
 * or -1 if the file is absent/empty (lookup then simply never matches). */
static int kb_load(EFI_FILE_HANDLE root) {
    VOID *data = NULL;
    UINTN size = 0;
    EFI_STATUS st = read_file(root, L"\\kb.txt", 0, &data, &size);
    if (EFI_ERROR(st) || data == NULL) return -1;
    if (size < 4) { FreePool(data); return -1; }

    g_kb_data = data;
    char *buf = (char *)data;
    g_kb = (KBEntry *)AllocateZeroPool(KB_MAX_ENTRIES * sizeof(KBEntry));
    if (g_kb == NULL) { return -1; }
    g_kb_n = 0;

    char *p = buf;
    char *end = buf + size;
    while (p < end && g_kb_n < KB_MAX_ENTRIES) {
        char *tab = p;
        while (tab < end && *tab != '\t' && *tab != '\n') tab++;
        if (tab >= end || *tab != '\t') {
            /* malformed line (no tab): skip to the next newline */
            while (p < end && *p != '\n') p++;
            if (p < end) p++;
            continue;
        }
        *tab = '\0';                 /* NUL-terminate the question */
        char *q = p;
        char *a = tab + 1;
        char *nl = a;
        while (nl < end && *nl != '\n') nl++;
        if (nl >= end) break;        /* last line lacks a newline (builder guarantees it) */
        *nl = '\0';                  /* NUL-terminate the answer */
        if (nl > a && nl[-1] == '\r') nl[-1] = '\0';   /* CRLF safety */

        if (*q != '\0' && *a != '\0') {
            KBEntry *e = &g_kb[g_kb_n];
            e->a = a;
            e->qn = utf8_norm(q, e->qcp, KB_MAX_Q);
            if (e->qn > 0) g_kb_n++;
        }
        p = nl + 1;
    }

    fb_printf("kb: %d entries loaded (%lu bytes)\n", g_kb_n, (UINT64)size);
    return g_kb_n;
}

/* Append one UTF-8 line to kb.txt on the boot volume (EFI_FILE_PROTOCOL write).
 * Returns 0 on success, -1 on failure (e.g. read-only volume). */
static int kb_append_file(EFI_FILE_HANDLE root, const char *line) {
    EFI_STATUS status;
    EFI_FILE_HANDLE file = NULL;
    EFI_FILE_INFO *info = NULL;
    UINTN info_size = SIZE_OF_EFI_FILE_INFO + 256;
    EFI_GUID fiGuid = EFI_FILE_INFO_ID;
    UINT64 mode = EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE;

    status = uefi_call_wrapper(root->Open, 5, root, &file, L"\\kb.txt", mode, 0);
    if (EFI_ERROR(status)) return -1;

    /* seek to end-of-file so the new line is appended, not overwriting */
    info = AllocateZeroPool(info_size);
    if (info != NULL) {
        status = uefi_call_wrapper(file->GetInfo, 4, file, &fiGuid, &info_size, info);
        if (!EFI_ERROR(status)) {
            uefi_call_wrapper(file->SetPosition, 2, file, (UINT64)info->FileSize);
        }
        FreePool(info);
    }

    UINTN want = strlen(line);
    UINTN len = want;
    status = uefi_call_wrapper(file->Write, 3, file, &len, (VOID *)line);
    int ok = (!EFI_ERROR(status) && len == want);
    uefi_call_wrapper(file->Flush, 1, file);
    uefi_call_wrapper(file->Close, 1, file);
    return ok ? 0 : -1;
}

/* "จำไว้: คำถาม = คำตอบ" — teach a new fact on-device (LEARNING_SYSTEM_PLAN.md
 * ชั้นที่ 3): append it to kb.txt on flash and add it to the in-memory KB so it
 * is answerable immediately, without any retraining. */
static void kb_remember(EFI_FILE_HANDLE root, char *arg) {
    /* skip leading spaces and an optional ':' */
    while (*arg == ' ' || *arg == '\t' || *arg == ':') arg++;

    char *eq = NULL;
    for (char *p = arg; *p; p++) { if (*p == '=') { eq = p; break; } }
    if (eq == NULL) {
        fb_puts("รูปแบบ: จำไว้: คำถาม = คำตอบ\n");
        return;
    }
    *eq = '\0';
    char *q = arg;
    char *a = eq + 1;

    /* trim ASCII whitespace around both sides */
    while (*a == ' ' || *a == '\t') a++;
    char *qe = q + strlen(q);
    while (qe > q && (qe[-1] == ' ' || qe[-1] == '\t')) { qe--; *qe = '\0'; }
    char *ae = a + strlen(a);
    while (ae > a && (ae[-1] == ' ' || ae[-1] == '\t')) { ae--; *ae = '\0'; }

    if (*q == '\0' || *a == '\0') {
        fb_puts("รูปแบบ: จำไว้: คำถาม = คำตอบ\n");
        return;
    }

    /* build the "question<TAB>answer\n" line and append it to kb.txt */
    char line[1024];
    int li = 0;
    for (char *p = q; *p && li < (int)sizeof(line) - 3; p++) line[li++] = *p;
    line[li++] = '\t';
    for (char *p = a; *p && li < (int)sizeof(line) - 3; p++) line[li++] = *p;
    line[li++] = '\n';
    line[li] = '\0';

    if (kb_append_file(root, line) != 0) {
        fb_puts("เขียนลง flash ไม่สำเร็จ (โวลุ่มอ่านอย่างเดียว?)\n");
        return;
    }

    /* update the in-memory KB so the fact answers right now */
    if (g_kb == NULL) {
        g_kb = (KBEntry *)AllocateZeroPool(KB_MAX_ENTRIES * sizeof(KBEntry));
    }
    if (g_kb != NULL && g_kb_n < KB_MAX_ENTRIES) {
        KBEntry *e = &g_kb[g_kb_n];
        e->qn = utf8_norm(q, e->qcp, KB_MAX_Q);
        char *copy = AllocatePool(strlen(a) + 1);
        if (copy != NULL && e->qn > 0) {
            strcpy(copy, a);
            e->a = copy;
            g_kb_n++;
        }
    }
    fb_puts("จำไว้แล้ว: ");
    fb_puts(a);
    fb_putc('\n');
}

/* ----------------------------------------------------------------------------
 * GUI layer: desktop + chat window + mouse pointer + clickable buttons
 *
 * Everything below draws straight into the GOP framebuffer (no window manager,
 * no OS).  Text reuses the same HRFONT renderer as the console; the mouse
 * pointer is a software cursor backed by a save/restore buffer so it never
 * leaves a trail.  The whole UI runs in one non-blocking event loop below.
 * ------------------------------------------------------------------------- */

#define MOUSE_CURSOR_W     16
#define MOUSE_CURSOR_H     16
#define MOUSE_SENSITIVITY  2     /* divide relative mickeys by this (tunable)  */

#define GUI_PAD            24    /* inner padding for window contents, px      */
#define GUI_POLL_US        10000 /* 10 ms idle yield between input polls       */
#define GUI_INPUT_LEN      512   /* editable input buffer, bytes               */
#define GUI_CHAT_MAX_LINES 48
#define GUI_CHAT_LINE_LEN  300   /* bytes: room for 80 three-byte Thai chars   */
#define GUI_CHAT_CHARS     80    /* characters per wrapped chat line           */
#define GUI_STATUS_H       40    /* top status bar height, px                  */
#define CARET_W            3     /* input caret (text cursor) width, px        */

/* Emotion panel (top-right): a drawn face + Thai caption for the AI's mood. */
#define EMOTION_FACE       32    /* face diameter, px                           */
#define EMOTION_PAD        10    /* inner padding inside the panel              */
#define EMOTION_GAP        10    /* gap between the face and its caption        */
#define EMOTION_PANEL_H    52    /* panel height, px                            */
#define EMOTION_THINK_US   600000 /* simulated "thinking" pause, microseconds   */
#define EMOTION_HOLD_LOOPS 250   /* answer mood -> idle after ~2.5 s (10ms ticks) */
#define GUI_GEN_STEPS      128   /* max tokens the model generates per answer    */

typedef struct { int x, y, w, h; } GuiRect;

/* The AI's on-screen mood: drawn as a face + Thai caption in the top-right. */
typedef enum {
    EMOTION_IDLE,       /* nobody is asking yet  */
    EMOTION_THINKING,   /* working on an answer  */
    EMOTION_HAPPY,      /* just answered         */
    EMOTION_CONFUSED,   /* did not find an answer */
} Emotion;

/* --- global state ---------------------------------------------------------- */

static EFI_SIMPLE_POINTER_PROTOCOL   *gMouse    = NULL;
static EFI_ABSOLUTE_POINTER_PROTOCOL *gAbsMouse = NULL;

/* 1 = a usable pointer (absolute tablet preferred, else relative mouse) exists */
static int mouse_available(void) { return gMouse != NULL || gAbsMouse != NULL; }

static int    gMouseX = 0, gMouseY = 0;           /* cursor top-left, absolute px */
static UINT32 gMouseSave[MOUSE_CURSOR_W * MOUSE_CURSOR_H]; /* pixels under cursor */
static int    gMouseSaved = 0;                    /* 1 = save buffer is valid     */
static int    gMouseLeftWasDown = 0;              /* click rising-edge detection   */

static UINT32 gColDesktop, gColChatBg, gColBorder, gColInputBg;
static UINT32 gColButtonBg, gColButtonFg, gColText, gColAccent, gColMouse;

static GuiRect gChatBox, gInputBox, gSendButton, gStatusBar;

static char gInputBuf[GUI_INPUT_LEN];
static int  gInputLen = 0;

static int    gCaretVisible = 1;     /* input caret on/off (blinks while idle) */
static UINT32 gCaretAccumUs = 0;      /* caret blink accumulator, microseconds  */

static Emotion gEmotion = EMOTION_IDLE; /* current on-screen mood                */
static int     gEmotionHoldLoops = 0;   /* ticks left before reverting to idle   */

static char gChatLog[GUI_CHAT_MAX_LINES][GUI_CHAT_LINE_LEN];
static int  gChatLines = 0;

static EFI_FILE_HANDLE gRoot = NULL;   /* set by gui_run(); used by gui_send() */

/* game mode: 0 = chat GUI, 1 = Pacman (entered via /games pacman) */
#define GAME_MODE_NONE   0
#define GAME_MODE_PACMAN 1
static int gGameMode = GAME_MODE_NONE;

/* --- live model state: built at boot by efi_main, swapped by /loadmodel ---- */
static Transformer gTransformer;          /* active checkpoint weights          */
static Tokenizer   gTokenizer;            /* tokenizer for the active vocab size */
static Sampler     gSampler;              /* sampling params + probindex buffer */
static VOID       *gTokenizerData = NULL; /* raw tokenizer.bin buffer (shared)  */
static int         gModelLoaded    = 0;   /* gTransformer holds a live model    */
static float       gTemperature    = 1.0f;
static float       gTopp           = 0.9f;
static unsigned long long gSeed    = 1337;

/* ----------------------------------------------------------------------------
 * 1. Mouse protocol: prefer EFI_SIMPLE_POINTER_PROTOCOL (relative mouse), fall
 *    back to EFI_ABSOLUTE_POINTER_PROTOCOL (tablet), and reset whichever wins.
 * ------------------------------------------------------------------------- */

static EFI_STATUS init_mouse(void) {
    /* Relative mouse first.  OVMF's UsbMouseAbsolutePointerDxe (the absolute
     * tablet driver) mis-handles QEMU's usb-tablet: it forces HID boot
     * protocol (which the tablet STALLs) and then reads the 5-byte absolute
     * report as a 3-byte boot-mouse report, so the cursor moves garbage.  The
     * relative SimplePointer is the one that actually works. */
    EFI_STATUS st = uefi_call_wrapper(BS->LocateProtocol, 3,
                                      &SimplePointerProtocol, NULL,
                                      (VOID **)&gMouse);
    if (!EFI_ERROR(st) && gMouse != NULL) {
        uefi_call_wrapper(gMouse->Reset, 2, gMouse, FALSE);
        return EFI_SUCCESS;
    }
    gMouse = NULL;

    st = uefi_call_wrapper(BS->LocateProtocol, 3,
                           &AbsolutePointerProtocol, NULL,
                           (VOID **)&gAbsMouse);
    if (EFI_ERROR(st) || gAbsMouse == NULL) {
        gAbsMouse = NULL;
        return EFI_UNSUPPORTED;
    }
    uefi_call_wrapper(gAbsMouse->Reset, 2, gAbsMouse, FALSE);
    return EFI_SUCCESS;
}

/* ----------------------------------------------------------------------------
 * 2. Software mouse cursor (ghosting prevention via save / draw / restore).
 * ------------------------------------------------------------------------- */

static UINT32 fb_get_pixel(int x, int y) {
    if (gFbBase == NULL || x < 0 || y < 0) return 0;
    if ((UINT32)x >= gFbWidth || (UINT32)y >= gFbHeight) return 0;
    return *(UINT32 *)(gFbBase + (UINTN)y * gFbScanLine * 4 + (UINTN)x * 4);
}

/* (a) Save the background under the cursor bounding box before drawing. */
static void mouse_save(void) {
    for (int r = 0; r < MOUSE_CURSOR_H; r++)
        for (int c = 0; c < MOUSE_CURSOR_W; c++)
            gMouseSave[r * MOUSE_CURSOR_W + c] = fb_get_pixel(gMouseX + c, gMouseY + r);
    gMouseSaved = 1;
}

/* (c) Restore the saved background when the cursor moves to a new position. */
static void mouse_restore(void) {
    if (!gMouseSaved) return;
    for (int r = 0; r < MOUSE_CURSOR_H; r++)
        for (int c = 0; c < MOUSE_CURSOR_W; c++)
            DrawPixel((UINT32)(gMouseX + c), (UINT32)(gMouseY + r),
                      gMouseSave[r * MOUSE_CURSOR_W + c]);
    gMouseSaved = 0;
}

/* Crosshair shape: a 2px '+' spanning the whole cell (1 = paint this pixel).
 * An arrow glyph could be swapped in here with a per-pixel bitmap the same way. */
static int mouse_cursor_pixel(int c, int r) {
    return (r == 7 || r == 8 || c == 7 || c == 8);
}

/* (b) Draw the pointer at the current position (call after mouse_save()). */
static void mouse_draw(void) {
    for (int r = 0; r < MOUSE_CURSOR_H; r++)
        for (int c = 0; c < MOUSE_CURSOR_W; c++)
            if (mouse_cursor_pixel(c, r))
                DrawPixel((UINT32)(gMouseX + c), (UINT32)(gMouseY + r), gColMouse);
}

/* The hot-spot (the point a click refers to) is the centre of the crosshair. */
static int mouse_hot_x(void) { return gMouseX + MOUSE_CURSOR_W / 2; }
static int mouse_hot_y(void) { return gMouseY + MOUSE_CURSOR_H / 2; }

/* Move the cursor to (nx, ny), clamped so the whole cursor stays on screen.
 * Erases the old cursor, saves the background at the new spot, then redraws. */
static void gui_draw_status(void);   /* forward decl: live coordinate readout */
static void mouse_move_to(int nx, int ny) {
    if (nx < 0) nx = 0;
    if (ny < 0) ny = 0;
    if (nx > (int)gFbWidth  - MOUSE_CURSOR_W) nx = (int)gFbWidth  - MOUSE_CURSOR_W;
    if (ny > (int)gFbHeight - MOUSE_CURSOR_H) ny = (int)gFbHeight - MOUSE_CURSOR_H;
    if (nx == gMouseX && ny == gMouseY) return;

    mouse_restore();            /* erase the old cursor, leaving no trail       */
    gMouseX = nx;
    gMouseY = ny;
    gui_draw_status();          /* refresh the coordinate readout (cursor hidden) */
    mouse_save();               /* remember what is under the new position      */
    mouse_draw();               /* paint the pointer on top                     */
}

/* ----------------------------------------------------------------------------
 * 3a. UI rendering engine: colours + rectangle primitives + text.
 * ------------------------------------------------------------------------- */

static void gui_init_colors(void) {
    gColDesktop  = fb_make_color(0x18, 0x22, 0x30);   /* dark slate blue        */
    gColChatBg   = fb_make_color(0x10, 0x14, 0x1A);   /* chat window interior   */
    gColBorder   = fb_make_color(0x3A, 0x4A, 0x60);   /* window borders         */
    gColInputBg  = fb_make_color(0x0C, 0x10, 0x16);   /* input box interior     */
    gColButtonBg = fb_make_color(0x2E, 0x8B, 0x57);   /* Send button (green)    */
    gColButtonFg = fb_make_color(0xFF, 0xFF, 0xFF);
    gColText     = fb_make_color(0xE6, 0xE6, 0xE6);
    gColAccent   = fb_make_color(0x6F, 0xC3, 0xDF);   /* mode chip / highlights */
    gColMouse    = fb_make_color(0xFF, 0xFF, 0xFF);
}

static void draw_rect(int x, int y, int w, int h, UINT32 color) {
    fb_fill((UINT32)x, (UINT32)y, (UINT32)w, 1, color);            /* top    */
    fb_fill((UINT32)x, (UINT32)(y + h - 1), (UINT32)w, 1, color);  /* bottom */
    fb_fill((UINT32)x, (UINT32)y, 1, (UINT32)h, color);            /* left   */
    fb_fill((UINT32)(x + w - 1), (UINT32)y, 1, (UINT32)h, color);  /* right  */
}

static void draw_filled_rect(int x, int y, int w, int h, UINT32 color) {
    fb_fill((UINT32)x, (UINT32)y, (UINT32)w, (UINT32)h, color);
}

/* Draw a UTF-8 string at pixel (px, py) with explicit fg/bg colours.  The
 * HRFONT renderer paints opaque cells using the global gBgColor, so it is
 * swapped around the call and restored afterwards. */
static void gui_text(int px, int py, const char *s, UINT32 fg, UINT32 bg) {
    UINT32 save_bg = gBgColor;
    gBgColor = bg;
    int cx = px, cy = py;
    hrf_puts(s, &cx, &cy, fg);
    gBgColor = save_bg;
}

/* Decimal string for a non-negative int, into a caller-provided buffer. */
static char *gui_itoa(int v, char *buf) {
    char tmp[12];
    int i = 0;
    do { tmp[i++] = (char)('0' + (v % 10)); v /= 10; } while (v > 0);
    int j = 0;
    while (i > 0) buf[j++] = tmp[--i];
    buf[j] = '\0';
    return buf;
}

/* Top status bar: shows the live pointer hot-spot coordinates, so the user can
 * always read exactly where the cursor is (ระบบบอกตำแหน่ง cursor). */
static void gui_draw_status(void) {
    draw_filled_rect(gStatusBar.x, gStatusBar.y, gStatusBar.w, gStatusBar.h,
                     gColInputBg);
    draw_rect(gStatusBar.x, gStatusBar.y, gStatusBar.w, gStatusBar.h, gColBorder);

    char buf[80];
    int n = 0;
    const char *p;
    for (p = "Mouse: X="; *p; p++) buf[n++] = *p;
    gui_itoa(mouse_hot_x(), buf + n); while (buf[n]) n++;
    for (p = "  Y="; *p; p++) buf[n++] = *p;
    gui_itoa(mouse_hot_y(), buf + n); while (buf[n]) n++;
    buf[n] = '\0';

    gui_text(gStatusBar.x + GUI_PAD, gStatusBar.y + (GUI_STATUS_H - HRFONT_H) / 2,
             buf, gColText, gColInputBg);
}

/* ----------------------------------------------------------------------------
 * 3a'. Emotion panel (top-right): a drawn face + Thai caption that reflects the
 *      AI's current mood (idle / thinking / answered / confused).
 * ------------------------------------------------------------------------- */

/* Display cells a UTF-8 string occupies (Thai combining marks have advance 0,
 * so byte length over-counts them). */
static int gui_text_cells(const char *s) {
    const char *p = s;
    int cells = 0;
    while (*p) {
        UINT32 cp = hrf_decode_utf8(&p);
        const HRFONT_glyph_t *g = hrf_lookup(cp);
        if (g != NULL && g->adv != 0) cells++;
    }
    return cells;
}

static const char *gui_emotion_text(Emotion e) {
    switch (e) {
    case EMOTION_THINKING: return "กำลังคิดหนัก...";
    case EMOTION_HAPPY:    return "ได้คำตอบแล้ว!";
    case EMOTION_CONFUSED: return "ไม่รู้เลย...";
    default:               return "รอคำถามอยู่...";
    }
}

/* 2px-thick arc: dir +1 = lower half (smile "⌣"), dir -1 = upper half (frown
 * "⌢").  Points where (r-th)^2 < x^2+y^2 <= r^2 on the chosen side. */
static void face_arc(int cx, int cy, int r, int dir, int th, UINT32 color) {
    for (int dy = 0; dy <= r; dy++) {
        int yy = cy + dir * dy;
        for (int dx = -r; dx <= r; dx++) {
            int d = dx * dx + dy * dy;
            if (d <= r * r && d > (r - th) * (r - th))
                DrawPixel((UINT32)(cx + dx), (UINT32)yy, color);
        }
    }
}

static void gui_draw_face(int cx, int cy, Emotion e) {
    int r = EMOTION_FACE / 2;
    UINT32 skin = fb_make_color(0xFF, 0xD9, 0x3D);
    UINT32 feat = fb_make_color(0x3A, 0x2A, 0x00);

    for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++)
            if (dx * dx + dy * dy <= r * r)
                DrawPixel((UINT32)(cx + dx), (UINT32)(cy + dy), skin);

    int ey = cy - 6;
    if (e == EMOTION_THINKING) ey = cy - 9;   /* eyes drift up while pondering */
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++)
            if (dx * dx + dy * dy <= 4) {
                DrawPixel((UINT32)(cx - 7 + dx), (UINT32)(ey + dy), feat);
                DrawPixel((UINT32)(cx + 7 + dx), (UINT32)(ey + dy), feat);
            }

    switch (e) {
    case EMOTION_HAPPY:
        face_arc(cx, cy + 3, 7, +1, 2, feat);          /* smile */
        break;
    case EMOTION_CONFUSED:
        face_arc(cx, cy + 11, 6, -1, 2, feat);         /* frown */
        break;
    case EMOTION_THINKING:                             /* small open "o" */
        for (int dy = -3; dy <= 3; dy++)
            for (int dx = -3; dx <= 3; dx++)
                if (dx * dx + dy * dy <= 9)
                    DrawPixel((UINT32)(cx + dx), (UINT32)(cy + 6 + dy), feat);
        break;
    default:                                           /* flat line (idle) */
        fb_fill((UINT32)(cx - 6), (UINT32)(cy + 6), 13, 2, feat);
        break;
    }
}

static void gui_draw_emotion(void) {
    const char *txt = gui_emotion_text(gEmotion);
    int w = EMOTION_PAD + EMOTION_FACE + EMOTION_GAP
          + gui_text_cells(txt) * HRFONT_W + EMOTION_PAD;
    int h = EMOTION_PANEL_H;
    int x = (int)gFbWidth - GUI_PAD - w;
    int y = GUI_STATUS_H + 8;

    UINT32 bg = fb_make_color(0x24, 0x2E, 0x3C);
    draw_filled_rect(x, y, w, h, bg);
    draw_rect(x, y, w, h, gColBorder);

    gui_draw_face(x + EMOTION_PAD + EMOTION_FACE / 2, y + h / 2, gEmotion);
    gui_text(x + EMOTION_PAD + EMOTION_FACE + EMOTION_GAP,
             y + (h - HRFONT_H) / 2, txt, gColText, bg);
}


/* ----------------------------------------------------------------------------
 * 3b. Layout, then the chat log, input box and Send button.
 * ------------------------------------------------------------------------- */

static void gui_layout(void) {
    int W = (int)gFbWidth, H = (int)gFbHeight;
    int m = 48;          /* outer margin */
    int ih = 112;        /* input box height */
    int gap = 40;        /* gap between chat window and input row */
    int btn_w = 336;     /* Send button width */
    int in_gap = 16;     /* gap between input box and Send button */

    gStatusBar.x = 0;
    gStatusBar.y = 0;
    gStatusBar.w = W;
    gStatusBar.h = GUI_STATUS_H;   /* top strip: live mouse-coordinate readout */

    gChatBox.x = m;
    gChatBox.y = m;
    gChatBox.w = W - 2 * m;
    gChatBox.h = H - 2 * m - ih - gap;

    gInputBox.x = m;
    gInputBox.y = H - m - ih;
    gInputBox.w = W - 2 * m - btn_w - in_gap;
    gInputBox.h = ih;

    gSendButton.x = W - m - btn_w;
    gSendButton.y = H - m - ih;
    gSendButton.w = btn_w;
    gSendButton.h = ih;
}

/* Bounded append of `line` into the chat log, scrolling when full. */
static void gui_chat_append(const char *line) {
    int dst = (gChatLines < GUI_CHAT_MAX_LINES) ? gChatLines : (GUI_CHAT_MAX_LINES - 1);

    if (gChatLines >= GUI_CHAT_MAX_LINES) {
        for (int i = 1; i < GUI_CHAT_MAX_LINES; i++)
            memcpy(gChatLog[i - 1], gChatLog[i], GUI_CHAT_LINE_LEN); /* drop oldest */
    }

    int n = 0;
    while (line[n] && n < GUI_CHAT_LINE_LEN - 1) { gChatLog[dst][n] = line[n]; n++; }
    gChatLog[dst][n] = '\0';

    if (gChatLines < GUI_CHAT_MAX_LINES) gChatLines++;
}

/* Append `text` to the chat, wrapping into GUI_CHAT_CHARS-character lines on
 * UTF-8 boundaries (never splits a multi-byte Thai character). */
static void gui_chat_add(const char *text) {
    const char *p = text;
    while (*p) {
        const char *seg = p;
        int chars = 0;
        while (*p && chars < GUI_CHAT_CHARS) {
            unsigned char b = (unsigned char)*p;
            int extra = 0;
            if      ((b & 0xE0) == 0xC0) extra = 1;
            else if ((b & 0xF0) == 0xE0) extra = 2;
            else if ((b & 0xF8) == 0xF0) extra = 3;
            p++;
            for (int i = 0; i < extra && *p && ((unsigned char)*p & 0xC0) == 0x80; i++)
                p++;
            chars++;
        }
        char line[GUI_CHAT_LINE_LEN];
        int n = 0;
        for (const char *q = seg; q < p && n < GUI_CHAT_LINE_LEN - 1; q++)
            line[n++] = *q;
        line[n] = '\0';
        gui_chat_append(line);
    }
}

/* Two-part append "a" then "b" (used for the "You: <prompt>" prefix). */
static void gui_chat_add2(const char *a, const char *b) {
    char tmp[GUI_INPUT_LEN + 8];
    int n = 0;
    for (const char *p = a; *p && n < (int)sizeof(tmp) - 1; p++) tmp[n++] = *p;
    for (const char *p = b; *p && n < (int)sizeof(tmp) - 1; p++) tmp[n++] = *p;
    tmp[n] = '\0';
    gui_chat_add(tmp);
}

static void gui_draw_send_button(void) {
    draw_filled_rect(gSendButton.x, gSendButton.y, gSendButton.w, gSendButton.h,
                     gColButtonBg);
    draw_rect(gSendButton.x, gSendButton.y, gSendButton.w, gSendButton.h,
              gColBorder);
    const char *label = "Send";
    int tw = (int)strlen(label) * HRFONT_W;
    int tx = gSendButton.x + (gSendButton.w - tw) / 2;
    int ty = gSendButton.y + (gSendButton.h - HRFONT_H) / 2;
    gui_text(tx, ty, label, gColButtonFg, gColButtonBg);
}

/* Column the input caret sits in: count spacing glyphs in gInputBuf so that
 * Thai combining marks (zero-width) do not shift the caret. */
static int gui_input_cells(void) {
    const char *p = gInputBuf;
    int cells = 0;
    while (*p) {
        UINT32 cp = hrf_decode_utf8(&p);
        const HRFONT_glyph_t *g = hrf_lookup(cp);
        if (g != NULL && g->adv != 0) cells++;
    }
    return cells;
}

/* Pixel x of the text caret, just past the last typed character. */
static int gui_caret_x(void) {
    return gInputBox.x + GUI_PAD + 3 * HRFONT_W + gui_input_cells() * HRFONT_W;
}

static void gui_draw_input(void) {
    draw_filled_rect(gInputBox.x, gInputBox.y, gInputBox.w, gInputBox.h, gColInputBg);
    draw_rect(gInputBox.x, gInputBox.y, gInputBox.w, gInputBox.h, gColBorder);
    int ty = gInputBox.y + (gInputBox.h - HRFONT_H) / 2;
    /* language-mode chip, then the text being typed */
    gui_text(gInputBox.x + GUI_PAD, ty, gThaiMode ? "TH" : "EN", gColAccent, gColInputBg);
    gui_text(gInputBox.x + GUI_PAD + 3 * HRFONT_W, ty, gInputBuf, gColText, gColInputBg);
    /* blinking text cursor so the insertion point is always visible */
    if (gCaretVisible) {
        int cx = gui_caret_x();
        if (cx + CARET_W <= gInputBox.x + gInputBox.w - 4)
            draw_filled_rect(cx, ty, CARET_W, HRFONT_H, gColText);
    }
}

static void gui_draw_chat(void) {
    draw_filled_rect(gChatBox.x, gChatBox.y, gChatBox.w, gChatBox.h, gColChatBg);
    int lx = gChatBox.x + GUI_PAD;
    int ly = gChatBox.y + GUI_PAD;
    for (int i = 0; i < gChatLines; i++) {
        int y = ly + i * HRFONT_H;
        if (y + HRFONT_H > gChatBox.y + gChatBox.h) break;   /* off the bottom */
        gui_text(lx, y, gChatLog[i], gColText, gColChatBg);
    }
    draw_rect(gChatBox.x, gChatBox.y, gChatBox.w, gChatBox.h, gColBorder);
}

/* 3c. draw_desktop(): background + chat window + input box + Send button. */
static void gui_draw_desktop(void) {
    fb_fill(0, 0, gFbWidth, gFbHeight, gColDesktop);
    gui_draw_status();
    gui_draw_chat();
    gui_draw_input();
    gui_draw_send_button();
    gui_draw_emotion();
}

/* Redraw helpers wrap the cursor save/restore so any repaint that could sit
 * under the pointer does not erase it (or leave a stale save buffer). */
static void gui_refresh_cursor(void) {
    if (!mouse_available()) return;  /* keyboard-only machine: no pointer */
    mouse_save();
    mouse_draw();
}

static void gui_repaint_input(void) {
    mouse_restore();
    gui_draw_input();
    gui_refresh_cursor();
}

static void gui_repaint_chat(void) {
    mouse_restore();
    gui_draw_chat();
    gui_draw_input();
    gui_draw_emotion();
    gui_refresh_cursor();
}

/* Change the on-screen mood and repaint the emotion panel immediately (used for
 * the transient "thinking" state while the KB/AI path runs). */
static void gui_set_emotion(Emotion e) {
    if (e == gEmotion) return;
    gEmotion = e;
    gEmotionHoldLoops = (e == EMOTION_HAPPY || e == EMOTION_CONFUSED)
                      ? EMOTION_HOLD_LOOPS : 0;
    mouse_restore();
    gui_draw_emotion();
    gui_refresh_cursor();
}

/* ----------------------------------------------------------------------------
 * Input caret blinking + UP/DOWN command history (GUI input box).
 * ------------------------------------------------------------------------- */

static void gui_caret_reset(void) { gCaretVisible = 1; gCaretAccumUs = 0; }

/* Toggle the input caret every CURSOR_BLINK_MS while idle. */
static void gui_caret_blink(void) {
    gCaretAccumUs += GUI_POLL_US;
    if (gCaretAccumUs >= (UINT32)(CURSOR_BLINK_MS * 1000)) {
        gCaretAccumUs = 0;
        gCaretVisible = !gCaretVisible;
        gui_repaint_input();   /* repaint to turn the caret bar on/off */
    }
}

/* Recall older prompts into the input box (UP arrow). */
static void gui_history_prev(void) {
    if (gHistoryCount == 0) return;
    if (gHistoryIndex == -1) {
        strcpy(gDraft, gInputBuf);              /* save the in-progress line */
        gHistoryIndex = gHistoryCount - 1;      /* newest entry */
    } else if (gHistoryIndex > 0) {
        gHistoryIndex--;                        /* step to an older entry */
    } else {
        return;                                 /* already at the oldest */
    }
    strcpy(gInputBuf, gHistory[gHistoryIndex]);
    gInputLen = (int)strlen(gInputBuf);
}

/* Recall newer prompts / restore the draft line (DOWN arrow). */
static void gui_history_next(void) {
    if (gHistoryIndex == -1) return;            /* not browsing */
    if (gHistoryIndex < gHistoryCount - 1) {
        gHistoryIndex++;                        /* step to a newer entry */
        strcpy(gInputBuf, gHistory[gHistoryIndex]);
    } else {
        gHistoryIndex = -1;                     /* back at newest -> restore draft */
        strcpy(gInputBuf, gDraft);
    }
    gInputLen = (int)strlen(gInputBuf);
}

/* ----------------------------------------------------------------------------
 * 4. Bounding-box click detection + the Send action.
 * ------------------------------------------------------------------------- */

static int is_clicked(int mx, int my, int bx, int by, int bw, int bh) {
    return mx >= bx && mx < bx + bw && my >= by && my < by + bh;
}

/* ----------------------------------------------------------------------------
 * Slash commands: '/'-prefixed input is intercepted before the KB / AI path.
 * ------------------------------------------------------------------------- */

static int gTrainingMode = 0;   /* placeholder state flag set by /train */

/* Split "cmd [arg...]" in place.  *cmd is the command token (NUL-terminated at
 * the first space); *arg is the remainder, or NULL when there is no argument. */
static void slash_split(char *s, char **cmd, char **arg) {
    *cmd = s;
    char *p = s;
    while (*p && *p != ' ') p++;   /* find the first space (or end) */
    if (*p == ' ') {
        *p = '\0';                 /* terminate the command token */
        p++;
        while (*p == ' ') p++;     /* skip spaces before the argument */
        *arg = p;
    } else {
        *arg = NULL;
    }
}

/* Turn an ASCII filename into a "\\name" CHAR16 path (the boot volume is flat,
 * so every model sits in the root).  Returns 0 if the name is empty/non-ASCII. */
static int make_model_path(CHAR16 *out, int cap, const char *name) {
    if (name == NULL || *name == '\0') return 0;
    int i = 0;
    out[i++] = L'\\';
    while (i < cap - 1 && *name) {
        if ((unsigned char)*name >= 0x80) { out[i] = 0; return 0; }
        out[i++] = (CHAR16)(unsigned char)*name++;
    }
    out[i] = 0;
    return i;
}

/* True if `name` (in the boot-volume root) starts with the Q8_0 v2 model magic,
 * so /listmodel shows only real checkpoints (not tokenizer.bin). */
static int is_model_checkpoint(const char *name) {
    CHAR16 path[256];
    if (make_model_path(path, 256, name) <= 1) return 0;
    EFI_FILE_HANDLE f = NULL;
    EFI_STATUS st = uefi_call_wrapper(gRoot->Open, 5, gRoot, &f, path,
                                      EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(st) || f == NULL) return 0;
    UINT8 hdr[4];
    UINTN sz = 4;
    st = uefi_call_wrapper(f->Read, 3, f, &sz, hdr);
    uefi_call_wrapper(f->Close, 1, f);
    if (EFI_ERROR(st) || sz < 4) return 0;
    UINT32 magic = 0;
    memcpy(&magic, hdr, sizeof(magic));
    return magic == 0x616b3432;
}

/* Enumerate "*.bin" files in the boot-volume root into `names` (ASCII) and
 * return the count (capped at `max`).  Uses directory Read on the root handle,
 * resetting its position so repeated /listmodel calls start from the top. */
static int list_model_files(char names[][64], int max) {
    UINTN buf_size = SIZE_OF_EFI_FILE_INFO + 256;
    EFI_FILE_INFO *info = AllocateZeroPool(buf_size);
    if (info == NULL) return 0;

    uefi_call_wrapper(gRoot->SetPosition, 2, gRoot, (UINT64)0);

    int n = 0;
    while (n < max) {
        UINTN sz = buf_size;
        EFI_STATUS st = uefi_call_wrapper(gRoot->Read, 3, gRoot, &sz, info);
        if (st == EFI_BUFFER_TOO_SMALL) {
            FreePool(info);
            buf_size = sz;
            info = AllocateZeroPool(buf_size);
            if (info == NULL) return n;
            continue;
        }
        if (EFI_ERROR(st) || sz == 0) break;   /* end of directory            */
        if (info->Attribute & EFI_FILE_DIRECTORY) continue;

        char ascii[64]; int len = 0, ok = 1;
        for (int i = 0; info->FileName[i]; i++) {
            if (info->FileName[i] >= 0x80) { ok = 0; break; }  /* non-ASCII name */
            if (len < 63) ascii[len++] = (char)info->FileName[i];
        }
        ascii[len] = '\0';
        if (ok && len >= 4 && strcmp(ascii + len - 4, ".bin") == 0
                 && is_model_checkpoint(ascii)) {
            strcpy(names[n], ascii);
            n++;
        }
    }
    FreePool(info);
    return n;
}

/* Swap the active checkpoint: read `filename` off the boot volume, build the new
 * Transformer, and rebuild the Tokenizer/Sampler only if the vocab size changed.
 * The old model is freed only after the new one is fully built, so a bad
 * checkpoint leaves the current model untouched. */
static void load_new_model(const char *filename) {
    CHAR16 path[256];
    if (make_model_path(path, 256, filename) <= 1) {
        gui_chat_add2("> bad model name: ", filename);
        return;
    }
    gui_chat_add2("> loading model: ", filename);

    VOID *data = NULL;
    UINTN size = 0;
    EFI_STATUS st = read_file(gRoot, path, 1, &data, &size);
    if (EFI_ERROR(st) || data == NULL) {
        gui_chat_add("> error: could not read checkpoint");
        return;
    }

    Transformer next;
    if (build_transformer(&next, data, size) != 0) {
        uefi_call_wrapper(BS->FreePages, 2, (UINTN)data, EFI_SIZE_TO_PAGES(size));
        gui_chat_add("> error: invalid model checkpoint");
        return;
    }

    /* Rebuild tokenizer/sampler only when the vocab size actually changes
     * (tokenizer.bin holds one vocab; both shipped models use 32000). */
    if (gModelLoaded && next.config.vocab_size != gTokenizer.vocab_size) {
        free_tokenizer(&gTokenizer);
        build_tokenizer(&gTokenizer, gTokenizerData, next.config.vocab_size);
        free_sampler(&gSampler);
        build_sampler(&gSampler, next.config.vocab_size, gTemperature, gTopp, gSeed);
    }

    if (gModelLoaded) free_transformer(&gTransformer);
    gTransformer = next;
    gModelLoaded = 1;

    char buf[96];
    int n = 0;
    const char *p;
    for (p = "> model loaded OK (dim="; *p; p++) buf[n++] = *p;
    gui_itoa(gTransformer.config.dim, buf + n); while (buf[n]) n++;
    for (p = " vocab="; *p; p++) buf[n++] = *p;
    gui_itoa(gTransformer.config.vocab_size, buf + n); while (buf[n]) n++;
    buf[n++] = ')'; buf[n] = '\0';
    gui_chat_add(buf);
}

/* ----------------------------------------------------------------------------
 * 4b. Pacman minigame (/games pacman).  A grid maze on the raw framebuffer:
 *     arrow keys steer Pacman, ghosts chase, dots/pellets score, ESC returns.
 * ------------------------------------------------------------------------- */

#define PAC_W 23
#define PAC_H 23
#define PAC_CELL       48      /* pixels per maze cell                         */
#define PAC_TICK_US    160000  /* ~6.25 moves/sec                              */
#define PAC_FRIGHT_TICKS 40    /* ticks ghosts stay edible after a pellet      */
#define PAC_MAX_GHOSTS 4

#define PAC_SCAN_UP    0x01
#define PAC_SCAN_DOWN  0x02
#define PAC_SCAN_RIGHT 0x03
#define PAC_SCAN_LEFT  0x04
#define PAC_SCAN_ESC   0x17

/* 23x23 maze: '#' wall, '.' dot, 'o' power pellet, 'P' Pacman, 'G' ghost. */
static const char *gPacMaze[PAC_H] = {
    "#######################",
    "#.o.#.....#.....#.#.#.#",
    "##G....#...#.#.....oG.#",
    "#.......#.#.#...#...#.#",
    "##...#................#",
    "#...#...#.#...#...#.#.#",
    "#..#.#.###.#...#......#",
    "#.....#.#.......#.#.#.#",
    "#..#.......#.#.###....#",
    "#...#.#.#.#.....#...#.#",
    "#.....................#",
    "#.#.#.....#P#.#...#.#.#",
    "####.#.#.#.#...#.###..#",
    "#.#.............#.#.#.#",
    "#....#.#...#.#........#",
    "#.#.#...#.....#.#.#.#.#",
    "#....#.....#..........#",
    "#.#.#...#...#.#...#.#.#",
    "#....#................#",
    "#.o.#...#.#.#.#.#...o.#",
    "#.G....#...#.#......G.#",
    "#.#.#.........#.#.#.#.#",
    "#######################",
};

static char gPacGrid[PAC_H][PAC_W];      /* mutable copy: walls + remaining dots */

static int gPacX, gPacY;                 /* Pacman cell                          */
static int gPacDirX, gPacDirY;           /* current facing                       */
static int gPacNextX, gPacNextY;         /* queued (buffered) turn               */
static int gPacSpawnX, gPacSpawnY;

static int gPacLives, gPacScore, gPacDotsLeft;
static int gPacOver, gPacWon;

static int gPacGhosts;
static int gPacGhostX[PAC_MAX_GHOSTS], gPacGhostY[PAC_MAX_GHOSTS];
static int gPacGhostDX[PAC_MAX_GHOSTS], gPacGhostDY[PAC_MAX_GHOSTS];
static int gPacGhostSpawnX[PAC_MAX_GHOSTS], gPacGhostSpawnY[PAC_MAX_GHOSTS];
static int gPacFright;                   /* remaining edible ticks (0 = normal) */

static UINT64 gPacAccum;                 /* tick accumulator, microseconds       */
static int gPacOX, gPacOY;               /* maze origin, px                      */

static UINT32 gPacBg, gPacWall, gPacWallHi, gPacDot, gPacPellet, gPacPac;
static UINT32 gPacGhostCol[PAC_MAX_GHOSTS], gPacFrightCol, gPacWhite, gPacBlack;
static UINT32 gPacText;

static int pac_abs(int v) { return v < 0 ? -v : v; }
static int pac_wall(int x, int y) {
    if (x < 0 || y < 0 || x >= PAC_W || y >= PAC_H) return 1;
    return gPacGrid[y][x] == '#';
}

static void pac_fill_circle(int cx, int cy, int r, UINT32 c) {
    for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++)
            if (dx * dx + dy * dy <= r * r)
                DrawPixel((UINT32)(cx + dx), (UINT32)(cy + dy), c);
}

/* Filled triangle via bounding box + edge side test. */
static void pac_fill_tri(int x0, int y0, int x1, int y1, int x2, int y2, UINT32 c) {
    int minx = x0 < x1 ? (x0 < x2 ? x0 : x2) : (x1 < x2 ? x1 : x2);
    int maxx = x0 > x1 ? (x0 > x2 ? x0 : x2) : (x1 > x2 ? x1 : x2);
    int miny = y0 < y1 ? (y0 < y2 ? y0 : y2) : (y1 < y2 ? y1 : y2);
    int maxy = y0 > y1 ? (y0 > y2 ? y0 : y2) : (y1 > y2 ? y1 : y2);
    for (int y = miny; y <= maxy; y++)
        for (int x = minx; x <= maxx; x++) {
            int a = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0);
            int b = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1);
            int cc = (x0 - x2) * (y - y2) - (y0 - y2) * (x - x2);
            if (!((a < 0 || b < 0 || cc < 0) && (a > 0 || b > 0 || cc > 0)))
                DrawPixel((UINT32)x, (UINT32)y, c);
        }
}

static void pac_draw_cell(int x, int y) {
    int px = gPacOX + x * PAC_CELL, py = gPacOY + y * PAC_CELL;
    int cx = px + PAC_CELL / 2, cy = py + PAC_CELL / 2;
    char c = gPacGrid[y][x];
    if (c == '#') {
        draw_filled_rect(px + 1, py + 1, PAC_CELL - 2, PAC_CELL - 2, gPacWall);
        draw_filled_rect(px + 2, py + 2, PAC_CELL - 4, 5, gPacWallHi);
        draw_filled_rect(px + 2, py + 2, 5, PAC_CELL - 4, gPacWallHi);
    } else if (c == '.') {
        pac_fill_circle(cx, cy, 5, gPacDot);
    } else if (c == 'o') {
        pac_fill_circle(cx, cy, 12, gPacPellet);
    }
}

static void pac_draw_pacman(void) {
    int cx = gPacOX + gPacX * PAC_CELL + PAC_CELL / 2;
    int cy = gPacOY + gPacY * PAC_CELL + PAC_CELL / 2;
    pac_fill_circle(cx, cy, PAC_CELL / 2 - 4, gPacPac);
    int dx = gPacDirX, dy = gPacDirY;
    if (dx == 0 && dy == 0) { dx = 1; dy = 0; }
    int px = -dy, py = dx;               /* perpendicular of the facing vector */
    pac_fill_tri(cx, cy,
                 cx + dx * 10 + px * 10, cy + dy * 10 + py * 10,
                 cx + dx * 10 - px * 10, cy + dy * 10 - py * 10,
                 gPacBg);
}

static void pac_draw_ghost(int i) {
    int cx = gPacOX + gPacGhostX[i] * PAC_CELL + PAC_CELL / 2;
    int cy = gPacOY + gPacGhostY[i] * PAC_CELL + PAC_CELL / 2;
    UINT32 body = gPacFright ? gPacFrightCol : gPacGhostCol[i];
    int r = PAC_CELL / 2 - 9;
    pac_fill_circle(cx, cy - 5, r, body);
    draw_filled_rect(cx - r, cy - 5, 2 * r, r + 9, body);
    pac_fill_circle(cx - r + 4, cy + r + 2, 4, body);   /* three wavy "feet" */
    pac_fill_circle(cx,           cy + r + 2, 4, body);
    pac_fill_circle(cx + r - 4, cy + r + 2, 4, body);
    pac_fill_circle(cx - 7, cy - 7, 5, gPacWhite);      /* eyes               */
    pac_fill_circle(cx + 7, cy - 7, 5, gPacWhite);
    int pr = gPacFright ? 2 : 3;
    pac_fill_circle(cx - 7, cy - 7, pr, gPacBlack);
    pac_fill_circle(cx + 7, cy - 7, pr, gPacBlack);
}

static void pac_draw_hud(void) {
    char buf[48];
    int n;
    const char *p;

    gui_text((int)(gFbWidth - 6 * HRFONT_W) / 2, 40, "PACMAN", gPacText, gPacBg);

    n = 0;
    for (p = "SCORE "; *p; p++) buf[n++] = *p;
    gui_itoa(gPacScore, buf + n); while (buf[n]) n++;
    buf[n] = '\0';
    gui_text(gPacOX, 104, buf, gPacText, gPacBg);

    n = 0;
    for (p = "LIVES "; *p; p++) buf[n++] = *p;
    gui_itoa(gPacLives, buf + n); while (buf[n]) n++;
    buf[n] = '\0';
    int lw = (int)strlen(buf) * HRFONT_W;
    gui_text(gPacOX + PAC_W * PAC_CELL - lw, 104, buf, gPacText, gPacBg);

    const char *hint = "Arrows / WASD = move    Q or ESC = quit";
    int hw = (int)strlen(hint) * HRFONT_W;
    gui_text((int)(gFbWidth - hw) / 2, gPacOY + PAC_H * PAC_CELL + 40,
             hint, gPacText, gPacBg);
}

static void pac_draw_over(void) {
    const char *msg = gPacWon ? "YOU WIN!" : "GAME OVER";
    const char *sub = "press ESC to return to chat";
    int mw = (int)strlen(msg) * HRFONT_W;
    int sw = (int)strlen(sub) * HRFONT_W;
    int w = (mw > sw ? mw : sw) + 64;
    int h = 96;
    int x = (int)(gFbWidth - w) / 2;
    int y = gPacOY + PAC_H * PAC_CELL / 2 - h / 2;
    UINT32 bg = fb_make_color(0x10, 0x14, 0x1A);
    draw_filled_rect(x, y, w, h, bg);
    draw_rect(x, y, w, h, gColBorder);
    gui_text(x + (w - mw) / 2, y + 16, msg, gPacPac, bg);
    gui_text(x + (w - sw) / 2, y + 16 + HRFONT_H, sub, gColText, bg);
}

static void pac_draw(void) {
    fb_fill(0, 0, gFbWidth, gFbHeight, gPacBg);
    for (int y = 0; y < PAC_H; y++)
        for (int x = 0; x < PAC_W; x++)
            pac_draw_cell(x, y);
    for (int i = 0; i < gPacGhosts; i++) pac_draw_ghost(i);
    pac_draw_pacman();
    pac_draw_hud();
    if (gPacOver) pac_draw_over();
}

static void pac_respawn(void) {
    gPacX = gPacSpawnX; gPacY = gPacSpawnY;
    gPacDirX = 1; gPacDirY = 0; gPacNextX = 1; gPacNextY = 0;
    for (int i = 0; i < gPacGhosts; i++) {
        gPacGhostX[i] = gPacGhostSpawnX[i];
        gPacGhostY[i] = gPacGhostSpawnY[i];
        gPacGhostDX[i] = 1; gPacGhostDY[i] = 0;
    }
    gPacFright = 0;
}

static void pac_init(void) {
    gPacOX = ((int)gFbWidth - PAC_W * PAC_CELL) / 2;
    gPacOY = 180;
    gPacLives = 3;
    gPacScore = 0;
    gPacDotsLeft = 0;
    gPacOver = 0;
    gPacWon = 0;
    gPacFright = 0;
    gPacAccum = 0;
    gPacDirX = 1; gPacDirY = 0; gPacNextX = 1; gPacNextY = 0;
    gPacGhosts = 0;

    for (int y = 0; y < PAC_H; y++) {
        for (int x = 0; x < PAC_W; x++) {
            char c = gPacMaze[y][x];
            if (c == 'P') {
                gPacX = x; gPacY = y; gPacSpawnX = x; gPacSpawnY = y;
                c = ' ';
            } else if (c == 'G') {
                if (gPacGhosts < PAC_MAX_GHOSTS) {
                    gPacGhostX[gPacGhosts] = x; gPacGhostY[gPacGhosts] = y;
                    gPacGhostSpawnX[gPacGhosts] = x; gPacGhostSpawnY[gPacGhosts] = y;
                    gPacGhostDX[gPacGhosts] = 1; gPacGhostDY[gPacGhosts] = 0;
                    gPacGhosts++;
                }
                c = ' ';
            }
            if (c == '.' || c == 'o') gPacDotsLeft++;
            gPacGrid[y][x] = c;
        }
    }

    gPacBg        = fb_make_color(0x00, 0x00, 0x00);
    gPacWall      = fb_make_color(0x1F, 0x33, 0x99);
    gPacWallHi    = fb_make_color(0x4A, 0x6C, 0xE0);
    gPacDot       = fb_make_color(0xF6, 0xD8, 0x8A);
    gPacPellet    = fb_make_color(0xFF, 0xE0, 0x9A);
    gPacPac       = fb_make_color(0xFF, 0xE0, 0x00);
    gPacGhostCol[0] = fb_make_color(0xFF, 0x3B, 0x3B);
    gPacGhostCol[1] = fb_make_color(0xFF, 0xB8, 0xDE);
    gPacGhostCol[2] = fb_make_color(0x3B, 0xFF, 0xFF);
    gPacGhostCol[3] = fb_make_color(0xFF, 0xB8, 0x5A);
    gPacFrightCol = fb_make_color(0x20, 0x40, 0xC0);
    gPacWhite     = fb_make_color(0xFF, 0xFF, 0xFF);
    gPacBlack     = fb_make_color(0x00, 0x00, 0x00);
    gPacText      = fb_make_color(0xE6, 0xE6, 0xE6);

    if (mouse_available()) mouse_restore();   /* erase pointer while playing */
    pac_draw();
}

static void pac_move_pacman(void) {
    int ox = gPacX, oy = gPacY;
    /* apply a buffered turn first, then keep going in the facing direction */
    if (gPacNextX || gPacNextY) {
        int tx = gPacX + gPacNextX, ty = gPacY + gPacNextY;
        if (!pac_wall(tx, ty)) {
            gPacX = tx; gPacY = ty;
            gPacDirX = gPacNextX; gPacDirY = gPacNextY;
        }
    }
    if (gPacX == ox && gPacY == oy) {   /* buffered turn was blocked: keep facing */
        int tx = gPacX + gPacDirX, ty = gPacY + gPacDirY;
        if (!pac_wall(tx, ty)) { gPacX = tx; gPacY = ty; }
    }
    char c = gPacGrid[gPacY][gPacX];
    if (c == '.') {
        gPacGrid[gPacY][gPacX] = ' ';
        gPacScore += 10; gPacDotsLeft--;
    } else if (c == 'o') {
        gPacGrid[gPacY][gPacX] = ' ';
        gPacScore += 50; gPacDotsLeft--;
        gPacFright = PAC_FRIGHT_TICKS;
    }
}

static void pac_move_ghost(int i) {
    static const int cd[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int x = gPacGhostX[i], y = gPacGhostY[i];
    int best = -1, bestscore = -1000000000;
    for (int k = 0; k < 4; k++) {
        int nx = x + cd[k][0], ny = y + cd[k][1];
        if (pac_wall(nx, ny)) continue;
        if (nx == x - gPacGhostDX[i] && ny == y - gPacGhostDY[i]) continue; /* no U-turn */
        int score;
        if (gPacFright)
            score = (nx * 7 + ny * 13) % 4;                 /* scatter / wander */
        else
            score = 1000 - (pac_abs(nx - gPacX) + pac_abs(ny - gPacY));  /* chase */
        if (cd[k][0] == gPacGhostDX[i] && cd[k][1] == gPacGhostDY[i]) score += 50;
        if (score > bestscore) { bestscore = score; best = k; }
    }
    if (best < 0) {                             /* only a U-turn is possible */
        for (int k = 0; k < 4; k++) {
            if (!pac_wall(x + cd[k][0], y + cd[k][1])) { best = k; break; }
        }
    }
    if (best < 0) return;                       /* trapped (should not happen) */
    gPacGhostDX[i] = cd[best][0];
    gPacGhostDY[i] = cd[best][1];
    gPacGhostX[i] = x + gPacGhostDX[i];
    gPacGhostY[i] = y + gPacGhostDY[i];
}

static void pac_collide(void) {
    for (int i = 0; i < gPacGhosts; i++) {
        if (gPacGhostX[i] == gPacX && gPacGhostY[i] == gPacY) {
            if (gPacFright) {
                gPacScore += 200;
                gPacGhostX[i] = gPacGhostSpawnX[i];
                gPacGhostY[i] = gPacGhostSpawnY[i];
                gPacGhostDX[i] = 1; gPacGhostDY[i] = 0;
            } else {
                gPacLives--;
                if (gPacLives <= 0) { gPacOver = 1; gPacWon = 0; return; }
                pac_respawn();
                return;
            }
        }
    }
    if (gPacDotsLeft <= 0) { gPacOver = 1; gPacWon = 1; }
}

static void pac_tick(void) {
    if (gPacOver) return;
    pac_move_pacman();
    for (int i = 0; i < gPacGhosts; i++) pac_move_ghost(i);
    if (gPacFright) gPacFright--;
    pac_collide();
    pac_draw();
}

static void pac_quit(void) {
    gGameMode = GAME_MODE_NONE;
    gui_draw_desktop();
    if (mouse_available()) { mouse_save(); mouse_draw(); }
}

static void pac_key(EFI_INPUT_KEY key) {
    CHAR16 ch = key.UnicodeChar;

    if (gPacOver) {
        if (key.ScanCode == PAC_SCAN_ESC || ch == CHAR_CARRIAGE_RETURN
            || ch == 'q' || ch == 'Q' || ch == 0x1B) pac_quit();
        return;
    }

    /* arrow keys (EFI scan codes) */
    switch (key.ScanCode) {
    case PAC_SCAN_LEFT:  gPacNextX = -1; gPacNextY =  0; return;
    case PAC_SCAN_RIGHT: gPacNextX =  1; gPacNextY =  0; return;
    case PAC_SCAN_UP:    gPacNextX =  0; gPacNextY = -1; return;
    case PAC_SCAN_DOWN:  gPacNextX =  0; gPacNextY =  1; return;
    case PAC_SCAN_ESC:   pac_quit(); return;
    default: break;
    }

    /* WASD fallback (UnicodeChar) — reliable on any keyboard, incl. QMP tests */
    switch (ch) {
    case 'w': case 'W': gPacNextX =  0; gPacNextY = -1; break;
    case 's': case 'S': gPacNextX =  0; gPacNextY =  1; break;
    case 'a': case 'A': gPacNextX = -1; gPacNextY =  0; break;
    case 'd': case 'D': gPacNextX =  1; gPacNextY =  0; break;
    case 'q': case 'Q': case 0x1B: pac_quit(); break;
    default: break;
    }
}

static void process_slash_command(char *input) {
    if (input[0] != '/') return;

    char *cmd, *arg;
    slash_split(input + 1, &cmd, &arg);   /* skip the leading '/' */

    if (strcmp(cmd, "reboot") == 0) {
        gui_chat_add("> rebooting...");
        uefi_call_wrapper(RT->ResetSystem, 4, EfiResetCold, EFI_SUCCESS, 0, NULL);
        gui_chat_add("> reboot failed (ResetSystem returned)");
    } else if (strcmp(cmd, "shutdown") == 0) {
        gui_chat_add("> powering off...");
        uefi_call_wrapper(RT->ResetSystem, 4, EfiResetShutdown, EFI_SUCCESS, 0, NULL);
        gui_chat_add("> shutdown failed (ResetSystem returned)");
    } else if (strcmp(cmd, "clear") == 0) {
        gChatLines = 0;               /* empty the log; gui_send repaints it */
    } else if (strcmp(cmd, "help") == 0) {
        gui_chat_add("> available commands:");
        gui_chat_add(">   /reboot          cold reboot");
        gui_chat_add(">   /shutdown        power off");
        gui_chat_add(">   /clear           clear this chat");
        gui_chat_add(">   /help            show this list");
        gui_chat_add(">   /loadmodel FILE  load a model checkpoint");
        gui_chat_add(">   /listmodel       list available models");
        gui_chat_add(">   /train           enter training mode");
        gui_chat_add(">   /games pacman    play Pacman");
    } else if (strcmp(cmd, "loadmodel") == 0) {
        if (arg && *arg) load_new_model(arg);
        else             gui_chat_add("> usage: /loadmodel <filename>");
    } else if (strcmp(cmd, "listmodel") == 0) {
        char names[16][64];
        int cnt = list_model_files(names, 16);
        if (cnt == 0) {
            gui_chat_add("> no models found (*.bin in boot root)");
        } else {
            gui_chat_add("> models:");
            for (int i = 0; i < cnt; i++) gui_chat_add2(">   ", names[i]);
        }
    } else if (strcmp(cmd, "train") == 0) {
        gTrainingMode = 1;
        gui_chat_add("> training mode ON (placeholder)");
    } else if (strcmp(cmd, "games") == 0) {
        if (arg && strcmp(arg, "pacman") == 0) {
            gGameMode = GAME_MODE_PACMAN;
            pac_init();
        } else {
            gui_chat_add("> usage: /games pacman");
        }
    } else {
        gui_chat_add2("> unknown command: /", cmd);
    }
}

static void gui_send(void) {
    if (gInputLen == 0) return;
    gInputBuf[gInputLen] = '\0';

    history_push(gInputBuf);   /* save for UP/DOWN recall (slash commands too) */

    /* Slash commands are intercepted here, before the KB / AI path. */
    if (gInputBuf[0] == '/') {
        process_slash_command(gInputBuf);
    } else {
        gui_chat_add2("You: ", gInputBuf);

        gui_set_emotion(EMOTION_THINKING);              /* "thinking hard"   */

        if (strncmp(gInputBuf, "จำไว้", strlen("จำไว้")) == 0) {
            kb_remember(gRoot, gInputBuf + strlen("จำไว้"));   /* teach a new fact */
            uefi_call_wrapper(BS->Stall, 1, EMOTION_THINK_US);
            gui_chat_add("จำไว้แล้ว");
            gui_set_emotion(EMOTION_HAPPY);
        } else {
            int ki = kb_lookup(gInputBuf);                     /* answer from KB   */
            if (ki >= 0) {
                uefi_call_wrapper(BS->Stall, 1, EMOTION_THINK_US);
                gui_chat_add(g_kb[ki].a);
                gui_set_emotion(EMOTION_HAPPY);
            } else {
                /* KB miss: fall back to real model generation (captured into
                 * gGenBuf, then routed into the chat log). */
                gui_repaint_chat();   /* show "You: ..." + thinking face first */
                gGenCapture = 1;
                generate(&gTransformer, &gTokenizer, &gSampler, gInputBuf, GUI_GEN_STEPS);
                gGenCapture = 0;
                if (gGenLen > 0) {
                    gui_chat_add(gGenBuf);
                    gui_set_emotion(EMOTION_HAPPY);
                } else {
                    gui_chat_add("ไม่ทราบ");
                    gui_set_emotion(EMOTION_CONFUSED);
                }
            }
        }
    }

    gInputLen = 0;
    gInputBuf[0] = '\0';
    if (gGameMode == GAME_MODE_NONE)
        gui_repaint_chat();   /* also repaints the now-empty input box */
}

static void gui_click(int mx, int my) {
    if (is_clicked(mx, my, gSendButton.x, gSendButton.y, gSendButton.w, gSendButton.h))
        gui_send();
}

/* ----------------------------------------------------------------------------
 * 5. Unified non-blocking event loop (keyboard + mouse + idle yield).
 * ------------------------------------------------------------------------- */

static void gui_handle_key(EFI_INPUT_KEY key) {
    gui_caret_reset();   /* any key wakes the caret back to solid */

    if (key.ScanCode == KBD_SCAN_F1) {
        gThaiMode = !gThaiMode;
        gui_repaint_input();
        return;
    }

    if (key.ScanCode == KBD_SCAN_UP) {          /* recall older prompt */
        gui_history_prev();
        gui_repaint_input();
        return;
    }
    if (key.ScanCode == KBD_SCAN_DOWN) {        /* recall newer prompt / draft */
        gui_history_next();
        gui_repaint_input();
        return;
    }

    CHAR16 ch = key.UnicodeChar;

    if (ch == CHAR_CARRIAGE_RETURN) { gui_send(); return; }

    if (ch == CHAR_BACKSPACE) {
        if (gInputLen > 0) {
            gHistoryIndex = -1;                 /* editing leaves history browsing */
            /* remove one full UTF-8 character (Thai chars are 3 bytes) */
            gInputLen--;
            gInputBuf[gInputLen] = '\0';
            while (gInputLen > 0 && ((unsigned char)gInputBuf[gInputLen - 1] & 0xC0) == 0x80) {
                gInputLen--;
                gInputBuf[gInputLen] = '\0';
            }
            gui_repaint_input();
        }
        return;
    }

    if (key.ScanCode != 0) return;            /* other F-keys / Esc */
    if (ch < 0x20 || ch >= 0x7f) return;      /* drop control characters      */

    gHistoryIndex = -1;                       /* typing leaves history browsing */

    if (gThaiMode) {
        UINT32 cp = gKedmanee[(unsigned char)ch];
        if (cp == 0) cp = (UINT32)ch;
        append_utf8(gInputBuf, &gInputLen, GUI_INPUT_LEN, cp);
    } else if (gInputLen < GUI_INPUT_LEN - 1) {
        gInputBuf[gInputLen++] = (char)ch;
        gInputBuf[gInputLen] = '\0';
    }
    gui_repaint_input();
}

static void gui_poll_mouse(void) {
    if (gAbsMouse != NULL) {
        /* Absolute tablet: map CurrentX/Y from [AbsoluteMin, AbsoluteMax] onto
         * the framebuffer, then reuse the same move/click path as the mouse. */
        EFI_ABSOLUTE_POINTER_STATE st;
        EFI_STATUS s = uefi_call_wrapper(gAbsMouse->GetState, 2, gAbsMouse, &st);
        if (s == EFI_NOT_READY || EFI_ERROR(s)) return;

        UINT64 minx = 0, maxx = 0x7FFF, miny = 0, maxy = 0x7FFF;
        if (gAbsMouse->Mode != NULL) {
            minx = gAbsMouse->Mode->AbsoluteMinX;
            maxx = gAbsMouse->Mode->AbsoluteMaxX;
            miny = gAbsMouse->Mode->AbsoluteMinY;
            maxy = gAbsMouse->Mode->AbsoluteMaxY;
        }
        UINT64 spanx = maxx > minx ? (maxx - minx) : 1;
        UINT64 spany = maxy > miny ? (maxy - miny) : 1;
        UINT64 cx = st.CurrentX >= minx ? (st.CurrentX - minx) : 0;
        UINT64 cy = st.CurrentY >= miny ? (st.CurrentY - miny) : 0;
        int nx = (int)(cx * (UINT64)(gFbWidth  - 1) / spanx);
        int ny = (int)(cy * (UINT64)(gFbHeight - 1) / spany);
        mouse_move_to(nx, ny);

        int left = (st.ActiveButtons & EFI_ABSP_TouchActive) != 0;
        if (left && !gMouseLeftWasDown)      /* rising edge = a fresh click */
            gui_click(mouse_hot_x(), mouse_hot_y());
        gMouseLeftWasDown = left;
        return;
    }

    if (gMouse == NULL) return;

    EFI_SIMPLE_POINTER_STATE st;
    EFI_STATUS s = uefi_call_wrapper(gMouse->GetState, 2, gMouse, &st);
    if (s == EFI_NOT_READY || EFI_ERROR(s)) return;

    int dx = (int)st.RelativeMovementX / MOUSE_SENSITIVITY;
    int dy = (int)st.RelativeMovementY / MOUSE_SENSITIVITY;
    if (dx != 0 || dy != 0)
        mouse_move_to(gMouseX + dx, gMouseY + dy);

    int left = (st.LeftButton != 0);
    if (left && !gMouseLeftWasDown)      /* rising edge = a fresh click */
        gui_click(mouse_hot_x(), mouse_hot_y());
    gMouseLeftWasDown = left;
}

static void gui_init(void) {
    gui_init_colors();
    gui_layout();
    init_mouse();
    if (mouse_available()) {        /* position the pointer at screen centre   */
        gMouseX = (int)gFbWidth / 2;
        gMouseY = (int)gFbHeight / 2;
    }
    gui_draw_desktop();             /* status bar shows the initial coordinates */
    if (mouse_available()) {
        mouse_save();
        mouse_draw();
    }
}

static void gui_run(SIMPLE_INPUT_INTERFACE *ConIn, EFI_FILE_HANDLE root) {
    gRoot = root;
    for (;;) {
        /* Pacman mode: run the game instead of the chat loop. */
        if (gGameMode == GAME_MODE_PACMAN) {
            EFI_INPUT_KEY key;
            EFI_STATUS ks = uefi_call_wrapper(ConIn->ReadKeyStroke, 2, ConIn, &key);
            if (ks == EFI_SUCCESS) pac_key(key);
            gPacAccum += GUI_POLL_US;
            if (gPacAccum >= PAC_TICK_US) {
                gPacAccum -= PAC_TICK_US;
                pac_tick();
            }
            uefi_call_wrapper(BS->Stall, 1, GUI_POLL_US);
            continue;
        }

        /* 1. keyboard: non-blocking read; act only when a key is present */
        EFI_INPUT_KEY key;
        EFI_STATUS ks = uefi_call_wrapper(ConIn->ReadKeyStroke, 2, ConIn, &key);
        if (ks == EFI_SUCCESS)
            gui_handle_key(key);

        /* 2. mouse: poll relative movement and button clicks */
        gui_poll_mouse();

        /* 3. blink the input caret while idle */
        gui_caret_blink();

        /* 4. revert a transient answer-mood back to idle once the hold expires */
        if (gEmotionHoldLoops > 0 && --gEmotionHoldLoops == 0)
            gui_set_emotion(EMOTION_IDLE);

        /* 5. idle yield so we do not spin the CPU at 100% */
        uefi_call_wrapper(BS->Stall, 1, GUI_POLL_US);
    }
}

/* ----------------------------------------------------------------------------
 * UEFI entry point
 * ------------------------------------------------------------------------- */

EFI_STATUS efi_main(EFI_HANDLE ImageHandle, EFI_SYSTEM_TABLE *SystemTable) {

    InitializeLib(ImageHandle, SystemTable);

    // Disable the UEFI watchdog timer.  OVMF's BDS arms a 5-minute watchdog
    // before invoking this boot option (UEFI spec 7.5 SetWatchdogTimer); if we
    // never cancel it, the firmware resets/shuts the machine down ~5 minutes in
    // — which looks like a spontaneous "crash" while the GUI idles.
    uefi_call_wrapper(BS->SetWatchdogTimer, 4, 0, 0, 0, NULL);

    // Switch to a graphics mode and take the framebuffer before printing
    // anything, so every message from here on lands on the GOP display.  (If
    // this fails, Simple Text Output is still valid and Print still works.)
    EFI_STATUS status = init_gop();
    if (EFI_ERROR(status)) {
        Print(L"error: could not initialize GOP / framebuffer (status %r)\n", status);
        return EFI_UNSUPPORTED;
    }

    fb_printf("llama2.c bare-metal UEFI inference\n");
    fb_printf("framebuffer %u x %u (%lu bytes, scanline %u px)\n",
              gFbWidth, gFbHeight, gFbSize, gFbScanLine);

    fb_puts("\n>> ระบบแสดงผลภาษาไทยพร้อมใช้งาน: สวัสดี Bare-Metal AI <<");
    fb_puts("\n>> กด F1 สลับภาษาไทย / อังกฤษ  ออกโปรแกรมให้กด enter<<\n\n");
    

    // --- tune these for your hardware ---
    CHAR16 *checkpoint_name = L"\\stories15M_thai_q80.bin";
    CHAR16 *tokenizer_name  = L"\\tokenizer.bin";
    int     steps           = 256;      // number of tokens to generate
    gTemperature = 1.0f;                // 0.0 = greedy
    gTopp        = 0.9f;                // nucleus sampling
    gSeed        = 1337;                // bare metal has no time(NULL); use a fixed seed

    // locate the volume this image was loaded from and open its root directory
    EFI_FILE_HANDLE root = open_root_volume(ImageHandle);
    if (root == NULL) {
        fb_puts("error: could not open the boot volume\n");
        return EFI_LOAD_ERROR;
    }

    // read the checkpoint into page-aligned memory (AllocatePages; see read_file)
    VOID *checkpoint_data = NULL;
    UINTN checkpoint_size = 0;
    status = read_file(root, checkpoint_name, 1, &checkpoint_data, &checkpoint_size);
    if (EFI_ERROR(status) || checkpoint_data == NULL) {
        fb_printf("error: could not read %s (status 0x%lx)\n",
                  checkpoint_name, (UINT64)status);
        return EFI_LOAD_ERROR;
    }
    fb_printf("loaded %s (%lu bytes = %lu MiB)\n",
              checkpoint_name, checkpoint_size, checkpoint_size / (1024 * 1024));

    // build the Transformer from the in-memory checkpoint
    if (build_transformer(&gTransformer, checkpoint_data, checkpoint_size) != 0) {
        fb_puts("error: invalid model checkpoint\n");
        return EFI_LOAD_ERROR;
    }
    if (steps == 0 || steps > gTransformer.config.seq_len) {
        steps = gTransformer.config.seq_len; // override to ~max length
    }
    fb_printf("config: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq_len=%d\n",
              gTransformer.config.dim, gTransformer.config.hidden_dim,
              gTransformer.config.n_layers, gTransformer.config.n_heads,
              gTransformer.config.n_kv_heads, gTransformer.config.vocab_size,
              gTransformer.config.seq_len);

    // read and build the tokenizer
    VOID *tokenizer_data = NULL;
    UINTN tokenizer_size = 0;
    status = read_file(root, tokenizer_name, 0, &tokenizer_data, &tokenizer_size);
    if (EFI_ERROR(status) || tokenizer_data == NULL) {
        fb_printf("error: could not read %s (status 0x%lx)\n",
                  tokenizer_name, (UINT64)status);
        return EFI_LOAD_ERROR;
    }
    gTokenizerData = tokenizer_data;
    build_tokenizer(&gTokenizer, tokenizer_data, gTransformer.config.vocab_size);

    // build the Sampler
    build_sampler(&gSampler, gTransformer.config.vocab_size, gTemperature, gTopp, gSeed);
    gModelLoaded = 1;

    // load the knowledge base; a matched fact is answered verbatim (no
    // generation), so known questions stop being hallucinated
    kb_load(root);
    fb_printf("\n"); // added new line befor ai prompt

    // GUI: draw the desktop and run the unified keyboard + mouse event loop.
    // The loop runs until the machine is reset (no text-mode exit anymore).
    gui_init();
    gui_run(SystemTable->ConIn, root);

    // memory and file handle cleanup
    free_sampler(&gSampler);
    free_tokenizer(&gTokenizer);
    free_transformer(&gTransformer);
    FreePool(gTokenizerData);
    // checkpoint_data is freed by free_transformer()
    if (g_kb)        FreePool(g_kb);
    if (g_kb_data)   FreePool(g_kb_data);

    return EFI_SUCCESS;
}
