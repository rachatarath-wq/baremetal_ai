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
    fb_puts(piece);
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

/* Read one line of text from the UEFI console into buf (max `max_len` bytes
 * including the NUL).  Echoes printable characters, handles Backspace, toggles
 * Thai input with F1, and returns when the user presses Enter.  Arrows, and
 * other F-keys / Delete / Esc are ignored.  Returns the number of bytes stored. */
static int read_line(SIMPLE_INPUT_INTERFACE *ConIn, char *buf, int max_len) {
    int len = 0;
    buf[0] = '\0';

    for (;;) {
        EFI_INPUT_KEY key;
        EFI_STATUS status = uefi_call_wrapper(ConIn->ReadKeyStroke, 2, ConIn, &key);

        /* ReadKeyStroke is non-blocking: it returns EFI_NOT_READY while the
         * input queue is empty, so spin until a key actually arrives. */
        if (status == EFI_NOT_READY) continue;
        if (EFI_ERROR(status)) continue;

        /* F1 toggles English <-> Thai (Kedmanee) input mode. */
        if (key.ScanCode == KBD_SCAN_F1) {
            gThaiMode = !gThaiMode;
            fb_puts(gThaiMode ? "[TH]" : "[EN]");
            continue;
        }

        CHAR16 ch = key.UnicodeChar;

        if (ch == CHAR_CARRIAGE_RETURN) {
            fb_putc('\n');
            break;                          /* Enter */
        }

        if (ch == CHAR_BACKSPACE) {
            if (len > 0) {
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
            continue;
        }

        if (key.ScanCode != 0) {
            continue;                       /* arrows / other F-keys / Delete / Esc */
        }

        /* printable characters only (drop control characters) */
        if (ch < 0x20 || ch >= 0x7f) continue;

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
    }

    return len;
}

/* ----------------------------------------------------------------------------
 * generation loop
 * ------------------------------------------------------------------------- */

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

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
    fb_putc('\n');

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        UINT64 end = time_in_ms();
        fb_printf("achieved tok/s: %f\n",
                  (double)(pos - 1) / (double)(end - start) * 1000.0);
    }

    FreePool(prompt_tokens);
}

/* ----------------------------------------------------------------------------
 * UEFI entry point
 * ------------------------------------------------------------------------- */

EFI_STATUS efi_main(EFI_HANDLE ImageHandle, EFI_SYSTEM_TABLE *SystemTable) {

    InitializeLib(ImageHandle, SystemTable);

    // Switch to a graphics mode and take the framebuffer before printing
    // anything, so every message from here on lands on the GOP display.  (If
    // this fails, Simple Text Output is still valid and Print still works.)
    EFI_STATUS status = init_gop();
    if (EFI_ERROR(status)) {
        Print(L"error: could not initialize GOP / framebuffer (status %r)\n", status);
        return EFI_UNSUPPORTED;
    }

    fb_printf("llama2.c bare-metal UEFI inference\n");
    fb_puts("\n>> ระบบแสดงผลภาษาไทยพร้อมใช้งาน: สวัสดี Bare-Metal AI <<\n\n");
    fb_printf("framebuffer %u x %u (%lu bytes, scanline %u px)\n",
              gFbWidth, gFbHeight, gFbSize, gFbScanLine);

    // --- tune these for your hardware ---
    CHAR16 *checkpoint_name = L"\\stories15M_thai_q80.bin";
    CHAR16 *tokenizer_name  = L"\\tokenizer.bin";
    char    prompt[512];               // filled in from the keyboard at runtime
    int     steps           = 256;      // number of tokens to generate
    float   temperature     = 1.0f;     // 0.0 = greedy
    float   topp            = 0.9f;     // nucleus sampling
    unsigned long long seed = 1337;     // bare metal has no time(NULL); use a fixed seed

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
    Transformer transformer;
    if (build_transformer(&transformer, checkpoint_data, checkpoint_size) != 0) {
        fb_puts("error: invalid model checkpoint\n");
        return EFI_LOAD_ERROR;
    }
    if (steps == 0 || steps > transformer.config.seq_len) {
        steps = transformer.config.seq_len; // override to ~max length
    }
    fb_printf("config: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq_len=%d\n",
              transformer.config.dim, transformer.config.hidden_dim,
              transformer.config.n_layers, transformer.config.n_heads,
              transformer.config.n_kv_heads, transformer.config.vocab_size,
              transformer.config.seq_len);

    // read and build the tokenizer
    VOID *tokenizer_data = NULL;
    UINTN tokenizer_size = 0;
    status = read_file(root, tokenizer_name, 0, &tokenizer_data, &tokenizer_size);
    if (EFI_ERROR(status) || tokenizer_data == NULL) {
        fb_printf("error: could not read %s (status 0x%lx)\n",
                  tokenizer_name, (UINT64)status);
        return EFI_LOAD_ERROR;
    }
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_data, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, seed);

    // interactive loop: ask for a prompt, generate, then repeat until the
    // user presses Enter on an empty line to quit
    for (;;) {
        fb_printf("Enter prompt [%a] (F1 toggles EN/TH) (empty to quit): ",
                  gThaiMode ? "TH" : "EN");
        read_line(SystemTable->ConIn, prompt, sizeof(prompt));
        if (prompt[0] == '\0') break;
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    }

    // memory and file handle cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    FreePool(tokenizer_data);
    // checkpoint_data is freed by free_transformer()

    return EFI_SUCCESS;
}
