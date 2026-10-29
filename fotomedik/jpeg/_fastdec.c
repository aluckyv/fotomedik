/* FotoMedik fast baseline entropy-decoding kernel.
 *
 * Decodes up to `max_mcus` MCUs of one restart interval starting at a bit
 * position, mirroring fotomedik/jpeg/decoder.py exactly (same errors, same
 * positions). Called through ctypes; the Python decoder keeps all the
 * structure (intervals, markers, resync) and only hands the hot loop here.
 *
 * Build:  gcc -O2 -shared -fPIC -o _fastdec.so _fastdec.c      (Linux)
 *         cl /O2 /LD _fastdec.c /Fe_fastdec.dll                (Windows, MSVC)
 *         gcc -O2 -shared -o _fastdec.dll _fastdec.c           (Windows, mingw)
 */
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

enum { ERR_NONE = 0, ERR_BAD_HUFFMAN = 1, ERR_COEF_OVERFLOW = 2, ERR_DC_RANGE = 3, ERR_MARKER = 4, ERR_EOF = 5 };

typedef struct {
    int32_t h, v, dc_t, ac_t, qdc, cid;
} comp_t;

typedef struct {
    const uint8_t *data;
    int64_t pos, end;
    uint64_t buf;
    int nbits, real;
    int hit;          /* 1 if a marker stopped filling */
    int hit_marker;   /* marker byte */
    int64_t hit_off;  /* offset of the marker's FF */
    int eof;
    int64_t bytepos[8]; /* ring of positions of loaded real bytes */
    int bp_n;
} reader_t;

static void fill(reader_t *r, int need) {
    while (r->nbits < need) {
        if (r->hit || r->eof) {
            r->buf = (r->buf << 8) & 0xFFFFFFFFFFFFULL;
            r->nbits += 8;
            continue;
        }
        int64_t p = r->pos;
        if (p >= r->end) { r->eof = 1; continue; }
        uint8_t b = r->data[p];
        if (b == 0xFF) {
            int64_t q = p + 1;
            while (q < r->end && r->data[q] == 0xFF) q++;
            if (q >= r->end) { r->eof = 1; continue; }
            uint8_t nxt = r->data[q];
            if (nxt == 0x00) {
                r->pos = q + 1;
            } else {
                r->hit = 1; r->hit_marker = nxt; r->hit_off = p;
                continue;
            }
        } else {
            r->pos = p + 1;
        }
        r->buf = ((r->buf << 8) | b) & 0xFFFFFFFFFFFFULL;
        r->nbits += 8;
        r->real += 8;
        if (r->bp_n < 8) r->bytepos[r->bp_n++] = p;
        else { memmove(r->bytepos, r->bytepos + 1, 7 * sizeof(int64_t)); r->bytepos[7] = p; }
    }
}

/* exact (byte, bit) of next unread real bit */
static void tell(reader_t *r, int64_t *off, int *bit) {
    if (r->real <= 0) { *off = r->hit ? r->hit_off : r->pos; *bit = 0; return; }
    int k = (r->real + 7) / 8;
    if (k > r->bp_n) { *off = r->bp_n ? r->bytepos[0] : r->pos; *bit = 0; return; }
    *off = r->bytepos[r->bp_n - k];
    *bit = 8 - (r->real - 8 * (k - 1));
}

static inline uint32_t peek16(reader_t *r) {
    if (r->nbits < 16) fill(r, 16);
    return (uint32_t)((r->buf >> (r->nbits - 16)) & 0xFFFF);
}

/* returns 0 ok, else error code */
static inline int skip(reader_t *r, int n) {
    if (r->nbits < n) fill(r, n);
    if (r->real < n) return r->hit ? ERR_MARKER : ERR_EOF;
    r->nbits -= n; r->real -= n;
    return 0;
}

static inline int read_bits(reader_t *r, int n, int32_t *out) {
    if (n == 0) { *out = 0; return 0; }
    if (r->nbits < n) fill(r, n);
    if (r->real < n) return r->hit ? ERR_MARKER : ERR_EOF;
    r->nbits -= n; r->real -= n;
    *out = (int32_t)((r->buf >> r->nbits) & ((1u << n) - 1));
    return 0;
}

/* Huffman tables: 8 slots (cls*4+id); fast[slot*512], maxcode[slot*17], valptr, mincode, symbols[slot*256] */
static inline int decode_symbol(reader_t *r, int slot, const int32_t *fast, const int32_t *maxcode,
                                const int32_t *valptr, const int32_t *mincode, const uint8_t *symbols, int *sym) {
    uint32_t code16 = peek16(r);
    int32_t e = fast[slot * 512 + (code16 >> 7)];
    if (e) {
        int err = skip(r, e >> 8);
        if (err) return err;
        *sym = e & 0xFF;
        return 0;
    }
    for (int length = 10; length <= 16; length++) {
        int32_t c = (int32_t)(code16 >> (16 - length));
        if (c <= maxcode[slot * 17 + length]) {
            int err = skip(r, length);
            if (err) return err;
            *sym = symbols[slot * 256 + valptr[slot * 17 + length] + c - mincode[slot * 17 + length]];
            return 0;
        }
    }
    return ERR_BAD_HUFFMAN;
}

/* Returns number of MCUs decoded. Fills out[]:
 *  out[0]=err, out[1]=err_byte, out[2]=err_bit, out[3]=marker (if ERR_MARKER), out[4]=end_byte, out[5]=end_bit
 * dc_vals: int32 [max_mcus * blocks_per_mcu] in scan order; mcu_bitstart: int64 [max_mcus] */
EXPORT int64_t decode_mcus(const uint8_t *data, int64_t start_byte, int start_bit, int64_t end,
                           const comp_t *comps, int ncomp,
                           const int32_t *fast, const int32_t *maxcode, const int32_t *valptr,
                           const int32_t *mincode, const uint8_t *symbols,
                           int64_t max_mcus, int32_t *preds,
                           int64_t *mcu_bitstart, int32_t *dc_vals, int64_t *out) {
    reader_t r;
    memset(&r, 0, sizeof r);
    r.data = data; r.pos = start_byte; r.end = end;
    int err = 0;
    if (start_bit) { int32_t dummy; err = read_bits(&r, start_bit, &dummy); }
    int blocks_per_mcu = 0;
    for (int c = 0; c < ncomp; c++) blocks_per_mcu += comps[c].h * comps[c].v;
    int64_t n = 0;
    int64_t eb; int ebit;
    for (n = 0; n < max_mcus && !err; n++) {
        tell(&r, &eb, &ebit);
        mcu_bitstart[n] = eb * 8 + ebit;
        int bi = 0;
        for (int c = 0; c < ncomp && !err; c++) {
            const comp_t *cp = &comps[c];
            int nb = cp->h * cp->v;
            for (int b = 0; b < nb && !err; b++) {
                int t;
                err = decode_symbol(&r, cp->dc_t, fast, maxcode, valptr, mincode, symbols, &t);
                if (err) break;
                if (t > 11) { err = ERR_BAD_HUFFMAN; break; }
                int32_t v;
                err = read_bits(&r, t, &v);
                if (err) break;
                if (t && v < (1 << (t - 1))) v -= (1 << t) - 1;
                preds[c] += v;
                int32_t dcv = preds[c];
                int64_t mag = (int64_t)dcv * cp->qdc;
                if (mag < 0) mag = -mag;
                if (mag > 1024 + cp->qdc) { err = ERR_DC_RANGE; break; }
                int k = 1;
                while (k < 64) {
                    int rs;
                    err = decode_symbol(&r, 4 + cp->ac_t, fast, maxcode, valptr, mincode, symbols, &rs);
                    if (err) break;
                    int s = rs & 15, run = rs >> 4;
                    if (s == 0) {
                        if (run == 15) { k += 16; continue; }
                        break;
                    }
                    k += run;
                    if (k > 63) { err = ERR_COEF_OVERFLOW; break; }
                    if (s > 10) { err = ERR_BAD_HUFFMAN; break; }
                    err = skip(&r, s);
                    if (err) break;
                    k++;
                }
                if (err) break;
                dc_vals[n * blocks_per_mcu + bi] = dcv;
                bi++;
            }
        }
        if (err) break;
    }
    out[0] = err;
    tell(&r, &eb, &ebit);
    if (err == ERR_MARKER) { out[1] = r.hit_off; out[2] = 0; out[3] = r.hit_marker; }
    else { out[1] = eb; out[2] = ebit; out[3] = 0; }
    out[4] = eb; out[5] = ebit;
    return n;
}
