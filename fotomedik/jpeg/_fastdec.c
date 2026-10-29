/* FotoMedik fast baseline entropy-decoding kernel.
 *
 * decode_mcus(): decodes up to `max_mcus` MCUs of one restart interval from a
 *   bit position. Mirrors fotomedik/jpeg/kernel.py:py_decode_mcus exactly
 *   (same errors, same positions).
 * eval_flips(): the bit-flip search loop. For each candidate (offset, mask) it
 *   flips the byte in place, decodes (cheap stage 1, then to the end of the
 *   interval for survivors), restores the byte, and reports how far decoding
 *   got, whether it landed exactly on the restart marker, whether the final
 *   byte is padded with 1-bits as the standard requires, and a DC-smoothness
 *   score. Mirrors kernel.py:py_eval_flips.
 *
 * Called through ctypes; Python keeps all structure (intervals, markers,
 * resync, candidate selection) and hands only the hot loops here.
 *
 * Build:  python build_fastdec.py
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#define FASTDEC_ABI 11  /* bump whenever an exported signature changes; fastdec.py checks it */

EXPORT int fastdec_abi(void) { return FASTDEC_ABI; }

static inline int popcount8(unsigned v) { int c = 0; while (v) { c += v & 1; v >>= 1; } return c; }

enum { ERR_NONE = 0, ERR_BAD_HUFFMAN = 1, ERR_COEF_OVERFLOW = 2, ERR_DC_RANGE = 3, ERR_MARKER = 4, ERR_EOF = 5 };

typedef struct {
    int32_t h, v, dc_t, ac_t, qdc, cid;
} comp_t;

typedef struct {
    const int32_t *fast;    /* [8][512]  (cls*4+id) */
    const int32_t *maxcode; /* [8][17] */
    const int32_t *valptr;  /* [8][17] */
    const int32_t *mincode; /* [8][17] */
    const uint8_t *symbols; /* [8][256] */
} tables_t;

typedef struct {
    const uint8_t *data;
    int64_t pos, end;
    uint64_t buf;
    int nbits, real;
    int hit;          /* 1 if a marker stopped filling */
    int hit_marker;   /* marker byte */
    int64_t hit_off;  /* offset of the marker's FF */
    int eof;
    int64_t bytepos[8]; /* positions of loaded real bytes, newest last */
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

static inline int decode_symbol(reader_t *r, int slot, const tables_t *t, int *sym) {
    uint32_t code16 = peek16(r);
    int32_t e = t->fast[slot * 512 + (code16 >> 7)];
    if (e) {
        int err = skip(r, e >> 8);
        if (err) return err;
        *sym = e & 0xFF;
        return 0;
    }
    for (int length = 10; length <= 16; length++) {
        int32_t c = (int32_t)(code16 >> (16 - length));
        if (c <= t->maxcode[slot * 17 + length]) {
            int err = skip(r, length);
            if (err) return err;
            *sym = t->symbols[slot * 256 + t->valptr[slot * 17 + length] + c - t->mincode[slot * 17 + length]];
            return 0;
        }
    }
    return ERR_BAD_HUFFMAN;
}

/* Decode one MCU; DC values go to dc_out[0..bpm). Returns 0 or an error code. */
static inline int decode_mcu(reader_t *r, const comp_t *comps, int ncomp, const tables_t *t,
                             int32_t *preds, int32_t *dc_out, int check_dc) {
    int err = 0;
    int bi = 0;
    for (int c = 0; c < ncomp; c++) {
        const comp_t *cp = &comps[c];
        int nb = cp->h * cp->v;
        for (int b = 0; b < nb; b++) {
            int s0;
            err = decode_symbol(r, cp->dc_t, t, &s0);
            if (err) return err;
            if (s0 > 11) return ERR_BAD_HUFFMAN;
            int32_t v;
            err = read_bits(r, s0, &v);
            if (err) return err;
            if (s0 && v < (1 << (s0 - 1))) v -= (1 << s0) - 1;
            preds[c] += v;
            int32_t dcv = preds[c];
            int64_t mag = (int64_t)dcv * cp->qdc;
            if (mag < 0) mag = -mag;
            if (check_dc && mag > 1024 + cp->qdc) return ERR_DC_RANGE;
            int k = 1;
            while (k < 64) {
                int rs;
                err = decode_symbol(r, 4 + cp->ac_t, t, &rs);
                if (err) return err;
                int s = rs & 15, run = rs >> 4;
                if (s == 0) {
                    if (run == 15) { k += 16; continue; }
                    break;
                }
                k += run;
                if (k > 63) return ERR_COEF_OVERFLOW;
                if (s > 10) return ERR_BAD_HUFFMAN;
                err = skip(r, s);
                if (err) return err;
                k++;
            }
            dc_out[bi++] = dcv;
        }
    }
    return 0;
}

static void reader_init(reader_t *r, const uint8_t *data, int64_t start_byte, int64_t end) {
    memset(r, 0, sizeof *r);
    r->data = data; r->pos = start_byte; r->end = end;
}

/* Core loop. Returns MCUs decoded; fills out[0..5] as documented on decode_mcus. */
static int64_t run_decode(const uint8_t *data, int64_t start_byte, int start_bit, int64_t end,
                          const comp_t *comps, int ncomp, const tables_t *t,
                          int64_t max_mcus, int32_t *preds,
                          int64_t *mcu_bitstart, int32_t *dc_vals, int64_t *out, int check_dc) {
    reader_t r;
    reader_init(&r, data, start_byte, end);
    int err = 0;
    if (start_bit) { int32_t dummy; err = read_bits(&r, start_bit, &dummy); }
    int bpm = 0;
    for (int c = 0; c < ncomp; c++) bpm += comps[c].h * comps[c].v;
    int64_t n = 0;
    int64_t eb; int ebit;
    for (n = 0; n < max_mcus && !err; n++) {
        if (mcu_bitstart) { tell(&r, &eb, &ebit); mcu_bitstart[n] = eb * 8 + ebit; }
        err = decode_mcu(&r, comps, ncomp, t, preds, dc_vals + n * bpm, check_dc);
        if (err) break;
    }
    out[0] = err;
    tell(&r, &eb, &ebit);
    if (err == ERR_MARKER) { out[1] = r.hit_off; out[2] = 0; out[3] = r.hit_marker; }
    else { out[1] = eb; out[2] = ebit; out[3] = 0; }
    out[4] = eb; out[5] = ebit;
    return n;
}

/* out[0]=err, out[1]=err_byte, out[2]=err_bit, out[3]=marker (if ERR_MARKER),
 * out[4]=end_byte, out[5]=end_bit (position of the next unread bit)
 * dc_vals: int32 [max_mcus * blocks_per_mcu] in scan order; mcu_bitstart: int64 [max_mcus] */
EXPORT int64_t decode_mcus(const uint8_t *data, int64_t start_byte, int start_bit, int64_t end,
                           const comp_t *comps, int ncomp,
                           const int32_t *fast, const int32_t *maxcode, const int32_t *valptr,
                           const int32_t *mincode, const uint8_t *symbols,
                           int64_t max_mcus, int32_t *preds,
                           int64_t *mcu_bitstart, int32_t *dc_vals, int64_t *out, int check_dc) {
    tables_t t = { fast, maxcode, valptr, mincode, symbols };
    return run_decode(data, start_byte, start_bit, end, comps, ncomp, &t, max_mcus, preds, mcu_bitstart, dc_vals, out, check_dc);
}

/* Where the interval ends after the last MCU: next byte boundary, FF00-stuffed
 * padding skipped. *pad_ok = 1 when the unused bits of the last byte are all 1s. */
static int64_t aligned_end(const uint8_t *data, int64_t len, int64_t eb, int ebit, int *pad_ok) {
    int64_t p = eb;
    *pad_ok = 1;
    if (ebit) {
        uint8_t mask = (uint8_t)((1u << (8 - ebit)) - 1);
        *pad_ok = (data[eb] & mask) == mask;
        p = eb + 1;
        if (data[eb] == 0xFF && p < len && data[p] == 0x00) p++;
    }
    for (int hops = 0; hops < 4 && p + 2 < len && data[p] == 0xFF && data[p + 1] == 0x00; hops++) p += 2;
    return p;
}

/* Smoothness score over `nblocks` candidate blocks (lower = smoother); mean of
 * weighted |DC differences| to the left neighbour (inside the candidate, weight 2)
 * and to known blocks above and below (weight 1). Returns 0 when there is nothing to compare. */
static double roughness(const int32_t *dc, int64_t nblocks, const double *up_v, const double *dn_v,
                        const int32_t *left_k, const double *w) {
    double total = 0.0;
    int64_t n = 0;
    for (int64_t k = 0; k < nblocks; k++) {
        double a = (double)dc[k];
        int32_t l = left_k[k];
        if (l >= 0 && l < k) { double d = a - (double)dc[l]; total += 2.0 * (d < 0 ? -d : d) * w[k]; n++; }
        if (!isnan(up_v[k])) { double d = a - up_v[k]; total += (d < 0 ? -d : d) * w[k]; n++; }
        if (!isnan(dn_v[k])) { double d = a - dn_v[k]; total += (d < 0 ? -d : d) * w[k]; n++; }
    }
    return n ? total / (double)n : 0.0;
}

/* For each candidate i: flip data[offs[i]] ^= masks[i], decode, restore the byte.
 *
 * Decoding resumes at the last known MCU starting at or before the flipped byte
 * (mcu_bits[j], from a previous decode of the same interval starting at MCU m0):
 * everything before it is unaffected by the flip, so its DC values are copied
 * from cur_dc and the predictors come from the last blocks of MCU j-1
 * (preds0 when j == 0). MCU counts below are relative to m0.
 *   res_dec[i]      MCUs decoded cleanly
 *   res_complete[i] 1 if all `needed` MCUs decoded and the data ends exactly at marker_pos;
 *                   with marker_pos < 0 and expect_marker >= 0 the data must end on a pair
 *                   within 3 flipped bits of FF <expect_marker>; with both < 0 any clean end counts
 *   res_pad[i]      1 if the final partial byte is padded with 1-bits
 *   res_rough[i]    smoothness over all blocks (complete) or over the first
 *                   max(1, dec - min_gain) MCUs (partial)
 * scratch_dc: int32[needed * blocks_per_mcu].  Returns number of candidates evaluated. */
EXPORT int64_t eval_flips(uint8_t *data, int64_t data_len, int64_t start_byte, int start_bit, int64_t end,
                          const comp_t *comps, int ncomp,
                          const int32_t *fast, const int32_t *maxcode, const int32_t *valptr,
                          const int32_t *mincode, const uint8_t *symbols,
                          const int32_t *preds0, int64_t needed, int64_t stage1_n,
                          int64_t marker_pos, int expect_marker, int min_gain,
                          const int64_t *mcu_bits, int64_t nknown, const int32_t *cur_dc,
                          const int64_t *offs, const uint8_t *masks, int64_t ncand,
                          const double *up_v, const double *dn_v, const int32_t *left_k, const double *w,
                          int32_t *res_dec, int8_t *res_complete, int8_t *res_pad, double *res_rough,
                          int32_t *scratch_dc) {
    tables_t t = { fast, maxcode, valptr, mincode, symbols };
    int bpm = 0;
    int last_block[8];
    for (int c = 0; c < ncomp; c++) { bpm += comps[c].h * comps[c].v; last_block[c] = bpm - 1; }
    int32_t preds[8];
    int32_t jpreds[8];
    int64_t out[6];
    for (int64_t i = 0; i < ncand; i++) {
        int64_t off = offs[i];
        uint8_t m = masks[i];
        /* resume MCU: largest j with mcu_bits[j] <= off*8 */
        int64_t j = 0;
        if (nknown > 0 && start_byte <= off) {
            int64_t lo = 0, hi = nknown - 1, bitpos = off * 8;
            while (lo < hi) { int64_t mid = (lo + hi + 1) / 2; if (mcu_bits[mid] <= bitpos) lo = mid; else hi = mid - 1; }
            j = (mcu_bits[lo] <= bitpos) ? lo : 0;
        }
        int64_t sb = start_byte; int sbit = start_bit;
        if (j > 0) {
            sb = mcu_bits[j] / 8; sbit = (int)(mcu_bits[j] % 8);
            for (int c = 0; c < ncomp; c++) jpreds[c] = cur_dc[(j - 1) * bpm + last_block[c]];
            memcpy(scratch_dc, cur_dc, sizeof(int32_t) * (size_t)(j * bpm));
        } else {
            memcpy(jpreds, preds0, sizeof(int32_t) * ncomp);
        }
        int32_t *dst = scratch_dc + j * bpm;
        data[off] ^= m;
        memcpy(preds, jpreds, sizeof(int32_t) * ncomp);
        int64_t lim = (stage1_n < needed ? stage1_n : needed) - j;
        if (lim < 1) lim = 1;
        int64_t full = needed - j;
        int64_t n = run_decode(data, sb, sbit, end, comps, ncomp, &t, lim, preds, NULL, dst, out, 1);
        if (out[0] == 0 && n == lim && lim < full) {
            memcpy(preds, jpreds, sizeof(int32_t) * ncomp);
            n = run_decode(data, sb, sbit, end, comps, ncomp, &t, full, preds, NULL, dst, out, 1);
        }
        int complete = 0, pad_ok = 0;
        if (out[0] == 0 && n == full) {
            int64_t p = aligned_end(data, data_len, out[4], (int)out[5], &pad_ok);
            if (marker_pos >= 0) complete = p == marker_pos;
            else if (expect_marker >= 0)  /* end unknown: must land on a (possibly damaged) marker */
                complete = p + 1 < data_len &&
                           popcount8(data[p] ^ 0xFFu) + popcount8(data[p + 1] ^ (unsigned)expect_marker) <= 3;
            else complete = 1;
        }
        data[off] ^= m;
        n += j;
        res_dec[i] = (int32_t)n;
        res_complete[i] = (int8_t)complete;
        res_pad[i] = (int8_t)pad_ok;
        int64_t nb;
        if (complete) nb = n * bpm;
        else { int64_t k = n - min_gain; if (k < 1) k = 1; if (k > n) k = n; nb = k * bpm; }
        res_rough[i] = nb > 0 ? roughness(scratch_dc, nb, up_v, dn_v, left_k, w) : INFINITY;
    }
    return ncand;
}

/* Tail recovery: the latest data of an interval survives after a damaged stretch,
 * but where its MCUs start is unknown. Decoding from any bit position yields a
 * deterministic sequence of MCU boundaries (the DC predictor does not change where
 * an MCU ends), so the parses from all start positions form a graph: every
 * position points to the end of the MCU decoded from it. Huffman streams
 * resynchronise, so paths merge quickly - junk mostly into the true parse, but junk
 * paths also merge with each other first and can run a dozen MCUs before joining.
 *
 * Trials start every `step` bytes (two bit phases) in [lo, marker_pos), each stops
 * when it reaches a position already in the graph. The flow of a position is the
 * number of trial starts whose path passes through it. In intact data nearly every
 * start converges on the true parse within a few MCUs, so walking back from the
 * last MCU (the one ending exactly on the marker) always to the predecessor with
 * the largest flow follows the true parse; junk branches carry little flow.
 *
 * Returns the number of MCU starts written to out_bits (ascending, ending with the
 * last MCU before the marker; at most cap), 0 if no path reaches the marker, -1 if
 * memory ran out. out_flow gets the flow of each. */
#define TF_NONE  (-1)
#define TF_TERM  (-2)

typedef struct {
    int64_t *pos; int64_t *next; int32_t *starts; int32_t *flow; int64_t *best; int32_t *bestf;
    int64_t n, cap;
    int64_t *hkey; int64_t *hval; int64_t hmask;
} tgraph_t;

static inline uint64_t tf_hash(int64_t k) {
    uint64_t x = (uint64_t)k * 0x9E3779B97F4A7C15ULL;
    return x ^ (x >> 29);
}

/* index of pos; inserts a fresh node when absent (*fresh = 1). -1 when full. */
static int64_t tf_get(tgraph_t *g, int64_t pos, int *fresh) {
    uint64_t h = tf_hash(pos) & (uint64_t)g->hmask;
    for (;;) {
        if (g->hkey[h] == -1) break;
        if (g->hkey[h] == pos) { *fresh = 0; return g->hval[h]; }
        h = (h + 1) & (uint64_t)g->hmask;
    }
    if (g->n >= g->cap) return -1;
    int64_t i = g->n++;
    g->hkey[h] = pos; g->hval[h] = i;
    g->pos[i] = pos; g->next[i] = TF_NONE; g->starts[i] = 0;
    *fresh = 1;
    return i;
}

/* sort key travels with the index: no shared state, so threads may run tail_flow at once */
typedef struct { int64_t pos, idx; } tf_pair_t;
static int tf_cmp(const void *a, const void *b) {
    int64_t x = ((const tf_pair_t *)a)->pos, y = ((const tf_pair_t *)b)->pos;
    return (x > y) - (x < y);
}

EXPORT int64_t tail_flow(const uint8_t *data, int64_t data_len, int64_t lo, int64_t marker_pos,
                         const comp_t *comps, int ncomp,
                         const int32_t *fast, const int32_t *maxcode, const int32_t *valptr,
                         const int32_t *mincode, const uint8_t *symbols,
                         int64_t step, int64_t max_nodes, int64_t cap, int64_t *out_bits, int32_t *out_flow) {
    tables_t t = { fast, maxcode, valptr, mincode, symbols };
    tgraph_t g; memset(&g, 0, sizeof g);
    int64_t hsize = 1; while (hsize < 2 * max_nodes) hsize <<= 1;
    g.cap = max_nodes; g.hmask = hsize - 1;
    g.pos = malloc(sizeof(int64_t) * (size_t)max_nodes);
    g.next = malloc(sizeof(int64_t) * (size_t)max_nodes);
    g.starts = malloc(sizeof(int32_t) * (size_t)max_nodes);
    g.flow = malloc(sizeof(int32_t) * (size_t)max_nodes);
    g.best = malloc(sizeof(int64_t) * (size_t)max_nodes);
    g.bestf = malloc(sizeof(int32_t) * (size_t)max_nodes);
    g.hkey = malloc(sizeof(int64_t) * (size_t)hsize);
    g.hval = malloc(sizeof(int64_t) * (size_t)hsize);
    tf_pair_t *order = malloc(sizeof(tf_pair_t) * (size_t)max_nodes);
    int64_t result = -1;
    if (!g.pos || !g.next || !g.starts || !g.flow || !g.best || !g.bestf || !g.hkey || !g.hval || !order) goto done;
    memset(g.hkey, 0xFF, sizeof(int64_t) * (size_t)hsize);  /* all -1 */
    int32_t preds[8], dc[64];
    int full = 0;
    for (int64_t p = marker_pos - 1; p >= lo && !full; p -= step) {
        for (int ph = 0; ph < 8 && !full; ph += 4) {
            int fresh;
            int64_t cur = tf_get(&g, p * 8 + ph, &fresh);
            if (cur < 0) { full = 1; break; }
            g.starts[cur]++;
            if (!fresh) continue;
            reader_t r;
            reader_init(&r, data, p, marker_pos + 2);
            if (ph) { int32_t d; if (read_bits(&r, ph, &d)) { g.next[cur] = TF_NONE; continue; } }
            for (;;) {
                memset(preds, 0, sizeof preds);
                if (decode_mcu(&r, comps, ncomp, &t, preds, dc, 0)) break;  /* dead end */
                int64_t eb; int ebit; tell(&r, &eb, &ebit);
                int pad_ok;
                if (aligned_end(data, data_len, eb, ebit, &pad_ok) == marker_pos && pad_ok) { g.next[cur] = TF_TERM; break; }
                if (eb > marker_pos) break;
                int64_t nx = tf_get(&g, eb * 8 + ebit, &fresh);
                if (nx < 0) { full = 1; break; }
                g.next[cur] = nx;
                if (!fresh) break;  /* merged into a known path */
                cur = nx;
            }
        }
    }
    /* flow: positions in ascending order, every node hands its flow to its successor */
    for (int64_t i = 0; i < g.n; i++) {
        order[i].pos = g.pos[i]; order[i].idx = i;
        g.flow[i] = g.starts[i]; g.best[i] = -1; g.bestf[i] = 0;
    }
    qsort(order, (size_t)g.n, sizeof(tf_pair_t), tf_cmp);
    int64_t term = -1;
    for (int64_t k = 0; k < g.n; k++) {
        int64_t v = order[k].idx;
        int64_t nx = g.next[v];
        if (nx == TF_TERM) { if (term < 0 || g.flow[v] > g.flow[term]) term = v; continue; }
        if (nx < 0) continue;
        g.flow[nx] += g.flow[v];
        if (g.flow[v] > g.bestf[nx]) { g.bestf[nx] = g.flow[v]; g.best[nx] = v; }
    }
    result = 0;
    if (term < 0) goto done;
    int64_t m = 0;
    for (int64_t v = term; v >= 0 && m < cap; v = g.best[v]) m++;
    int64_t i = m;
    for (int64_t v = term; v >= 0 && i > 0; v = g.best[v]) { i--; out_bits[i] = g.pos[v]; out_flow[i] = g.flow[v]; }
    result = m;
done:
    free(g.pos); free(g.next); free(g.starts); free(g.flow); free(g.best); free(g.bestf);
    free(g.hkey); free(g.hval); free(order);
    return result;
}

/* Whole-interval start search (R10 spans): byte offsets s in [lo, hi) are tried as the
 * start of an interval of `count` MCUs ending at marker_pos (fresh predictors, DC range
 * checks on). A probe of up to `probe` MCUs must land on the reference parse known[]
 * (ascending MCU start bits of the tail, ending at the marker), which fixes the MCU
 * count; candidates with the right count are decoded in full and must end exactly on
 * the marker with valid padding. Among them the one preceded by the most marker-like
 * byte pair (prev_marker; < 0: unknown) wins, then the earliest. Returns s or -1. */
static int64_t find_known(const int64_t *known, int64_t n, int64_t pos) {
    int64_t lo = 0, hi = n - 1;
    while (lo <= hi) {
        int64_t mid = (lo + hi) / 2;
        if (known[mid] == pos) return mid;
        if (known[mid] < pos) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

EXPORT int64_t find_interval_start(const uint8_t *data, int64_t data_len, int64_t lo, int64_t hi, int64_t marker_pos,
                                   const comp_t *comps, int ncomp,
                                   const int32_t *fast, const int32_t *maxcode, const int32_t *valptr,
                                   const int32_t *mincode, const uint8_t *symbols,
                                   int64_t count, const int64_t *known, int64_t nknown, int prev_marker, int probe) {
    tables_t t = { fast, maxcode, valptr, mincode, symbols };
    int32_t preds[8], dc[64];
    int64_t best = -1; int best_ham = 1 << 20;
    for (int64_t s = lo; s < hi; s++) {
        reader_t r;
        reader_init(&r, data, s, marker_pos + 2);
        memset(preds, 0, sizeof preds);
        int64_t total = -1;
        for (int k = 1; k <= probe; k++) {
            if (decode_mcu(&r, comps, ncomp, &t, preds, dc, 1)) break;
            int64_t eb; int ebit; tell(&r, &eb, &ebit);
            int64_t j = find_known(known, nknown, eb * 8 + ebit);
            if (j >= 0) { total = k + (nknown - j); break; }
        }
        if (total != count) continue;
        reader_init(&r, data, s, marker_pos + 2);
        memset(preds, 0, sizeof preds);
        int64_t n = 0;
        while (n < count && !decode_mcu(&r, comps, ncomp, &t, preds, dc, 1)) n++;
        if (n != count) continue;
        int64_t eb; int ebit; tell(&r, &eb, &ebit);
        int pad_ok;
        if (aligned_end(data, data_len, eb, ebit, &pad_ok) != marker_pos || !pad_ok) continue;
        int ham = 16;
        if (s >= 2 && prev_marker >= 0) ham = popcount8(data[s - 2] ^ 0xFF) + popcount8(data[s - 1] ^ (unsigned)prev_marker);
        if (ham < best_ham) { best_ham = ham; best = s; }
    }
    return best;
}

/* ------------------------------------------------------------------ bit writer
 * State st[3] = {bytes written, bit accumulator, bits in accumulator}; bytes are
 * FF00-stuffed as they are written. Functions return 0, or -1 when `cap` is too small. */
static inline int bw_emit(uint8_t *out, int64_t cap, int64_t *st, uint32_t val, int n) {
    if (n <= 0) return 0;
    uint64_t acc = (uint64_t)st[1];
    int nb = (int)st[2];
    acc = (acc << n) | ((uint64_t)val & ((1ULL << n) - 1));
    nb += n;
    while (nb >= 8) {
        nb -= 8;
        uint8_t b = (uint8_t)(acc >> nb);
        if (st[0] + 2 > cap) return -1;
        out[st[0]++] = b;
        if (b == 0xFF) out[st[0]++] = 0x00;
    }
    acc &= nb ? ((1ULL << nb) - 1) : 0;
    st[1] = (int64_t)acc; st[2] = nb;
    return 0;
}

EXPORT int bw_put(uint8_t *out, int64_t cap, int64_t *st, uint32_t val, int n) {
    return bw_emit(out, cap, st, val, n);
}

/* Pad the last byte with 1-bits. */
EXPORT int bw_finish(uint8_t *out, int64_t cap, int64_t *st) {
    int nb = (int)st[2];
    if (nb) return bw_emit(out, cap, st, (1u << (8 - nb)) - 1, 8 - nb);
    return 0;
}

/* Append entropy-coded bits [start_bit, end_bit) of data (positions as the decoder
 * reports them: offsets in the stuffed stream), unstuffed, to the writer. */
EXPORT int bw_copy(uint8_t *out, int64_t cap, int64_t *st, const uint8_t *data, int64_t data_len,
                   int64_t start_bit, int64_t end_bit) {
    int64_t p = start_bit / 8;
    int first = 1;
    while (p * 8 < end_bit && p < data_len) {
        uint8_t b = data[p];
        int lo_bit = first ? (int)(start_bit % 8) : 0;
        int hi_bit = (p + 1) * 8 <= end_bit ? 8 : (int)(end_bit - p * 8);
        int n = hi_bit - lo_bit;
        if (n > 0 && bw_emit(out, cap, st, (uint32_t)(b >> (8 - hi_bit)) & ((1u << n) - 1), n)) return -1;
        first = 0;
        if (b == 0xFF && p + 1 < data_len && data[p + 1] == 0x00) p += 2; else p += 1;
    }
    return 0;
}

/* DC-only MCUs (flat blocks: DC difference, then EOB) with the file's own tables.
 * targets: int32[mcu_count * bpm] DC per block in MCU order; preds: int32[ncomp] in/out.
 * dc_len/dc_code: int32[ncomp][16] Huffman code per DC category (len 0 = absent);
 * eob_len/eob_code: int32[ncomp]. Steps a table cannot express are clamped.
 * Returns 0, -1 (buffer too small) or -2 (DC table without category 0). */
EXPORT int encode_dc_fill(uint8_t *out, int64_t cap, int64_t *st, const comp_t *comps, int ncomp,
                          const int32_t *dc_len, const int32_t *dc_code, const int32_t *eob_len, const int32_t *eob_code,
                          int64_t mcu_count, const int32_t *targets, int32_t *preds) {
    int64_t k = 0;
    for (int64_t m = 0; m < mcu_count; m++) {
        for (int c = 0; c < ncomp; c++) {
            int nb = comps[c].h * comps[c].v;
            const int32_t *ln = dc_len + c * 16, *cd = dc_code + c * 16;
            for (int b = 0; b < nb; b++) {
                int32_t diff = targets[k++] - preds[c];
                int32_t a = diff < 0 ? -diff : diff;
                int want = 0;
                while (a >> want) want++;
                int s = want;
                while (s > 0 && ln[s] == 0) s--;
                if (s == 0 && ln[0] == 0) return -2;
                if (s < want) {
                    int32_t lim = (1 << s) - 1;
                    if (diff > lim) diff = lim;
                    if (diff < -lim) diff = -lim;
                }
                if (bw_emit(out, cap, st, (uint32_t)cd[s], ln[s])) return -1;
                if (s) {
                    uint32_t mag = (uint32_t)(diff >= 0 ? diff : diff + (1 << s) - 1);
                    if (bw_emit(out, cap, st, mag, s)) return -1;
                }
                preds[c] += diff;
                if (bw_emit(out, cap, st, (uint32_t)eob_code[c], eob_len[c])) return -1;
            }
        }
    }
    return 0;
}
