#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define P 8            /* neighbors */
#define RADIUS 1       /* radius */
#define BINS 256       /* 0..255 */
#define GRID_X 8
#define GRID_Y 8

/* Simple PGM (P5) loader */
unsigned char *read_pgm(const char *path, int *width, int *height) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return NULL; }

    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1) { fclose(f); return NULL; }
    if (strcmp(magic, "P5") != 0) { fprintf(stderr, "Not P5 PGM: %s\n", path); fclose(f); return NULL; }

    /* skip comments */
    int c = fgetc(f);
    while (c == '\n' || c == '\r' || c == ' ' || c == '#') {
        if (c == '#') {
            while ((c = fgetc(f)) != '\n' && c != EOF) {}
        } else {
            c = fgetc(f);
        }
    }
    ungetc(c, f);

    int w, h, maxv;
    if (fscanf(f, "%d %d %d", &w, &h, &maxv) != 3) { fclose(f); return NULL; }
    if (maxv > 255) { fprintf(stderr, "Only 8-bit PGM supported\n"); fclose(f); return NULL; }

    /* skip single whitespace char after header */
    fgetc(f);

    unsigned char *data = (unsigned char*)malloc(w * h);
    if (!data) { fclose(f); return NULL; }
    if (fread(data, 1, w * h, f) != (size_t)(w * h)) {
        fprintf(stderr, "Unexpected EOF reading image\n");
        free(data); fclose(f); return NULL;
    }
    fclose(f);
    *width = w; *height = h;
    return data;
}

/* Compute LBP image (same size as input). Border pixels set to 0. */
uint8_t *compute_lbp(const unsigned char *img, int w, int h) {
    uint8_t *lbp = (uint8_t*)calloc(w * h, 1);
    if (!lbp) return NULL;

    /* neighbor offsets for P=8, R=1 (clockwise starting at (x+1,y)) */
    const int dx[P] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    const int dy[P] = { 0, -1, -1, -1, 0, 1, 1, 1 };

    for (int y = RADIUS; y < h - RADIUS; y++) {
        for (int x = RADIUS; x < w - RADIUS; x++) {
            unsigned char center = img[y * w + x];
            unsigned char code = 0;
            for (int n = 0; n < P; n++) {
                int xn = x + dx[n];
                int yn = y + dy[n];
                unsigned char neighbor = img[yn * w + xn];
                code <<= 1;
                if (neighbor >= center) code |= 1;
            }
            lbp[y * w + x] = code;
        }
    }
    return lbp;
}

/* Compute concatenated histogram feature vector (HFV).
 * hf_length = GRID_X * GRID_Y * BINS
 * Caller must free returned array.
 */
float *compute_hfv(const uint8_t *lbp, int w, int h, int grid_x, int grid_y) {
    int cell_w = w / grid_x;
    int cell_h = h / grid_y;

    int hf_len = grid_x * grid_y * BINS;
    float *hf = (float*)calloc(hf_len, sizeof(float));
    if (!hf) return NULL;

    for (int gy = 0; gy < grid_y; gy++) {
        for (int gx = 0; gx < grid_x; gx++) {
            int x0 = gx * cell_w;
            int y0 = gy * cell_h;
            int x1 = (gx == grid_x - 1) ? (w) : (x0 + cell_w);
            int y1 = (gy == grid_y - 1) ? (h) : (y0 + cell_h);

            int base = (gy * grid_x + gx) * BINS;
            for (int yy = y0; yy < y1; yy++) {
                for (int xx = x0; xx < x1; xx++) {
                    uint8_t v = lbp[yy * w + xx];
                    hf[base + v] += 1.0f;
                }
            }

            /* optional: normalize histogram to sum=1 (helps with illumination) */
            float sum = 0.0f;
            for (int b = 0; b < BINS; b++) sum += hf[base + b];
            if (sum > 0.0f) {
                for (int b = 0; b < BINS; b++) hf[base + b] /= sum;
            }
        }
    }
    return hf;
}

/* Chi-square distance between two HF vectors */
double chi_square(const float *a, const float *b, int len) {
    double s = 0.0;
    for (int i = 0; i < len; i++) {
        double num = (double)a[i] - (double)b[i];
        double den = (double)a[i] + (double)b[i] + 1e-10;
        s += 0.5 * (num * num) / den;
    }
    return s;
}

/* A simple structure to hold training sample */
typedef struct {
    int label;
    float *hfv; /* length hf_len */
} TrainSample;

/* Read training list file: each line "label path" */
TrainSample *load_training(const char *list_path, int *out_count, int *w, int *h, int grid_x, int grid_y) {
    FILE *f = fopen(list_path, "r");
    if (!f) { perror("fopen train list"); return NULL; }

    /* First pass: count lines */
    int count = 0;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '#') continue;
        count++;
    }
    if (count == 0) { fclose(f); return NULL; }
    rewind(f);

    TrainSample *samples = (TrainSample*)calloc(count, sizeof(TrainSample));
    int idx = 0;

    int img_w = 0, img_h = 0;

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '#') continue;
        /* parse label and path */
        int label;
        char path[512];
        if (sscanf(line, "%d %s", &label, path) != 2) {
            fprintf(stderr, "Bad line in train list: %s\n", line);
            continue;
        }

        int tw, th;
        unsigned char *img = read_pgm(path, &tw, &th);
        if (!img) {
            fprintf(stderr, "Failed to load %s\n", path);
            continue;
        }
        if (img_w == 0) { img_w = tw; img_h = th; }
        if (tw != img_w || th != img_h) {
            fprintf(stderr, "Image size mismatch: %s (expected %dx%d got %dx%d)\n",
                    path, img_w, img_h, tw, th);
            free(img); continue;
        }

        uint8_t *lbp = compute_lbp(img, tw, th);
        free(img);
        if (!lbp) { fprintf(stderr, "LBP failure\n"); continue; }

        float *hfv = compute_hfv(lbp, tw, th, grid_x, grid_y);
        free(lbp);
        if (!hfv) { fprintf(stderr, "HFV failure\n"); continue; }

        samples[idx].label = label;
        samples[idx].hfv = hfv;
        idx++;
    }
    fclose(f);

    *out_count = idx;
    *w = img_w; *h = img_h;
    return samples;
}

void free_training(TrainSample *samples, int count, int hf_len) {
    for (int i = 0; i < count; i++) {
        if (samples[i].hfv) free(samples[i].hfv);
    }
    free(samples);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s train_list.txt test_image.pgm\n", argv[0]);
        return 1;
    }

    const char *train_list = argv[1];
    const char *test_path = argv[2];

    int grid_x = GRID_X, grid_y = GRID_Y;

    int train_count = 0;
    int img_w = 0, img_h = 0;
    TrainSample *samples = load_training(train_list, &train_count, &img_w, &img_h, grid_x, grid_y);
    if (!samples || train_count == 0) {
        fprintf(stderr, "No training samples loaded\n");
        return 1;
    }
    int hf_len = grid_x * grid_y * BINS;

    /* Load test image */
    int tw, th;
    unsigned char *test_img = read_pgm(test_path, &tw, &th);
    if (!test_img) { fprintf(stderr, "Failed to load test image\n"); free_training(samples, train_count, hf_len); return 1; }
    if (tw != img_w || th != img_h) {
        fprintf(stderr, "Test image size doesn't match training images: %dx%d vs %dx%d\n", tw, th, img_w, img_h);
        free(test_img); free_training(samples, train_count, hf_len); return 1;
    }

    uint8_t *test_lbp = compute_lbp(test_img, tw, th);
    free(test_img);
    if (!test_lbp) { fprintf(stderr, "LBP failure on test\n"); free_training(samples, train_count, hf_len); return 1; }

    float *test_hfv = compute_hfv(test_lbp, tw, th, grid_x, grid_y);
    free(test_lbp);
    if (!test_hfv) { fprintf(stderr, "HFV failure on test\n"); free_training(samples, train_count, hf_len); return 1; }

    /* Find best match */
    double best_d = 1e300;
    int best_label = -1;
    for (int i = 0; i < train_count; i++) {
        double d = chi_square(test_hfv, samples[i].hfv, hf_len);
        // printf("Compare to sample %d label %d dist %.6f\n", i, samples[i].label, d);
        if (d < best_d) { best_d = d; best_label = samples[i].label; }
    }

    printf("Best match: label=%d distance=%.6f\n", best_label, best_d);

    free(test_hfv);
    free_training(samples, train_count, hf_len);
    return 0;
}
