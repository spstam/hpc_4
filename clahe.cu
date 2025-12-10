#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "clahe.h"

// ==========================================
// 1. I/O FUNCTIONS
// ==========================================
PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    PGM_IMG result;
    int v_max;

    in_file = fopen(path, "rb");
    if (in_file == NULL){
        printf("Error: Input file not found!\n");
        exit(1);
    }
    fscanf(in_file, "%s", sbuf); 
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d",&v_max);
    fgetc(in_file); 

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    fread(result.img, sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n", img.w, img.h);
    fwrite(img.img, sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img) {
    if(img.img) free(img.img);
}

// ==========================================
// 2. CPU IMPLEMENTATION (Reference Code)
// ==========================================
void compute_histogram_cpu(unsigned char* data, int w, int h, int start_x, int start_y, int tile_w, int tile_h, int* lut) {
    int hist[256] = {0};
    int excess = 0, cdf = 0, total_pixels = tile_w * tile_h; 

    for (int y = start_y; y < start_y + tile_h; ++y) {
        for (int x = start_x; x < start_x + tile_w; ++x) {
            if(x < w && y < h) hist[data[y * w + x]]++;
        }
    }

    for (int i = 0; i < 256; ++i) {
        if (hist[i] > CLIP_LIMIT) {
            excess += (hist[i] - CLIP_LIMIT);
            hist[i] = CLIP_LIMIT;
        }
    }

    int avg_inc = excess / 256;
    for (int i = 0; i < 256; ++i) hist[i] += avg_inc;
    
    for (int i = 0; i < 256; ++i) {
        cdf += hist[i];
        int val = (int)((float)cdf * 255.0f / total_pixels + 0.5f);
        if (val > 255) val = 255;
        lut[i] = val;
    }
}
__global__ void histogram_lut_kernel(unsigned char* img, int* all_luts, int w, int h, int grid_w, int clip_limit) {
    __shared__ int s_hist[256];
    __shared__ int s_excess[256];
    int tx = blockIdx.x, ty = blockIdx.y;
    int x_start = tx * TILE_SIZE;
    int y_start = ty * TILE_SIZE;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int actual_w = (x_start + TILE_SIZE > w) ? (w - x_start) : TILE_SIZE;
    int actual_h = (y_start + TILE_SIZE > h) ? (h - y_start) : TILE_SIZE;
    int total_pixels = actual_w * actual_h;

    for (int i = tid; i < 256; i += blockDim.x * blockDim.y) {
        s_hist[i] = 0;
        s_excess[i] = 0;
    }
    __syncthreads();

    for (int y = threadIdx.y; y < actual_h; y += blockDim.y) {
        for (int x = threadIdx.x; x < actual_w; x += blockDim.x) {
            int global_y = y_start + y;
            int global_x = x_start + x;
            if (global_x < w && global_y < h) {
                unsigned char val = img[global_y * w + global_x];
                atomicAdd(&s_hist[val], 1);
            }
        }
    }

    __syncthreads();
    for (int i = tid; i < 256; i += blockDim.x * blockDim.y) {
        if (s_hist[i] > clip_limit) {
            s_excess[i] = (s_hist[i] - clip_limit);
            s_hist[i] = clip_limit;
        }
    }
    __syncthreads();

    if (tid < 128) s_excess[tid] += s_excess[tid + 128];
    __syncthreads();
    if (tid < 64) s_excess[tid] += s_excess[tid + 64];
    __syncthreads();
    if (tid < 32) s_excess[tid] += s_excess[tid + 32];
    __syncthreads();
    if (tid < 16) s_excess[tid] += s_excess[tid + 16];
    __syncthreads();
    if (tid < 8) s_excess[tid] += s_excess[tid + 8];
    __syncthreads();
    if (tid < 4) s_excess[tid] += s_excess[tid + 4];
    __syncthreads();
    if (tid < 2) s_excess[tid] += s_excess[tid + 2];
    __syncthreads();
    if (tid < 1) s_excess[tid] += s_excess[tid + 1];
    __syncthreads();

    int avg_inc = s_excess[0] / 256;
    s_hist[tid] += avg_inc;

    if(tid==0){
        int cdf = 0;
        for (int i = 0; i < 256; ++i) {
            cdf += s_hist[i];
            int val = (int)((float)cdf * 255.0f / total_pixels + 0.5f);
            if (val > 255) val = 255;
            all_luts[(ty * grid_w + tx) * 256 + i] = val;
        }
    }
}

PGM_IMG apply_clahe_cpu(PGM_IMG img_in) {
    PGM_IMG img_out;
    int w = img_in.w;
    int h = img_in.h;
    img_out.w = w; img_out.h = h;
    img_out.img = (unsigned char *)malloc(w * h * sizeof(unsigned char));

    int grid_w = (w + TILE_SIZE - 1) / TILE_SIZE;
    int grid_h = (h + TILE_SIZE - 1) / TILE_SIZE;
    int *all_luts = (int *)malloc(grid_w * grid_h * 256 * sizeof(int));

    // Precompute LUTs
    for (int ty = 0; ty < grid_h; ++ty) {
        for (int tx = 0; tx < grid_w; ++tx) {
            int x_start = tx * TILE_SIZE;
            int y_start = ty * TILE_SIZE;
            int actual_w = (x_start + TILE_SIZE > w) ? (w - x_start) : TILE_SIZE;
            int actual_h = (y_start + TILE_SIZE > h) ? (h - y_start) : TILE_SIZE;
            compute_histogram_cpu(img_in.img, w, h, x_start, y_start, actual_w, actual_h, &all_luts[(ty * grid_w + tx) * 256]);
        }
    }


    // Bilinear Interpolation
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float ty_f = (float)y / TILE_SIZE - 0.5f;
            float tx_f = (float)x / TILE_SIZE - 0.5f;
            int y1 = (int)floor(ty_f), x1 = (int)floor(tx_f);
            int y2 = y1 + 1, x2 = x1 + 1;
            float y_weight = ty_f - y1, x_weight = tx_f - x1;

            if (x1 < 0) x1 = 0; if (x2 >= grid_w) x2 = grid_w - 1;
            if (y1 < 0) y1 = 0; if (y2 >= grid_h) y2 = grid_h - 1;

            int val = img_in.img[y * w + x];
            int tl = all_luts[(y1 * grid_w + x1) * 256 + val];
            int tr = all_luts[(y1 * grid_w + x2) * 256 + val];
            int bl = all_luts[(y2 * grid_w + x1) * 256 + val];
            int br = all_luts[(y2 * grid_w + x2) * 256 + val];

            float top = tl * (1.0f - x_weight) + tr * x_weight;
            float bot = bl * (1.0f - x_weight) + br * x_weight;
            img_out.img[y * w + x] = (unsigned char)(top * (1.0f - y_weight) + bot * y_weight + 0.5f);
        }
    }
    free(all_luts);
    return img_out;
}


__global__ void render_clahe_kernel(const unsigned char* __restrict__ img_in, 
    unsigned char* __restrict__ img_out, 
    const int* __restrict__ all_luts, 
    int w, int h, int grid_w, int grid_h) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float ty_f = (float)y / TILE_SIZE - 0.5f;
    float tx_f = (float)x / TILE_SIZE - 0.5f;
    int y1 = (int)floor(ty_f), x1 = (int)floor(tx_f);
    int y2 = y1 + 1;
    int x2 = x1 + 1;

    float xw = tx_f - x1, yw = ty_f - y1;

    x1= max(x1,0);
    x2= min(x2,grid_w-1);
    y1= max(y1,0);
    y2= min(y2,grid_h-1); 

    int wy1 = y1 * grid_w;
    int wy2 = y2 * grid_w;

    int val = img_in[y * w + x];
    int tl = all_luts[(wy1 + x1) * 256 + val];
    int tr = all_luts[(wy1 + x2) * 256 + val];
    int bl = all_luts[(wy2 + x1) * 256 + val];
    int br = all_luts[(wy2 + x2) * 256 + val];

    float w_tl = (1.0f - xw) * (1.0f - yw);
    float w_tr = xw * (1.0f - yw);
    float w_bl = (1.0f - xw) * yw;
    float w_br = xw * yw;

    float result = w_tl * tl;
    result += w_tr * tr;
    result += w_bl * bl;
    result += w_br * br;

    img_out[y * w + x] = (unsigned char)(result + 0.5f);
}


