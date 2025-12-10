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

// ==========================================
// 3. GPU KERNELS
// ==========================================
// __global__ void histogram_lut_kernel(unsigned char* img, int* all_luts, int w, int h, int grid_w, int clip_limit) {
//     __shared__ int s_hist[256];
//     __shared__ int s_temp[256];

//     __shared__ int s_excess[256];
//     int tx = blockIdx.x, ty = blockIdx.y;
//     int x_start = tx * TILE_SIZE, tid = threadIdx.y * blockDim.x + threadIdx.x;
//     int y_start = ty * TILE_SIZE;
//     if (tid < 256) { s_hist[tid] = 0; s_excess[tid] = 0; }
//     __syncthreads();

//     int px = x_start + threadIdx.x, py = ty * TILE_SIZE + threadIdx.y;
//     if (px < w && py < h) atomicAdd(&s_hist[img[py * w + px]], 1);
//     __syncthreads();

//     if (tid < 256) {
//         if (s_hist[tid] > clip_limit) { s_excess[tid] = s_hist[tid] - clip_limit; s_hist[tid] = clip_limit; }
//     }
//     __syncthreads();

//     // for (unsigned int stride = 128; stride > 0; stride >>= 1) {
//         if (tid < 128) s_excess[tid] += s_excess[tid + 128];
//         __syncthreads();
//         if (tid < 64) s_excess[tid] += s_excess[tid + 64];
//         __syncthreads();
//         if (tid < 32) s_excess[tid] += s_excess[tid + 32];
//         __syncthreads();
//         if (tid < 16) s_excess[tid] += s_excess[tid + 16];
//         __syncthreads();
//         if (tid < 8) s_excess[tid] += s_excess[tid + 8];
//         __syncthreads();
//         if (tid < 4) s_excess[tid] += s_excess[tid + 4];
//         __syncthreads();
//         if (tid < 2) s_excess[tid] += s_excess[tid + 2];
//         __syncthreads();
//         if (tid < 1) s_excess[tid] += s_excess[tid + 1];
//         __syncthreads();
//     // }
//     // OLD single thread calculation CDF
//     // if (tid == 0) {
//         // int avg_inc = s_excess[0] / 256;
//         // int cdf = 0;
//         // int tile_pixels = ((x_start + TILE_SIZE > w) ? (w - x_start) : TILE_SIZE) * ((ty * TILE_SIZE + TILE_SIZE > h) ? (h - ty * TILE_SIZE) : TILE_SIZE);
//         // int* my_lut = &all_luts[(ty * grid_w + tx) * 256];
//         // for (int i = 0; i < 256; ++i) {
//             // s_hist[i] += avg_inc; cdf += s_hist[i];
//             // int val = (int)((float)cdf * 255.0f / tile_pixels + 0.5f);
//             // my_lut[i] = (val > 255) ? 255 : val;
//         // }
//     // }
//       __shared__ int avg_inc;
//     if (tid == 0) avg_inc = s_excess[0] / 256;
//     __syncthreads();

//     if (tid < 256) {
//         s_hist[tid] += avg_inc;
//         s_temp[tid] = s_hist[tid];
//     }
//     __syncthreads();


    
//     for (unsigned int stride = 1; stride <= 128; stride *= 2) {
//         int index = (tid + 1) * stride * 2 - 1;
//         if (index < 256) {
//         s_temp[index] += s_temp[index - stride];
//         }
//         __syncthreads();
//     }

//     __shared__ int total_sum;
//     if (tid == 0) {
//         total_sum = s_temp[255]; 
//         s_temp[255] = 0; 
//     }
//     __syncthreads();


//     for (unsigned int stride = 128; stride > 0; stride >>= 1) {
//         int index = (tid + 1) * stride * 2 - 1; // Δείκτης "Ρίζας" υποδέντρου (Δεξί παιδί)
//         if (index < 256) {
//         // Εδώ γίνεται η ανταλλαγή πληροφορίας
//         int t = s_temp[index - stride];      // Κρατάμε την τιμή του Αριστερού παιδιού
//         s_temp[index - stride] = s_temp[index]; // Το Αριστερό παιδί παίρνει την τιμή της Ρίζας
//         s_temp[index] += t;                  // Η Ρίζα (Δεξί) παίρνει την (Παλιά Ρίζα + Παλιό Αριστερό)
//         }
//         __syncthreads();
//     }

//     // 6. Final Write to Global Memory
//     if (tid < 256) {
//         int cdf_val;
//         if (tid == 255) {
//             cdf_val = total_sum;
//         } else {
//             cdf_val = s_temp[tid + 1];
//         }

//         int aw = (x_start + TILE_SIZE > w) ? (w - x_start) : TILE_SIZE;
//         int ah = (y_start + TILE_SIZE > h) ? (h - y_start) : TILE_SIZE;
//         int pixels = aw * ah;
        
//         int val = (int)((float)cdf_val * 255.0f / pixels + 0.5f);
//         if (val > 255) val = 255;
        
//         all_luts[(ty * grid_w + tx) * 256 + tid] = val;
//     }


// }
__global__ void histogram_lut_kernel(unsigned char* img, int* all_luts, int w, int h, int grid_w, int clip_limit) {
    // Shared memory for histogram and scan
    __shared__ int s_hist[256];
    
    // We need a temp buffer for the scan to prevent race conditions
    __shared__ int s_temp[256];
    
    // Shared accumulator for excess pixels
    __shared__ int s_excess_sum;

    int tx = blockIdx.x;
    int ty = blockIdx.y;
    
    // Linear thread ID (0 to 255)
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // --------------------------------------------------------
    // 1. INITIALIZATION
    // --------------------------------------------------------
    // Perfect mapping: 1 thread initializes 1 bin
    s_hist[tid] = 0;
    if (tid == 0) s_excess_sum = 0;
    __syncthreads();

    // --------------------------------------------------------
    // 2. BUILD HISTOGRAM (Strided Loop)
    // --------------------------------------------------------
    int x_start = tx * TILE_SIZE;
    int y_start = ty * TILE_SIZE;

    // Even though we only have 256 threads, we need to read 1024 pixels (32x32).
    // Each thread will read 4 pixels on average.
    for (int y = threadIdx.y; y < TILE_SIZE; y += blockDim.y) {
        for (int x = threadIdx.x; x < TILE_SIZE; x += blockDim.x) {
            
            int px = x_start + x;
            int py = y_start + y;

            if (px < w && py < h) {
                // Use __ldg to force read via read-only cache
                unsigned char val = __ldg(&img[py * w + px]);
                atomicAdd(&s_hist[val], 1);
            }
        }
    }
    __syncthreads();

    // --------------------------------------------------------
    // 3. CLIP & CALCULATE EXCESS
    // --------------------------------------------------------
    // 1 thread manages 1 bin. No complex reduction needed.
    int val = s_hist[tid];
    if (val > clip_limit) {
        atomicAdd(&s_excess_sum, val - clip_limit);
        val = clip_limit;
        s_hist[tid] = val;
    }
    __syncthreads();

    // --------------------------------------------------------
    // 4. REDISTRIBUTE EXCESS
    // --------------------------------------------------------
    int avg_inc = s_excess_sum / 256;
    val += avg_inc;
    s_hist[tid] = val;
    __syncthreads();

    // --------------------------------------------------------
    // 5. PARALLEL SCAN (Hillis-Steele / Prefix Sum)
    // --------------------------------------------------------
    // This calculates the CDF in log2(256) = 8 steps.
    // We swap reading from s_hist/s_temp to avoid race conditions.
    
    // Load current value into temp for the first pass
    s_temp[tid] = val;
    __syncthreads();

    // Step 1: Stride 1
    if (tid >= 1) val += s_temp[tid - 1];
    __syncthreads(); s_hist[tid] = val; __syncthreads();

    // Step 2: Stride 2
    if (tid >= 2) val += s_hist[tid - 2];
    __syncthreads(); s_temp[tid] = val; __syncthreads();

    // Step 3: Stride 4
    if (tid >= 4) val += s_temp[tid - 4];
    __syncthreads(); s_hist[tid] = val; __syncthreads();

    // Step 4: Stride 8
    if (tid >= 8) val += s_hist[tid - 8];
    __syncthreads(); s_temp[tid] = val; __syncthreads();

    // Step 5: Stride 16
    if (tid >= 16) val += s_temp[tid - 16];
    __syncthreads(); s_hist[tid] = val; __syncthreads();

    // Step 6: Stride 32
    if (tid >= 32) val += s_hist[tid - 32];
    __syncthreads(); s_temp[tid] = val; __syncthreads();

    // Step 7: Stride 64
    if (tid >= 64) val += s_temp[tid - 64];
    __syncthreads(); s_hist[tid] = val; __syncthreads();

    // Step 8: Stride 128 (Final)
    if (tid >= 128) val += s_hist[tid - 128];
    // Final CDF value is now in 'val'

    // --------------------------------------------------------
    // 6. NORMALIZE & WRITE TO GLOBAL MEMORY
    // --------------------------------------------------------
    // Calculate scale factor only once (saves division operations)
    __shared__ float s_scale;
    if (tid == 0) {
        int actual_w = (x_start + TILE_SIZE > w) ? (w - x_start) : TILE_SIZE;
        int actual_h = (y_start + TILE_SIZE > h) ? (h - y_start) : TILE_SIZE;
        s_scale = 255.0f / (float)(actual_w * actual_h);
    }
    __syncthreads();

    int result = (int)(val * s_scale + 0.5f);
    if (result > 255) result = 255;
    
    // Coalesced write
    all_luts[(ty * grid_w + tx) * 256 + tid] = result;
}
// __global__ void render_clahe_kernel(
//     const unsigned char* __restrict__ img_in, 
//     unsigned char* __restrict__ img_out, 
//     const int* __restrict__ all_luts, 
//     int w, int h, 
//     int grid_w, int grid_h) {
    
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x >= w || y >= h) return;

//     float tx_f = (float)x / TILE_SIZE - 0.5f, ty_f = (float)y / TILE_SIZE - 0.5f;
//     int x1 = floorf(tx_f), y1 = floorf(ty_f);
//     int x2 = x1 + 1, y2 = y1 + 1;
//     float xw = tx_f - x1, yw = ty_f - y1;

//     //removing if removes thread divergence
//     //min/max converts to single hardware instructions so it does not branch
//     x1 = max(0, x1);
//     x2 = min(grid_w - 1, x2);

//     y1 = max(0, y1);
//     y2 = min(grid_h -1 , y2);

//     //if (x1 < 0) x1 = 0; if (x2 >= grid_w) x2 = grid_w - 1;
//     //if (y1 < 0) y1 = 0; if (y2 >= grid_h) y2 = grid_h - 1;

//     int val = img_in[y * w + x];
//     int tl = all_luts[(y1 * grid_w + x1) * 256 + val];
//     int tr = all_luts[(y1 * grid_w + x2) * 256 + val];
//     int bl = all_luts[(y2 * grid_w + x1) * 256 + val];
//     int br = all_luts[(y2 * grid_w + x2) * 256 + val];

//     float top = tl * (1.0f - xw) + tr * xw;
//     float bot = bl * (1.0f - xw) + br * xw;
//     img_out[y * w + x] = (unsigned char)(top * (1.0f - yw) + bot * yw + 0.5f);
// }

//MARK: IMPORTANT

// __global__ void render_clahe_kernel(
//     const unsigned char* __restrict__ img_in, 
//     unsigned char* __restrict__ img_out, 
//     cudaTextureObject_t tex_luts, 
//     int w, int h, 
//     int grid_w, int grid_h) 
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x >= w || y >= h) return;

//     // Standard Coordinate Math
//     //MARK: Made div to mul
//     float tx_f = (float)x * INVERSE_TILE_SIZE - 0.5f;
//     float ty_f = (float)y * INVERSE_TILE_SIZE - 0.5f;
//     // int x1 = floorf(tx_f), y1 = floorf(ty_f);
//     int x1 = (int)tx_f;
//     int y1 = (int)ty_f;

//     int x2 = x1 + 1, y2 = y1 + 1;
//     float xw = tx_f - x1, yw = ty_f - y1;

//     x1 = max(0, x1);
//     x2 = min(grid_w - 1, x2);
//     y1 = max(0, y1);
//     y2 = min(grid_h - 1, y2);

//     int val = img_in[y * w + x];

//     // LUT Access
//     // Optimization: Pre-calculate the row offsets to avoid multiplication repetition
//     int row1 = y1 * grid_w;
//     int row2 = y2 * grid_w;

//     // We map the 2D (grid_index, val) to 1D linear index manually
//     // index = (grid_index * 256) + val
//     int tl = tex1Dfetch<int>(tex_luts, (row1 + x1) * 256 + val);
//     int tr = tex1Dfetch<int>(tex_luts, (row1 + x2) * 256 + val);
//     int bl = tex1Dfetch<int>(tex_luts, (row2 + x1) * 256 + val);
//     int br = tex1Dfetch<int>(tex_luts, (row2 + x2) * 256 + val);

//     // float top = tl * (1.0f - xw) + tr * xw;
//     // float bot = bl * (1.0f - xw) + br * xw;
//     // img_out[y * w + x] = (unsigned char)(top * (1.0f - yw) + bot * yw + 0.5f);
    
//     //utilise fused multiply and add 
//     float w_tl = (1.0f - xw) * (1.0f - yw);
//     float w_tr = xw * (1.0f - yw);
//     float w_bl = (1.0f - xw) * yw;
//     float w_br = xw * yw;

//     float result = w_tl * tl;
//     result += w_tr * tr;
//     result += w_bl * bl;
//     result += w_br * br;

//     img_out[y * w + x] = (unsigned char)(result + 0.5f);
// }


__device__ inline unsigned char calculate_clahe_pixel(
    int x, int y, 
    unsigned char val, 
    int w, int grid_w, int grid_h, 
    cudaTextureObject_t tex_luts) 
{
    // Standard coordinate calculation
    float tx_f = x * (1.0f / TILE_SIZE) - 0.5f;
    float ty_f = y * (1.0f / TILE_SIZE) - 0.5f;
    
    int x1 = (int)tx_f;
    int y1 = (int)ty_f;
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float xw = tx_f - x1;
    float yw = ty_f - y1;

    // Boundary checks (fast hardware instructions)
    x1 = max(0, x1); x2 = min(grid_w - 1, x2);
    y1 = max(0, y1); y2 = min(grid_h - 1, y2);

    // Pre-calculate row offsets
    int row1 = y1 * grid_w;
    int row2 = y2 * grid_w;

    // Fetch LUT values
    int tl = tex1Dfetch<int>(tex_luts, (row1 + x1) * 256 + val);
    int tr = tex1Dfetch<int>(tex_luts, (row1 + x2) * 256 + val);
    int bl = tex1Dfetch<int>(tex_luts, (row2 + x1) * 256 + val);
    int br = tex1Dfetch<int>(tex_luts, (row2 + x2) * 256 + val);

    // Interpolate
    float top_val = tl + xw * (tr - tl);

    float bot_val = bl + xw * (br - bl);

    float result = top_val + yw * (bot_val - top_val);

    return (unsigned char)(result + 0.5f);
}

__global__ void render_clahe_kernel(
    const unsigned char* __restrict__ img_in, 
    unsigned char* __restrict__ img_out, 
    cudaTextureObject_t tex_luts, 
    int w, int h, 
    int grid_w, int grid_h) 
{
    // Thread ID now corresponds to an integer (chunk of 4 pixels), not a single pixel
    int vec_x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds (width is divided by 4 because we process 4 pixels at once)
    if (vec_x >= w / 4 || y >= h) return;

    // --- 1. READ 4 PIXELS (One 32-bit Load) ---
    // Cast the byte pointer to an int pointer
    const int* in_ptr_as_int = (const int*)img_in;
    
    // Load 32 bits (4 pixels) in one go. Much faster bus utilization.
    int packed_pixels = in_ptr_as_int[y * (w / 4) + vec_x];

    // Unpack using bitwise operations
    unsigned char p0 = packed_pixels & 0xFF;         // Byte 0
    unsigned char p1 = (packed_pixels >> 8) & 0xFF;  // Byte 1
    unsigned char p2 = (packed_pixels >> 16) & 0xFF; // Byte 2
    unsigned char p3 = (packed_pixels >> 24) & 0xFF; // Byte 3

    // --- 2. PROCESS (Helper function recommended) ---
    int base_x = vec_x * 4; // The actual pixel coordinate
    
    // We reuse the calculation logic (same as previous examples)
    // Note: You would paste your CLAHE math here for each pixel
    unsigned char r0 = calculate_clahe_pixel(base_x + 0, y, p0, w, grid_w, grid_h, tex_luts);
    unsigned char r1 = calculate_clahe_pixel(base_x + 1, y, p1, w, grid_w, grid_h, tex_luts);
    unsigned char r2 = calculate_clahe_pixel(base_x + 2, y, p2, w, grid_w, grid_h, tex_luts);
    unsigned char r3 = calculate_clahe_pixel(base_x + 3, y, p3, w, grid_w, grid_h, tex_luts);

    // --- 3. WRITE 4 PIXELS (One 32-bit Store) ---
    // Pack results back into a single integer
    int packed_result = 0;
    packed_result |= r0;
    packed_result |= (r1 << 8);
    packed_result |= (r2 << 16);
    packed_result |= (r3 << 24);

    // Cast output pointer and write
    int* out_ptr_as_int = (int*)img_out;
    out_ptr_as_int[y * (w / 4) + vec_x] = packed_result;
}