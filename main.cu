#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "clahe.h"
#include "gputimer.h"

double get_time_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]){
    if (argc != 3) {
        printf("Usage: %s <input.pgm> <output.pgm>\n", argv[0]);
        return 1;
    }

    //load image
    printf("Loading image...\n");
    // NOTE: read_pgm typically uses standard malloc. 
    // For optimal pinned memory usage, we ideally load directly into pinned memory,
    // but for simplicity here, we will allocate pinned memory and copy the data into it.
    PGM_IMG img_temp = read_pgm(argv[1]);
    
    PGM_IMG img_in;
    img_in.w = img_temp.w;
    img_in.h = img_temp.h;
    size_t img_size = img_in.w * img_in.h * sizeof(unsigned char);

    // ALLOCATE PINNED HOST MEMORY
    cudaMallocHost((void**)&img_in.img, img_size);
    memcpy(img_in.img, img_temp.img, img_size); // Move data to pinned memory
    free_pgm(img_temp); // Free the original standard memory

    // ============================================
    // 1. EXEC CPU (Serial)
    // ============================================
    printf("\n--- Starting CPU Execution ---\n");
    double start_cpu = get_time_sec();
    
    PGM_IMG img_out_cpu = apply_clahe_cpu(img_in);
    
    double time_cpu = get_time_sec() - start_cpu;
    printf("CPU Time:       %.6f sec\n", time_cpu);
    printf("CPU Throughput: %.2f MPixels/s\n", (img_in.w * img_in.h) / (time_cpu * 1e6));

    write_pgm(img_out_cpu, "out_seq.pgm");

    // ============================================
    // 2. EXEC GPU (Parallel with Streams & Pinned Memory)
    // ============================================
    
    int grid_w = (img_in.w + TILE_SIZE - 1) / TILE_SIZE;
    int grid_h = (img_in.h + TILE_SIZE - 1) / TILE_SIZE;
    printf("Image: %dx%d, Grid: %dx%d\n", img_in.w, img_in.h, grid_w, grid_h);

    printf("\n--- Starting GPU Execution (Streams + Pinned) ---\n");
    
    unsigned char *d_in, *d_out;
    int *d_luts;
    size_t lut_size = grid_w * grid_h * 256 * sizeof(int);

    // Prepare Host output buffer (Pinned)
    PGM_IMG img_out_gpu;
    img_out_gpu.w = img_in.w; img_out_gpu.h = img_in.h;
    cudaMallocHost((void**)&img_out_gpu.img, img_size);

    {
        // 1. Create Stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        GpuTimer timer;
        timer.Start();

        // 2. Allocate Device Memory
        cudaMalloc((void**)&d_in, img_size);
        cudaMalloc((void**)&d_out, img_size);
        cudaMalloc((void**)&d_luts, lut_size);

        // 3. Copy Input Host -> Device (Async)
        // Pinned memory makes this significantly faster
        cudaMemcpyAsync(d_in, img_in.img, img_size, cudaMemcpyHostToDevice, stream);

        // 4. Launch Kernels on Stream
        dim3 hist_block(TILE_SIZE, 8);
        dim3 hist_grid(grid_w, grid_h);
        
        histogram_lut_kernel<<<hist_grid, hist_block, 0, stream>>>(d_in, d_luts, img_in.w, img_in.h, grid_w, CLIP_LIMIT);

        dim3 render_block(32, 32);
        dim3 render_grid((img_in.w/4 + 31) / 32, (img_in.h + 31) / 32);
        render_clahe_kernel<<<render_grid, render_block, 0, stream>>>(d_in, d_out, d_luts, img_in.w, img_in.h, grid_w, grid_h);

        // 5. Copy Output Device -> Host (Async)
        cudaMemcpyAsync(img_out_gpu.img, d_out, img_size, cudaMemcpyDeviceToHost, stream);

        // 6. Synchronize
        cudaStreamSynchronize(stream);

        timer.Stop();
        float time_gpu = timer.Elapsed();
        
        printf("GPU Time:       %.6f ms\n", time_gpu);
        time_gpu = time_gpu / 1000; 
        printf("GPU Throughput: %.2f MPixels/s\n", (img_in.w * img_in.h) / (time_gpu * 1e6));
    
        // ============================================
        // 3. COMPARE
        // ============================================
        printf("\n--- Results ---\n");
        printf("Speedup: %.2fx faster on GPU\n", time_cpu / time_gpu);

        write_pgm(img_out_gpu, argv[2]);
        printf("Output image saved to %s\n", argv[2]);

        // Cleanup
        cudaFree(d_in); 
        cudaFree(d_out); 
        cudaFree(d_luts);
        cudaStreamDestroy(stream);
        
        // Free Pinned Memory
        cudaFreeHost(img_in.img);
        cudaFreeHost(img_out_gpu.img);
        
        // Note: apply_clahe_cpu typically allocates with standard malloc, so we free that normally
        free_pgm(img_out_cpu);
    }
    return 0;
}