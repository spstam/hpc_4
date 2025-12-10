#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* * Standard definition for PSNR calculation:
 * PSNR = 10 * log10 ( MAX^2 / MSE )
 * Where MAX is usually 255 for 8-bit images.
 */

// Function to calculate MSE (Mean Squared Error)
double calculate_mse(unsigned char *img1, unsigned char *img2, long num_pixels) {
    double sum_sq_diff = 0.0;
    long i;

    for (i = 0; i < num_pixels; i++) {
        double diff = (double)img1[i] - (double)img2[i];
        sum_sq_diff += (diff * diff);
    }

    return sum_sq_diff / (double)num_pixels;
}

// Function to get file size
long get_file_size(FILE *f) {
    long size;
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    rewind(f);
    return size;
}

int main(int argc, char *argv[]) {
    // 1. Validate Arguments
    if (argc != 3) {
        printf("Usage: %s <image1_filename> <image2_filename>\n", argv[0]);
        return 1;
    }

    FILE *f1, *f2;
    unsigned char *buffer1, *buffer2;
    long size1, size2;
    double mse, psnr;
    double max_val = 255.0; // Assuming 8-bit depth

    // 2. Open Files
    f1 = fopen(argv[1], "rb"); // 'rb' for read binary
    f2 = fopen(argv[2], "rb");

    if (!f1) {
        fprintf(stderr, "Error: Could not open file %s\n", argv[1]);
        return 1;
    }
    if (!f2) {
        fprintf(stderr, "Error: Could not open file %s\n", argv[2]);
        fclose(f1);
        return 1;
    }

    // 3. Check File Sizes
    size1 = get_file_size(f1);
    size2 = get_file_size(f2);

    if (size1 != size2) {
        fprintf(stderr, "Error: Image dimensions/sizes do not match.\n");
        fprintf(stderr, "%s: %ld bytes\n", argv[1], size1);
        fprintf(stderr, "%s: %ld bytes\n", argv[2], size2);
        fclose(f1);
        fclose(f2);
        return 1;
    }

    // 4. Allocate Memory
    buffer1 = (unsigned char *)malloc(size1 * sizeof(unsigned char));
    buffer2 = (unsigned char *)malloc(size2 * sizeof(unsigned char));

    if (!buffer1 || !buffer2) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        fclose(f1);
        fclose(f2);
        return 1;
    }

    // 5. Read Data
    size_t result1 = fread(buffer1, 1, size1, f1);
    size_t result2 = fread(buffer2, 1, size2, f2);

    if (result1 != size1 || result2 != size2) {
        fprintf(stderr, "Error: Reading error.\n");
        free(buffer1); free(buffer2);
        fclose(f1); fclose(f2);
        return 1;
    }

    // 6. Calculate MSE and PSNR
    mse = calculate_mse(buffer1, buffer2, size1);

    if (mse == 0.0) {
        // If MSE is 0, images are identical, PSNR is infinite
        printf("Images are identical.\n");
        printf("MSE: 0.00\n");
        printf("PSNR: Infinity\n");
    } else {
        // PSNR formula: 10 * log10( (MAX_I^2) / MSE )
        psnr = 10.0 * log10((max_val * max_val) / mse);

        printf("------------------------------\n");
        printf("File 1: %s\n", argv[1]);
        printf("File 2: %s\n", argv[2]);
        printf("Total Pixels/Bytes: %ld\n", size1);
        printf("------------------------------\n");
        printf("MSE:  %.4f\n", mse);
        printf("PSNR: %.4f dB\n", psnr);
        printf("------------------------------\n");
    }

    // 7. Cleanup
    free(buffer1);
    free(buffer2);
    fclose(f1);
    fclose(f2);

    return 0;
}