#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>


#include <stdint.h>
#include <stddef.h>

/// Convert an array of Q7 (int8_t) values into Q31 (int32_t) by left-shifting 24 bits:
///   dst[n] = (int32_t)src[n] << 24;
static inline void q7_to_q31(
    const int8_t * __restrict src,
    int32_t      * __restrict dst,
    size_t        blockSize
) __attribute__((always_inline));

static inline void add_q31(
    const int32_t * __restrict srcA,
    const int32_t * __restrict srcB,
    int32_t       * __restrict dst,
    size_t          blockSize
) __attribute__((always_inline));




static inline void q7_to_q31(
    const int8_t * __restrict src,
    int32_t      * __restrict dst,
    size_t        blockSize
) {
    size_t i = 0;
    // process 4 at a time for unrolled/SIMD-friendly code
    for (; i + 3 < blockSize; i += 4) {
        dst[i]   = ((int32_t)src[i]   << 24);
        dst[i+1] = ((int32_t)src[i+1] << 24);
        dst[i+2] = ((int32_t)src[i+2] << 24);
        dst[i+3] = ((int32_t)src[i+3] << 24);
    }
    // leftover
    for (; i < blockSize; i++) {
        dst[i] = ((int32_t)src[i] << 24);
    }
}

/// Element-wise add two Q31 arrays into dst:
///   dst[n] = srcA[n] + srcB[n];
static inline void add_q31(
    const int32_t * __restrict srcA,
    const int32_t * __restrict srcB,
    int32_t       * __restrict dst,
    size_t          blockSize
) {
    size_t i = 0;
    // unroll 4 at a time
    for (; i + 3 < blockSize; i += 4) {
        dst[i]   = srcA[i]   + srcB[i];
        dst[i+1] = srcA[i+1] + srcB[i+1];
        dst[i+2] = srcA[i+2] + srcB[i+2];
        dst[i+3] = srcA[i+3] + srcB[i+3];
    }
    for (; i < blockSize; i++) {
        dst[i] = srcA[i] + srcB[i];
    }
}


#define INPUT_SIZE 8

int main() {

    // Example usage of q7_to_q31 and add_q31 functions
    int8_t srcA[INPUT_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8};
    int8_t srcB[INPUT_SIZE] = {8, 7, 6, 5, 4, 3, 2, 1};
    int32_t dstA[INPUT_SIZE] = {0};
    int32_t dstB[INPUT_SIZE] = {0};
    size_t blockSize = INPUT_SIZE;

    // Convert srcA and srcB to Q31 format
    q7_to_q31(srcA, dstA, blockSize);
    q7_to_q31(srcB, dstB, blockSize);

    
    struct timeval start, end;

    // Add the two Q31 arrays
    int32_t result[INPUT_SIZE] = {0};
        gettimeofday(&start, NULL);
    add_q31(dstA, dstB, result, blockSize);
        gettimeofday(&end, NULL);

    // Print the result
    printf("Q31 addition result:\n");
    for (size_t i = 0; i < blockSize; i++) {
        printf("result[%zu] = %d\n", i, result[i]);
    }
    printf( "CPU run time = %0.6f s\n", (float)(end.tv_sec - start.tv_sec\
                + (end.tv_usec - start.tv_usec) / (float)1000000));


    // Add them tradionally
    int32_t result_2[INPUT_SIZE] = {0};
        gettimeofday(&start, NULL);
    for (size_t i = 0; i < blockSize; i++) {
        result_2[i] = ((int32_t)srcA[i] << 24) + ((int32_t)srcB[i] << 24);
    }
        gettimeofday(&end, NULL);
    printf("\nTraditional addition:\n");
    for (size_t i = 0; i < blockSize; i++) {
        printf("result[%zu] = %d\n", i, result[i]);
    }
    printf( "CPU run time = %0.6f s\n", (float)(end.tv_sec - start.tv_sec\
                + (end.tv_usec - start.tv_usec) / (float)1000000));

    return 0;

}