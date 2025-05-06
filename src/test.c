#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


#include <stdint.h>
#include <stddef.h>

/// Convert an array of Q7 (int8_t) values into Q31 (int32_t) by left-shifting 24 bits:
///   dst[n] = (int32_t)src[n] << 24;
static inline void q7_to_q31(
    const int8_t * __restrict src,
    int32_t      * __restrict dst,
    size_t        blockSize
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
) __attribute__((always_inline));
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


int main() {

    uint8_t num = 128;

    int ret = __builtin_ctz(num);
    printf("The number of trailing zeros in %u is: %d\n", num, ret);

    return 0;

}