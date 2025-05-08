#include "dsp_helper.h"

inline void vectorize_q7_add_to_q31(
    const int8_t * __restrict srcA,
    int32_t       * __restrict dst,
    size_t          blockSize
) {
    size_t i = 0;
    // unroll 4 at a time
    for (; i + 3 < blockSize; i += 4) {
        dst[i]     = ((int32_t)srcA[i]   << 24) + dst[i];
        dst[i+1]   = ((int32_t)srcA[i+1]   << 24) + dst[i+1];
        dst[i+2]   = ((int32_t)srcA[i+2]   << 24) + dst[i+2];
        dst[i+3]   = ((int32_t)srcA[i+3]   << 24) + dst[i+3];
    }
    // leftover
    for (; i < blockSize; i++) {
        dst[i] = ((int32_t)srcA[i] << 24) + dst[i];                
    }
}

inline void vectorize_q31_add_to_q31(
    const int32_t * __restrict srcA,
    int32_t       * __restrict dst,
    size_t          blockSize
) {
    size_t i = 0;
    // unroll 4 at a time
    for (; i + 3 < blockSize; i += 4) {
        dst[i]     = srcA[i] + dst[i];
        dst[i+1]   = srcA[i+1] + dst[i+1];
        dst[i+2]   = srcA[i+2] + dst[i+2];
        dst[i+3]   = srcA[i+3] + dst[i+3];
    }
    // leftover
    for (; i < blockSize; i++) {
        dst[i] = srcA[i] + dst[i];                
    }
}

inline void vectorize_float_add_to_float(
    const float * __restrict srcA,
    float       * __restrict dst,
    size_t          blockSize
) {
    size_t i = 0;
    // unroll 4 at a time
    for (; i + 3 < blockSize; i += 4) {
        dst[i]     = srcA[i] + dst[i];
        dst[i+1]   = srcA[i+1] + dst[i+1];
        dst[i+2]   = srcA[i+2] + dst[i+2];
        dst[i+3]   = srcA[i+3] + dst[i+3];
    }
    // leftover
    for (; i < blockSize; i++) {
        dst[i] = srcA[i] + dst[i];                
    }
}