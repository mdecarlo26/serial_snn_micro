#include "dsp_helper.h"
#include <string.h>

inline void vectorize_q7_add_to_q31(
    const int8_t * __restrict srcA,
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

/// Scale Q31 array by a Q7 fixed-point factor (DECAY_FP7 >> DECAY_SHIFT).
inline void vector_scale_q31(
    const int32_t * src,
    int16_t         scale_fp7,
    int              shift,
    int32_t       * dst,
    size_t          blockSize
) {
    size_t i=0;
    for (; i+3<blockSize; i+=4) {
        dst[i]   = (src[i]   * scale_fp7) >> shift;
        dst[i+1] = (src[i+1] * scale_fp7) >> shift;
        dst[i+2] = (src[i+2] * scale_fp7) >> shift;
        dst[i+3] = (src[i+3] * scale_fp7) >> shift;
    }
    for (; i<blockSize; i++) {
        dst[i] = (src[i] * scale_fp7) >> shift;
    }
}

/// Compare src >= thresh elementwise, write 0/1 into mask[]
inline void vector_compare_ge_q31(
    const int32_t * __restrict src,
    const int16_t * __restrict thresh,
    int32_t       * __restrict mask,
    size_t          blockSize
) {
    size_t i=0;
    for (; i+3<blockSize; i+=4) {
        mask[i]   = (src[i]   >= thresh[i]);
        mask[i+1] = (src[i+1] >= thresh[i+1]);
        mask[i+2] = (src[i+2] >= thresh[i+2]);
        mask[i+3] = (src[i+3] >= thresh[i+3]);
    }
    for (; i<blockSize; i++) {
        mask[i] = (src[i] >= thresh[i]);
    }
}

/// Subtract thresh[i] from dst[i] only where mask[i]==1
inline void vector_sub_where_q31(
    const int32_t * __restrict mask,
    const int16_t * __restrict thresh,
    int32_t       * __restrict dst,
    size_t          blockSize
) {
    size_t i=0;
    for (; i+3<blockSize; i+=4) {
        if (mask[i  ]) dst[i  ] -= thresh[i  ];
        if (mask[i+1]) dst[i+1] -= thresh[i+1];
        if (mask[i+2]) dst[i+2] -= thresh[i+2];
        if (mask[i+3]) dst[i+3] -= thresh[i+3];
    }
    for (; i<blockSize; i++) {
        if (mask[i]) dst[i] -= thresh[i];
    }
}

/// Pack a 0/1 mask[] back into a bit-buffer
inline void vector_pack_bits(
    const int32_t * __restrict mask,
    uint8_t       * __restrict bitbuf,
    size_t          blockSize
) {
    // Clear buffer
    memset(bitbuf, 0, (blockSize+7)/8);
    for (size_t i=0; i<blockSize; i++) {
        if (mask[i]) bitbuf[i>>3] |= 1u << (i & 7);
    }
}