#ifndef DSP_HELPER_H
#define DSP_HELPER_H

#include <stdint.h>
#include <stddef.h>

void vectorize_q7_add_to_q31(
    const int8_t * __restrict srcA,
    int32_t       * __restrict dst,
    size_t          blockSize
);

void vectorize_q31_add_to_q31(
    const int32_t * __restrict srcA,
    int32_t       * __restrict dst,
    size_t          blockSize
);

void vectorize_float_add_to_float(
    const float * __restrict srcA,
    float * __restrict dst,
    size_t          blockSize
);

#endif // DSP_HELPER_H