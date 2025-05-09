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

void vector_pack_bits(
    const int32_t * __restrict mask,
    uint8_t       * __restrict bitbuf,
    size_t          blockSize
);

void vector_sub_where_q31(
    const int32_t * __restrict mask,
    const int16_t * __restrict thresh,
    int32_t       * __restrict dst,
    size_t          blockSize
);


void vector_scale_q31(
    const int32_t * __restrict src,
    int16_t         scale_fp7,
    int              shift,
    int32_t       * __restrict dst,
    size_t          blockSize
);

void vector_compare_ge_q31(
    const int32_t * __restrict src,
    const int16_t * __restrict thresh,
    int32_t       * __restrict mask,
    size_t          blockSize
);

#endif // DSP_HELPER_H