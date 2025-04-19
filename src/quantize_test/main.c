#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

#define NUM_SAMPLES 16

uint16_t float32_to_float16(float value) {
    uint32_t f32;
    memcpy(&f32, &value, sizeof(f32));

    uint32_t sign     = (f32 >> 31) & 0x1;
    int32_t  exponent = ((f32 >> 23) & 0xFF) - 127 + 15; // re-bias
    uint32_t mantissa = f32 & 0x7FFFFF;

    uint16_t result;

    if (exponent <= 0) {
        // Subnormal or zero
        if (exponent < -10) {
            result = (uint16_t)(sign << 15);  // Too small → 0
        } else {
            mantissa |= 0x800000;  // Implicit leading 1
            int shift = 14 - exponent;
            result = (uint16_t)((sign << 15) | (mantissa >> shift));
        }
    } else if (exponent >= 31) {
        // Overflow → Inf
        result = (uint16_t)((sign << 15) | (0x1F << 10));
    } else {
        // Normalized
        result = (uint16_t)((sign << 15) | (exponent << 10) | (mantissa >> 13));
    }

    return result;
}

// === Convert float16 to float32 ===
float float16_to_float32(uint16_t h) {
    uint32_t sign     = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t f32;

    if (exponent == 0) {
        // Subnormal or zero
        if (mantissa == 0) {
            f32 = sign << 31;  // Zero
        } else {
            // Subnormal
            exponent = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            exponent = exponent - 1 + (127 - 15);
            f32 = (sign << 31) | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        // Inf or NaN
        f32 = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        // Normalized
        exponent = exponent + (127 - 15);
        f32 = (sign << 31) | (exponent << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

int main() {
// Range: [-1.0, 0.99609375]
    float nums[NUM_SAMPLES] = {0.004, 0.006, 0.007, 0.008,  0.01, 0.012, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023};
    uint16_t q[NUM_SAMPLES] = {0};
    float d[NUM_SAMPLES] = {0};

    // Encode and decode the numbers
    for (int i = 0; i < NUM_SAMPLES; i++) {
        q[i] = float32_to_float16(nums[i]);
        d[i] = float16_to_float32(q[i]);
    }

    // Print the results
    printf("Original\tEncoded\tDecoded\n");
    for (int i = 0; i < NUM_SAMPLES; i++) {
        printf("%f\t%d\t%f\n", nums[i], q[i], d[i]);
    }
    return 0;
}