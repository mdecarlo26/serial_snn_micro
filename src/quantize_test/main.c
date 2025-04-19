#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

#define NUM_SAMPLES 16
#define Q07_SCALE      128.0f
#define Q07_INV_SCALE  (1.0f / 128.0f)
#define Q07_MAX_FLOAT  0.9921875f   // 127 / 128
#define Q07_MIN_FLOAT -1.0f
#define Q07_MAX_INT8   127
#define Q07_MIN_INT8  -128


// === Quantize float32 to Q0.8 (int8_t) ===
// Range: [-1.0, 0.99609375]
int8_t quantize_q07(float x) {
    // Clamp to representable Q0.7 range
    if (x > Q07_MAX_FLOAT)  x = Q07_MAX_FLOAT;
    if (x < Q07_MIN_FLOAT)  x = Q07_MIN_FLOAT;

    // Scale and round
    int32_t scaled = (int32_t)(x * Q07_SCALE + (x >= 0 ? 0.5f : -0.5f));

    // Clamp to int8_t just in case
    if (scaled > Q07_MAX_INT8)  scaled = Q07_MAX_INT8;
    if (scaled < Q07_MIN_INT8)  scaled = Q07_MIN_INT8;

    return (int8_t)scaled;
}

// === Dequantize Q0.8 (int8_t) to float32 ===
// Output = q / 256.0
float dequantize_q07(int8_t q) {
    return ((float)q) * Q07_INV_SCALE;
}


int main() {
// Range: [-1.0, 0.99609375]
    float nums[NUM_SAMPLES] = {0.005, 0.006, 0.007, 0.008,  0.01, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024};
    int8_t q[NUM_SAMPLES] = {0};
    float d[NUM_SAMPLES] = {0};

    // Encode and decode the numbers
    for (int i = 0; i < NUM_SAMPLES; i++) {
        q[i] = quantize_q07(nums[i]);
        d[i] = dequantize_q07(q[i]);
    }

    // Print the results
    printf("Original\tEncoded\tDecoded\n");
    for (int i = 0; i < NUM_SAMPLES; i++) {
        printf("%f\t%d\t%f\n", nums[i], q[i], d[i]);
    }
    return 0;
}