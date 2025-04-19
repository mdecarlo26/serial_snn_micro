#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

#define NUM_SAMPLES 4
#define Q08_INV_SCALE (1.0f / 256.0f)

// === Quantize float32 to Q0.8 (int8_t) ===
// Range: [-1.0, 0.99609375]
int8_t quantize_q08(float x) {
    // Clamp to Q0.8 range
    if (x > 0.99609375f) x = 0.99609375f;
    if (x < -1.0f)        x = -1.0f;

    // Scale and round
    int32_t scaled = (int32_t)(x * 256.0f + (x >= 0 ? 0.5f : -0.5f));

    // Final clamp to int8_t limits
    if (scaled > 127)  scaled = 127;
    if (scaled < -128) scaled = -128;

    return (int8_t)scaled;
}

// === Dequantize Q0.8 (int8_t) to float32 ===
// Output = q / 256.0
float dequantize_q08(int8_t q) {
    return ((float)q) * Q08_INV_SCALE;
}


int main() {
// Range: [-1.0, 0.99609375]
    float nums[NUM_SAMPLES] = {-1.0,-0.5,0.0,0.5,};
    int8_t q[NUM_SAMPLES] = {0};
    float d[NUM_SAMPLES] = {0};

    // Encode and decode the numbers
    for (int i = 0; i < NUM_SAMPLES; i++) {
        q[i] = quantize_q08(nums[i]);
        d[i] = dequantize_q08(q[i]);
    }

    // Print the results
    printf("Original\tEncoded\tDecoded\n");
    for (int i = 0; i < NUM_SAMPLES; i++) {
        printf("%f\t%d\t%f\n", nums[i], q[i], d[i]);
    }
    return 0;
}