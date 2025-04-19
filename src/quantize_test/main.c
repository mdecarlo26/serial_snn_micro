#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

uint8_t encode_q17(float x) {
    if (x > 1.984375f) x = 1.984375f;
    if (x < -1.984375f) x = -1.984375f;

    uint8_t sign = (x < 0);
    float abs_x = fabsf(x);
    uint8_t integer = (uint8_t)abs_x;  // 0 or 1
    uint8_t fraction = (uint8_t)((abs_x - integer) * 64.0f);  // [0,63]

    return (sign << 7) | (integer << 6) | (fraction & 0x3F);
}

float decode_q17(uint8_t q) {
    uint8_t sign = (q >> 7) & 0x1;
    uint8_t integer = (q >> 6) & 0x1;
    uint8_t fraction = q & 0x3F;

    float val = integer + (fraction / 64.0f);
    return sign ? -val : val;
}


int main() {
    // Example usage
    float input = 1.00039f;
    uint8_t encoded = encode_q17(input);
    float decoded = decode_q17(encoded);

    printf("Original: %f\n", input);
    printf("Encoded: %u\n", encoded);
    printf("Decoded: %f\n", decoded);

    return 0;
}