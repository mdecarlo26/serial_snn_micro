#ifndef DUMMY_H
#define DUMMY_H

#include "define.h"
#include <stdint.h>

extern const uint8_t input_data[784];
extern const char label;

// extern const int8_t bias_fc1[HIDDEN_LAYER_1];
extern const int8_t bias_fc2[NUM_CLASSES];

// extern const int8_t weights_fc1_data[INPUT_SIZE][HIDDEN_LAYER_1];
extern const int8_t weights_fc2_data[CONV1_NEURONS][NUM_CLASSES];

extern const int8_t conv1_weights_col[CONV1_KERNEL_FLATTEN][CONV1_FILTERS];
extern const int8_t conv1_bias[CONV1_FILTERS];

#endif // DUMMY_H
