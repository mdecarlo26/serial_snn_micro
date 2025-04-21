#ifndef DUMMY_H
#define DUMMY_H

#include "define.h"
#include <stdint.h>

extern const uint8_t input_data[784];
extern const char label;

extern const int8_t bias_fc1[HIDDEN_LAYER_1];
extern const int8_t bias_fc2[NUM_CLASSES];

extern const int8_t weights_fc1_data[HIDDEN_LAYER_1][INPUT_SIZE];
extern const int8_t weights_fc2_data[NUM_CLASSES][HIDDEN_LAYER_1];

#endif // DUMMY_H
