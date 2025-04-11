#ifndef DUMMY_H
#define DUMMY_H

#include "define.h"
#include <stdint.h>

extern uint8_t input_data[784];
extern char label;

extern float bias_fc1[HIDDEN_LAYER_1];
extern float bias_fc2[NUM_CLASSES];

extern float weights_fc1_data[HIDDEN_LAYER_1][INPUT_SIZE];
extern float weights_fc2_data[NUM_CLASSES][HIDDEN_LAYER_1];

#endif // DUMMY_H
