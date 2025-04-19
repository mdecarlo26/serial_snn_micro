#ifndef DEFINE_H
#define DEFINE_H

#define MAX_LAYERS 3
#define MAX_NEURONS 784
#define TAU 10
#define VOLTAGE_THRESH 1.0
#define DECAY_RATE 0.95
#define NUM_SAMPLES 1 // Total dataset size
#define TIME_WINDOW 20         // Temporal steps in spike train
#define INPUT_SIZE 784 // 28x28 flattened images
#define NUM_CLASSES 10 
#define HIDDEN_LAYER_1 256
#define NUM_LAYERS 3

#define BITMASK_BYTES ((TAU + 7) / 8)
#define INPUT_BYTES ((INPUT_SIZE + 7) / 8)

#define Q07_SCALE      128.0f
#define Q07_INV_SCALE  (1.0f / 128.0f)
#define Q07_MAX_FLOAT  0.9921875f   // 127 / 128
#define Q07_MIN_FLOAT -1.0f
#define Q07_MAX_INT8   127
#define Q07_MIN_INT8  -128

#endif // DEFINE_H
