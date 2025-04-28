#ifndef DEFINE_H
#define DEFINE_H

// Max neurons and layers
#define MAX_LAYERS 3
#define MAX_NEURONS 784

// Network parameters
#define VOLTAGE_THRESH 1.0
#define DECAY_RATE 0.95

// Data parameters
#define NUM_SAMPLES 1

// Network parameters
#define INPUT_SIZE 784 // 28x28 flattened images
#define NUM_CLASSES 10 
#define HIDDEN_LAYER_1 256
#define NUM_LAYERS 3

// Temporal parameters
#define TIME_WINDOW 20 // Temporal steps in spike train
#define TAU 10

// Masking parameters
#define BITMASK_BYTES ((TAU + 7) / 8)
#define INPUT_BYTES ((INPUT_SIZE + 7) / 8)

// Define the quantization parameters for Q0.7
#define Q07_FLAG       1
#define Q07_SCALE      128.0f
#define Q07_INV_SCALE  (1.0f / 128.0f)
#define Q07_MAX_FLOAT  0.9921875f   // 127 / 128
#define Q07_MIN_FLOAT -1.0f
#define Q07_MAX_INT8   127
#define Q07_MIN_INT8  -128

#define LIF 1
#define IF  0

#endif // DEFINE_H
