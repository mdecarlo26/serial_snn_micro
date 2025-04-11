#ifndef DEFINE_H
#define DEFINE_H

#define MAX_LAYERS 10
#define MAX_NEURONS 1000
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


#endif // DEFINE_H
