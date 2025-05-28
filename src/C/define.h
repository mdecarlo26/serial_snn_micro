#ifndef DEFINE_H
#define DEFINE_H

// Macros for max and min operations
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define MAX3(a, b, c) MAX2(a, MAX2(b, c))
#define MAX4(a, b, c, d) MAX2(MAX2(a, b), MAX2(c, d))
#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MIN3(a, b, c) MIN2(a, MIN2(b, c))
#define MIN4(a, b, c, d) MIN2(MIN2(a, b), MIN2(c, d))

// Convolution Size Macro
#define CONV_OUT_DIM_SIZE(in_dim, kernel_size, stride, padding) \
  ((in_dim - kernel_size + 2 * padding) / stride + 1)

// Convolution parameters
#define CONV1_FILTERS      4
#define CONV1_KERNEL       3
#define CONV1_KERNEL_FLATTEN (CONV1_KERNEL * CONV1_KERNEL)
#define CONV1_H_IN         28
#define CONV1_W_IN         28
#define CONV1_H_OUT        CONV_OUT_DIM_SIZE(CONV1_H_IN, CONV1_KERNEL, 1, 0)
#define CONV1_W_OUT        CONV_OUT_DIM_SIZE(CONV1_W_IN, CONV1_KERNEL, 1, 0)
#define CONV1_NEURONS      (CONV1_FILTERS * CONV1_H_OUT * CONV1_W_OUT)

// Network parameters
#define INPUT_SIZE 784 // 28x28 flattened images
#define NUM_CLASSES 10 
#define HIDDEN_LAYER_1 256
#define NUM_LAYERS 3

// Max neurons and layers
#define MAX_LAYERS 2
#define MAX_NEURONS MAX2(CONV1_NEURONS, NUM_CLASSES)

// Network parameters
#define VOLTAGE_THRESH 1.0f
#define DECAY_RATE 0.95f

// Data parameters
#define NUM_SAMPLES 1

// Temporal parameters
#define TIME_WINDOW 20 // Temporal steps in spike train
#define TAU 10

// Masking parameters
#define BITMASK_BYTES ((TAU + 7) / 8)
#define INPUT_BYTES ((MAX_NEURONS + 7) / 8)

// Define the quantization parameters for Q0.7
#define Q07_FLAG       1
#define Q07_SCALE      128.0f
#define Q07_INV_SCALE  (1.0f / 128.0f)
#define Q07_MAX_FLOAT  0.9921875f   // 127 / 128
#define Q07_MIN_FLOAT -1.0f
#define Q07_MAX_INT8   127
#define Q07_MIN_INT8  -128

// Clamp, scale, round-to-nearest, and cast to int8 Q0.7
#define FLOAT_TO_Q07(x)                                                      \
  ( (int8_t)(                                                              \
      ( (x) >  Q07_MAX_FLOAT ) ?  Q07_MAX_INT8  :                           \
      ( (x) <  Q07_MIN_FLOAT ) ?  Q07_MIN_INT8  :                           \
      (int32_t)( (x) * Q07_SCALE + ((x) >= 0 ? 0.5f : -0.5f) )               \
    )                                                                       \
  )

#define VOLTAGE_THRESH_FP7  ((int16_t)(VOLTAGE_THRESH * 128))   // 128
#define DECAY_FP7           ((int16_t)(DECAY_RATE * 128))  // ~121
#define DECAY_SHIFT         7                         // because scale = 128 = 2^7

#ifdef Q07_FLAG
  // For fixed-point: mem and thresh are integer Q0.7 values
  #define HEAVISIDE(mem, thresh) (((mem) >= (thresh)) ? 1 : 0)
#else
  // For float path: mem and thresh are floats
  #define HEAVISIDE(mem, thresh) (((mem) >= (thresh)) ? 1 : 0)
#endif

#define GET_BIT(buffer, idx) ((buffer[(idx) >> 3] >> ((idx) & 7)) & 1)
#define SET_BIT(buffer, idx, value) \
    ((value) ? ((buffer)[(idx) >> 3] |= (1 << ((idx) & 7))) \
             : ((buffer)[(idx) >> 3] &= ~(1 << ((idx) & 7))))
#define CLEAR_BIT(buffer, idx) (buffer[(idx) >> 3] &= ~(1 << ((idx) & 7)))

#define LIF 1
#define IF  0

#endif // DEFINE_H
