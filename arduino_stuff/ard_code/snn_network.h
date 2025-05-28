#ifndef SNN_NETWORK_H
#define SNN_NETWORK_H

#include "define.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    LAYER_INPUT,
    LAYER_CONV,
    LAYER_FC,
    LAYER_OUTPUT
} LayerType;

typedef struct {
#if (Q07_FLAG)
    int32_t  membrane_potentials[MAX_NEURONS];
    int32_t  delayed_resets     [MAX_NEURONS];
    int16_t  voltage_thresholds [MAX_NEURONS];
    int16_t  decay_rates        [MAX_NEURONS];
#else
    float    membrane_potentials[MAX_NEURONS];
    float    delayed_resets     [MAX_NEURONS];
    float    voltage_thresholds [MAX_NEURONS];
    float    decay_rates        [MAX_NEURONS];
#endif

    const int8_t (*conv_weights_col)[CONV1_FILTERS]; // [9][4]
    const int8_t *conv_bias;                         // [4]

    int8_t **weights;
    int8_t *bias;
    int num_neurons;
    int layer_num;
    LayerType type;
} Layer;

typedef struct {
    Layer *layers;
    int num_layers;
} Snn_Network;

// void initialize_network(int neurons_per_layer[],const int8_t weights_fc1[INPUT_SIZE][HIDDEN_LAYER_1],
//     const int8_t weights_fc2[HIDDEN_LAYER_1][NUM_CLASSES],const int8_t *bias_fc1, const int8_t *bias_fc2);

void initialize_network(int neurons_per_layer[],
    // new conv layer weights & bias:
    const int8_t conv1_weights_col[CONV1_KERNEL*CONV1_KERNEL][CONV1_FILTERS],
    const int8_t conv1_bias[CONV1_FILTERS],
    // final FC layer (from conv → output):
    const int8_t weights_fc2_data[CONV1_NEURONS][NUM_CLASSES],
    const int8_t bias_fc2[NUM_CLASSES]);

void zero_network();
void free_network();

void update_layer(const uint8_t input[TAU][INPUT_BYTES],
                  uint8_t output[TAU][INPUT_BYTES],
                  Layer *layer, int input_size);

int inference(const uint8_t input[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES], int sample_idx);
int classify_inference(int **firing_counts, int num_neurons, int num_chunks);

int get_input_spike(const uint8_t buffer[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES],
                    int sample, int t, int neuron_idx);
void set_input_spike(uint8_t buffer[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES],
                     int sample, int t, int neuron_idx, int value);

int8_t quantize_q07(float x); 
float dequantize_q07(int32_t q);

void compute_buffer_sparsity(const uint8_t buffer[TAU][INPUT_BYTES],
                             int num_neurons,
                             float sparsity[TAU]);

void conv_sparse_q7_add(
    const uint8_t *in_bits,                 
    const int8_t  (*W)[CONV1_FILTERS],      
    const int8_t  *bias,                    
    int32_t       *sums                     
);
#endif // NETWORK_H
