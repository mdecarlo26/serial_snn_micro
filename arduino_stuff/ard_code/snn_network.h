#ifndef SNN_NETWORK_H
#define SNN_NETWORK_H

#include "define.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
#if (Q07_FLAG)
    int32_t membrane_potential;
    int32_t delayed_reset;
    int16_t voltage_thresh;
    int16_t decay_rate;
#else
    float membrane_potential;
    float voltage_thresh;
    float decay_rate;
    float delayed_reset;
#endif
} Neuron;

typedef struct {
    Neuron *neurons;
    int8_t **weights;
    int8_t *bias;
    int num_neurons;
    int layer_num;
} Layer;

typedef struct {
    Layer *layers;
    int num_layers;
} Snn_Network;

void initialize_network(int neurons_per_layer[],const int8_t weights_fc1[INPUT_SIZE][HIDDEN_LAYER_1],
    const int8_t weights_fc2[HIDDEN_LAYER_1][NUM_CLASSES],const int8_t *bias_fc1, const int8_t *bias_fc2);
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
#endif // NETWORK_H
