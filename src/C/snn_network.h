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

// Static memory for network structure



void set_bit(uint8_t buffer[MAX_NEURONS][BITMASK_BYTES], int neuron_idx, int t, int value);
int get_bit(const uint8_t buffer[MAX_NEURONS][BITMASK_BYTES], int neuron_idx, int t);

#if (Q07_FLAG)
    int heaviside(int32_t x, int16_t threshold);
#else
    int heaviside(float x, float threshold);
#endif

void initialize_network(int neurons_per_layer[],const int8_t weights_fc1[HIDDEN_LAYER_1][INPUT_SIZE],
    const int8_t weights_fc2[NUM_CLASSES][HIDDEN_LAYER_1],const int8_t *bias_fc1, const int8_t *bias_fc2);
void zero_network();
void free_network();

void update_layer(const uint8_t input[MAX_NEURONS][BITMASK_BYTES],
                  uint8_t output[MAX_NEURONS][BITMASK_BYTES],
                  Layer *layer, int input_size);

int inference(const uint8_t input[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES], int sample_idx);
int classify_inference(int **firing_counts, int num_neurons, int num_chunks);

int get_input_spike(const uint8_t buffer[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES],
                    int sample, int t, int neuron_idx);
void set_input_spike(uint8_t buffer[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES],
                     int sample, int t, int neuron_idx, int value);

int8_t quantize_q07(float x); 
float dequantize_q07(int32_t q);

#endif // NETWORK_H
