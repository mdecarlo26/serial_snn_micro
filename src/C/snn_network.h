#ifndef SNN_NETWORK_H
#define SNN_NETWORK_H

#include "define.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float membrane_potential;
    float voltage_thresh;
    float decay_rate;
    float delayed_reset;
} Neuron;

typedef struct {
    Neuron *neurons;
    float **weights;
    float *bias;
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
int heaviside(float x, int threshold);

void initialize_network(int neurons_per_layer[], float **weights_fc1, float **weights_fc2, float *bias_fc1, float *bias_fc2);
void zero_network();
void free_network();

void update_layer(const uint8_t input[MAX_NEURONS][BITMASK_BYTES],
                  uint8_t output[MAX_NEURONS][BITMASK_BYTES],
                  Layer *layer, int input_size);

int inference(const uint8_t input[NUM_SAMPLES][TIME_WINDOW][INPUT_SIZE],
              uint8_t ping_pong_1[MAX_NEURONS][BITMASK_BYTES],
              uint8_t ping_pong_2[MAX_NEURONS][BITMASK_BYTES],
              int sample_idx);
int classify_inference(int **firing_counts, int num_neurons, int num_chunks);

int get_input_spike(const uint8_t buffer[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES],
                    int sample, int t, int neuron_idx);
void set_input_spike(uint8_t buffer[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES],
                     int sample, int t, int neuron_idx, int value);
#endif // NETWORK_H
