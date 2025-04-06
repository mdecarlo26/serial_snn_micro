#ifndef NETWORK_H
#define NETWORK_H

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
} Network;

// Static memory for network structure
static Layer static_layers[MAX_LAYERS];
static Neuron static_neurons[MAX_LAYERS][MAX_NEURONS];
static float* static_weights[MAX_LAYERS][MAX_NEURONS];
static float  static_weight_data[MAX_LAYERS][MAX_NEURONS][MAX_NEURONS];
static float  static_bias[MAX_LAYERS][MAX_NEURONS];


void set_bit(char **buffer, int x, int y, int value);
int get_bit(const char **buffer, int x, int y);
int heaviside(float x, int threshold);

void initialize_network(int neurons_per_layer[], float **weights_fc1, float **weights_fc2, float *bias_fc1, float *bias_fc2);
void zero_network();
void free_network();

void update_layer(const char **input, char **output, Layer *layer, int input_size);
#endif // NETWORK_H
