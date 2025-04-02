#include "network.h"
#include "define.h"
#include "debug.h"
#include <stdlib.h>

extern Network network;
// Function to set a value in the buffer
void set_bit(char **buffer, int x, int y, int value) {
    buffer[x][y] = value;
}

// Function to get a value from the buffer
int get_bit(const char **buffer, int x, int y) {
    return buffer[x][y];
}

int heaviside(float x, int threshold) {
    return (x >= threshold) ? 1 : 0;
}

// Function to update the entire layer based on the buffer and bias
void update_layer(const char **input, char **output, Layer *layer, int input_size) {
    for (int t = 0; t < TAU; t++) {
            // printf("Time step %d\n", t);
        for (int i = 0; i < layer->num_neurons; i++) {

            float sum = 0.0f;
            if (layer->layer_num > 0) {
                sum += layer->bias[i];
            }

            for (int j = 0; j < input_size; j++) {
                if (get_bit(input, j, t)) { // if incoming spike is present
                    if (layer->layer_num == 0) {
                        sum += 1.0f; // For the input layer, each spike contributes a value of 1
                    } else {
                        sum += layer->weights[i][j];
                    }
                }
            }
            printf("Neuron %d: Old Membrane Potential = %f\n", i, layer->neurons[i].membrane_potential);

            float new_mem = 0;
            int reset_signal = heaviside(layer->neurons[i].membrane_potential,0);
            new_mem = layer->neurons[i].decay_rate * layer->neurons[i].membrane_potential + sum - reset_signal * layer->neurons[i].voltage_thresh;
            layer->neurons[i].membrane_potential = new_mem;
            int output_spike = heaviside(layer->neurons[i].membrane_potential, layer->neurons[i].voltage_thresh);
            set_bit(output, i, t, output_spike); // Reset output for this time step
            printf("Neuron %d: Membrane Potential = %f, Output = %d\n", i, layer->neurons[i].membrane_potential, output_spike);
        }
    }
}


// Function to initialize weights, biases, and neuron properties
void initialize_network(int neurons_per_layer[], float **weights_fc1, float **weights_fc2, float *bias_fc1, float *bias_fc2) {
    network.layers = (Layer *)malloc(network.num_layers * sizeof(Layer));
    for (int l = 0; l < network.num_layers; l++) {
        printf("Initializing Layer %d\n", l);
        network.layers[l].layer_num = l;
        network.layers[l].num_neurons = neurons_per_layer[l];
        network.layers[l].neurons = (Neuron *)malloc(network.layers[l].num_neurons * sizeof(Neuron));
        network.layers[l].weights = (float **)malloc(network.layers[l].num_neurons * sizeof(float *));
        network.layers[l].bias = (float *)malloc(network.layers[l].num_neurons * sizeof(float));
        for (int i = 0; i < network.layers[l].num_neurons; i++) {
            network.layers[l].neurons[i].voltage_thresh = VOLTAGE_THRESH;
            network.layers[l].neurons[i].decay_rate = DECAY_RATE;
            if (l > 0) { // Allocate weights and biases for layers after the input layer
                network.layers[l].weights[i] = (float *)malloc(network.layers[l - 1].num_neurons * sizeof(float));
                if (l == 1) {
                    memcpy(network.layers[l].weights[i], weights_fc1[i], network.layers[l - 1].num_neurons * sizeof(float));
                    network.layers[l].bias[i] = bias_fc1[i];
                } else if (l == 2) {
                    memcpy(network.layers[l].weights[i], weights_fc2[i], network.layers[l - 1].num_neurons * sizeof(float));
                    network.layers[l].bias[i] = bias_fc2[i];
                }
            } else {
                network.layers[l].weights[i] = NULL; // No weights for the input layer
                network.layers[l].bias[i] = 0;         // No bias for the input layer
            }
        }
    }
}


void zero_network() {
    for (int l = 0; l < network.num_layers; l++) {
        for (int i = 0; i < network.layers[l].num_neurons; i++) {
            network.layers[l].neurons[i].membrane_potential = 0;
            network.layers[l].neurons[i].delayed_reset = 0;
        }
    }
}