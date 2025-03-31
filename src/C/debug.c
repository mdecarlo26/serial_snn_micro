#include "debug.h"
#include "define.h"

extern Network network; // Declare the external network variable

void print_weights(float **weights, float *bias, int rows, int cols) {
    printf("Weights and Biases:\n");
    for (int i = 0; i < rows; i++) {
        printf("Neuron %d weights: ", i);
        for (int j = 0; j < cols; j++) {
            printf("%f ", weights[i][j]);
        }
        printf(" | Bias: %f\n", bias[i]);
    }
}

// Function to print the model overview
void print_model_overview() {
    printf("\033[1;32mModel Overview:\033[0m\n");
    for (int l = 0; l < network.num_layers; l++) {
        printf("Layer %d: %d neurons\n", l, network.layers[l].num_neurons);
        if (l > 0) { // Only print weights and biases for layers after the first layer
            // print_weights(network.layers[l].weights, network.layers[l].bias, network.layers[l].num_neurons, network.layers[l - 1].num_neurons);
        }
    }
}

// Function to print the states of neurons in a layer
void print_neuron_states(Layer *layer) {
    for (int i = 0; i < layer->num_neurons; i++) {
        printf("  Neuron %d: Membrane Potential = %f\n", i, layer->neurons[i].membrane_potential);
    }
}

// Function to print the spike buffer
void print_spike_buffer(const char **buffer, int size) {
    for (int i = 0; i < size; i++) {
        for (int t = 0; t < TAU; t++) {
            printf("%d ", get_bit(buffer, i, t));
        }
        printf("\n");
    }
}

// Function to print the ping pong buffers
void print_ping_pong_buffers(const char **buffer1, const char **buffer2, int size) {
    printf("\033[1;33mPing:\033[0m\n");
    print_spike_buffer(buffer1, size);
    printf("\033[1;33mPong:\033[0m\n");
    print_spike_buffer(buffer2, size);
}


void print_firing_counts(int **firing_counts, int num_neurons, int num_chunks) {
    for (int i = 0; i < num_neurons; i++) {
        printf("Neuron %d: ", i);
        for (int j = 0; j < num_chunks; j++) {
            printf("%d ", firing_counts[i][j]);
        }
        printf("\n");
    }
}