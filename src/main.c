#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LAYERS 10
#define MAX_NEURONS 100
#define TAU 10
#define VOLTAGE_THRESH 1.0
#define DECAY_RATE 0.9

// Define the weight matrices in a struct
typedef struct {
    float weights[MAX_LAYERS][MAX_NEURONS][MAX_NEURONS];
    int num_layers;
    int neurons_per_layer[MAX_LAYERS];
} Network;

Network network;

// Define the global ping pong buffers as bit masks using chars
unsigned char ping_pong_buffer_1[(MAX_NEURONS + 7) / 8];
unsigned char ping_pong_buffer_2[(MAX_NEURONS + 7) / 8];

// Function to initialize weights (for demonstration purposes)
void initialize_weights() {
    for (int l = 0; l < network.num_layers; l++) {
        for (int i = 0; i < network.neurons_per_layer[l]; i++) {
            for (int j = 0; j < (l == 0 ? network.neurons_per_layer[l] : network.neurons_per_layer[l - 1]); j++) {
                // network.weights[l][i][j] = (float)rand() / RAND_MAX;
                network.weights[l][i][j] = 1;
            }
        }
    }
}

// Function to set a bit in the buffer
void set_bit(unsigned char buffer[], int index, int value) {
    int byte_index = index / 8;
    int bit_index = index % 8;
    if (value) {
        buffer[byte_index] |= (1 << bit_index);
    } else {
        buffer[byte_index] &= ~(1 << bit_index);
    }
}

// Function to get a bit from the buffer
int get_bit(unsigned char buffer[], int index) {
    int byte_index = index / 8;
    int bit_index = index % 8;
    return (buffer[byte_index] >> bit_index) & 1;
}

// Function to simulate neuron firing in a layer
void simulate_layer(unsigned char input[], unsigned char output[], float weights[][MAX_NEURONS], int num_neurons, int input_size) {
    for (int i = 0; i < num_neurons; i++) {
        float sum = 0;
        for (int j = 0; j < input_size; j++) {
            sum += get_bit(input, j) * weights[i][j];
        }
        set_bit(output, i, sum > VOLTAGE_THRESH); // Assuming a threshold of 0.5 for neuron firing
    }
}

// Function to update the entire layer based on the buffer
void update_layer(unsigned char input[], unsigned char output[], float weights[][MAX_NEURONS], int num_neurons, int input_size) {
    static float membrane_potential[MAX_NEURONS] = {0};

    for (int i = 0; i < num_neurons; i++) {
        float sum = 0;
        for (int j = 0; j < input_size; j++) {
            sum += get_bit(input, j) * weights[i][j];
        }
        membrane_potential[i] *= DECAY_RATE; // Apply decay
        if (sum > VOLTAGE_THRESH) {
            membrane_potential[i] += 1.0; // Increase potential if neuron fired
        }
        if (membrane_potential[i] >= VOLTAGE_THRESH) {
            set_bit(output, i, 1); // Neuron fires
            membrane_potential[i] = 0; // Reset potential
        } else {
            set_bit(output, i, 0); // Neuron does not fire
        }
    }
}

// Function to initialize input spikes for the first layer
void initialize_input_spikes(unsigned char input[], int num_neurons) {
    for (int i = 0; i < num_neurons; i++) {
        set_bit(input, i, rand() % 2); // Randomly set spikes (0 or 1)
    }
}

int main() {
    // Example initialization
    network.num_layers = 3;
    network.neurons_per_layer[0] = 10;
    network.neurons_per_layer[1] = 10;
    network.neurons_per_layer[2] = 10;

    initialize_weights();

    // Initialize input to the first layer
    unsigned char *input = ping_pong_buffer_1;
    unsigned char *output = ping_pong_buffer_2;

    // Initialize input spikes for the first layer
    initialize_input_spikes(input, network.neurons_per_layer[0]);

    // Process each layer
    for (int l = 0; l < network.num_layers; l++) {
        int num_neurons = network.neurons_per_layer[l];
        int input_size = (l == 0) ? num_neurons : network.neurons_per_layer[l - 1];

        // Simulate tau time steps for the current layer
        for (int t = 0; t < TAU; t++) {
            update_layer(input, output, network.weights[l], num_neurons, input_size);
            // Swap the buffers
            unsigned char *temp = input;
            input = output;
            output = temp;
        }
    }

    // Print the output of the last layer
    for (int i = 0; i < network.neurons_per_layer[network.num_layers - 1]; i++) {
        printf("Neuron %d output: %d\n", i, get_bit(input, i));
    }

    return 0;
}