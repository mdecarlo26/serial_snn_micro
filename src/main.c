#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LAYERS 10
#define MAX_NEURONS 100
#define TAU 10
#define VOLTAGE_THRESH 1.0
#define DECAY_RATE 0.9

typedef struct {
    float membrane_potential;
    float voltage_thresh;
    float decay_rate;
} Neuron;

typedef struct {
    Neuron *neurons;
    float **weights;
    int num_neurons;
} Layer;

typedef struct {
    Layer *layers;
    int num_layers;
} Network;

Network network;

// Define the global ping pong buffers as bit masks using chars
unsigned char ping_pong_buffer_1[(MAX_NEURONS + 7) / 8];
unsigned char ping_pong_buffer_2[(MAX_NEURONS + 7) / 8];

// Function prototypes
void initialize_network(int neurons_per_layer[]);
void free_network();
void set_bit(unsigned char buffer[], int index, int value);
int get_bit(const unsigned char buffer[], int index);
void simulate_layer(const unsigned char input[], unsigned char output[], float weights[][MAX_NEURONS], int num_neurons, int input_size);
void update_layer(const unsigned char input[], unsigned char output[], Layer *layer, int input_size);
void initialize_input_spikes(unsigned char input[], int num_neurons);

// Function to initialize weights and neuron properties
void initialize_network(int neurons_per_layer[]) {
    network.layers = (Layer *)malloc(network.num_layers * sizeof(Layer));
    for (int l = 0; l < network.num_layers; l++) {
        network.layers[l].num_neurons = neurons_per_layer[l];
        network.layers[l].neurons = (Neuron *)malloc(network.layers[l].num_neurons * sizeof(Neuron));
        network.layers[l].weights = (float **)malloc(network.layers[l].num_neurons * sizeof(float *));
        for (int i = 0; i < network.layers[l].num_neurons; i++) {
            network.layers[l].neurons[i].membrane_potential = 0;
            network.layers[l].neurons[i].voltage_thresh = VOLTAGE_THRESH;
            network.layers[l].neurons[i].decay_rate = DECAY_RATE;
            network.layers[l].weights[i] = (float *)malloc((l == 0 ? network.layers[l].num_neurons : network.layers[l - 1].num_neurons) * sizeof(float));
            for (int j = 0; j < (l == 0 ? network.layers[l].num_neurons : network.layers[l - 1].num_neurons); j++) {
                network.layers[l].weights[i][j] = 1;
            }
        }
    }
}

// Function to free the allocated memory
void free_network() {
    for (int l = 0; l < network.num_layers; l++) {
        for (int i = 0; i < network.layers[l].num_neurons; i++) {
            free(network.layers[l].weights[i]);
        }
        free(network.layers[l].weights);
        free(network.layers[l].neurons);
    }
    free(network.layers);
}

// Function to set a bit in the buffer
void set_bit(unsigned char buffer[], int index, int value) {
    unsigned char mask = 1 << (index & 7);
    if (value) {
        buffer[index >> 3] |= mask;
    } else {
        buffer[index >> 3] &= ~mask;
    }
}

// Function to get a bit from the buffer
int get_bit(const unsigned char buffer[], int index) {
    unsigned char mask = 1 << (index & 7);
    return (buffer[index >> 3] & mask) != 0;
}

// Function to simulate neuron firing in a layer
void simulate_layer(const unsigned char input[], unsigned char output[], float weights[][MAX_NEURONS], int num_neurons, int input_size) {
    for (int i = 0; i < num_neurons; i++) {
        float sum = 0;
        for (int j = 0; j < input_size; j++) {
            sum += get_bit(input, j) * weights[i][j];
        }
        set_bit(output, i, sum > VOLTAGE_THRESH); // Assuming a threshold of 0.5 for neuron firing
    }
}

// Function to update the entire layer based on the buffer
void update_layer(const unsigned char input[], unsigned char output[], Layer *layer, int input_size) {
    for (int i = 0; i < layer->num_neurons; i++) {
        float sum = 0;
        int any_fired = 0;
        for (int j = 0; j < input_size; j++) {
            if (get_bit(input, j)) {
                any_fired = 1;
                sum += layer->weights[i][j];
            }
        }
        if (!any_fired) {
            layer->neurons[i].membrane_potential *= layer->neurons[i].decay_rate; // Apply decay only
            set_bit(output, i, 0); // Neuron does not fire
            continue;
        }
        layer->neurons[i].membrane_potential *= layer->neurons[i].decay_rate; // Apply decay
        if (sum > layer->neurons[i].voltage_thresh) {
            layer->neurons[i].membrane_potential += 1.0; // Increase potential if neuron fired
        }
        if (layer->neurons[i].membrane_potential >= layer->neurons[i].voltage_thresh) {
            set_bit(output, i, 1); // Neuron fires
            layer->neurons[i].membrane_potential = 0; // Reset potential
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
    int neurons_per_layer[] = {10, 10, 10};

    initialize_network(neurons_per_layer);

    // Initialize input to the first layer
    unsigned char *input = ping_pong_buffer_1;
    unsigned char *output = ping_pong_buffer_2;

    // Initialize input spikes for the first layer
    initialize_input_spikes(input, network.layers[0].num_neurons);

    // Process each layer
    for (int l = 0; l < network.num_layers; l++) {
        int num_neurons = network.layers[l].num_neurons;
        int input_size = (l == 0) ? num_neurons : network.layers[l - 1].num_neurons;

        // Simulate tau time steps for the current layer
        for (int t = 0; t < TAU; t++) {
            update_layer(input, output, &network.layers[l], input_size);
            // Swap the buffers
            unsigned char *temp = input;
            input = output;
            output = temp;
        }
    }

    // Print the output of the last layer
    for (int i = 0; i < network.layers[network.num_layers - 1].num_neurons; i++) {
        printf("Neuron %d output: %d\n", i, get_bit(input, i));
    }

    // Free the allocated memory
    free_network();

    return 0;
}