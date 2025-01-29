#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "file_operations.h"
#include "rate_encoding.h"
#include <time.h>

#define MAX_LAYERS 10
#define MAX_NEURONS 100
#define TAU 10
#define VOLTAGE_THRESH 1.0
#define DECAY_RATE 0.9
#define TIME_WINDOW 100  // Length of the time window in ms
#define MAX_RATE 50      // Maximum spike rate (spikes per time window)

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
void initialize_network(int neurons_per_layer[], float **weights_fc1, float **weights_fc2);
void free_network();
void set_bit(unsigned char buffer[], int index, int value);
int get_bit(const unsigned char buffer[], int index);
void simulate_layer(const unsigned char input[], unsigned char output[], float weights[][MAX_NEURONS], int num_neurons, int input_size);
void update_layer(const unsigned char input[], unsigned char output[], Layer *layer, int input_size);
void initialize_input_spikes(unsigned char input[], int num_neurons);
void classify_spike_trains(int *firing_counts, int num_neurons, FILE *output_file, int sample_index);
void print_weights(float **weights, int rows, int cols);
void print_model_overview();

int main() {
    srand(time(NULL));  // Seed the random number generator

    // Example initialization
    network.num_layers = 3;
    int neurons_per_layer[] = {1, 10, 2};

    // Load weights from files
    float **weights_fc1 = (float **)malloc(10 * sizeof(float *));
    float **weights_fc2 = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 10; i++) {
        weights_fc1[i] = (float *)malloc(1 * sizeof(float));
    }
    for (int i = 0; i < 2; i++) {
        weights_fc2[i] = (float *)malloc(10 * sizeof(float));
    }
    load_weights("weights_fc1.txt", weights_fc1, 10, 1);
    load_weights("weights_fc2.txt", weights_fc2, 2, 10);
    printf("Weights loaded\n");
    initialize_network(neurons_per_layer, weights_fc1, weights_fc2);
    printf("Network initialized\n");

    // Print model overview
    print_model_overview();

    // Load data from file
    float data[200];
    load_data("data.txt", data, 200);

    // Allocate memory for spike trains
    unsigned char **spike_trains = (unsigned char **)malloc(200 * sizeof(unsigned char *));
    for (int i = 0; i < 200; i++) {
        spike_trains[i] = (unsigned char *)calloc(TIME_WINDOW, sizeof(unsigned char));
    }

    // Perform rate encoding
    rate_encoding(data, 200, TIME_WINDOW, MAX_RATE, spike_trains);
    printf("Encoding Spikes\n");
    // Print the spike trains
    // print_spike_trains(spike_trains, 200, TIME_WINDOW);

    // Initialize input to the first layer
    unsigned char *input = ping_pong_buffer_1;
    unsigned char *output = ping_pong_buffer_2;

    // Process each data point
    FILE *output_file = fopen("model_output.txt", "w");
    if (output_file == NULL) {
        perror("Failed to open output file");
        exit(EXIT_FAILURE);
    }

    printf("Starting Sim\n");
    for (int d = 0; d < 200; d++) {
        int firing_counts[MAX_NEURONS] = {0};

        // Initialize input spikes for the first layer using spike trains
        for (int t = 0; t < TIME_WINDOW; t++) {
            for (int i = 0; i < network.layers[0].num_neurons; i++) {
                set_bit(input, i, spike_trains[d][t]);
            }

            // Process each layer
            for (int l = 0; l < network.num_layers; l++) {
                int num_neurons = network.layers[l].num_neurons;
                int input_size = (l == 0) ? network.layers[l].num_neurons : network.layers[l - 1].num_neurons;

                // Simulate tau time steps for the current layer
                for (int tau = 0; tau < TAU; tau++) {
                    update_layer(input, output, &network.layers[l], input_size);
                    // Swap the buffers
                    unsigned char *temp = input;
                    input = output;
                    output = temp;
                }
            }

            // Accumulate firing counts for the last layer
            for (int i = 0; i < network.layers[network.num_layers - 1].num_neurons; i++) {
                if (get_bit(input, i)) {
                    firing_counts[i]++;
                }
            }
        }

        // Classify the spike train for the current data sample
        classify_spike_trains(firing_counts, network.layers[network.num_layers - 1].num_neurons, output_file, d);
    }

    fclose(output_file);

    // Free the allocated memory
    for (int i = 0; i < 10; i++) {
        free(weights_fc1[i]);
    }
    for (int i = 0; i < 2; i++) {
        free(weights_fc2[i]);
    }
    free(weights_fc1);
    free(weights_fc2);

    for (int i = 0; i < 200; i++) {
        free(spike_trains[i]);
    }
    free(spike_trains);

    return 0;
}

// Function to initialize weights and neuron properties
void initialize_network(int neurons_per_layer[], float **weights_fc1, float **weights_fc2) {
    network.layers = (Layer *)malloc(network.num_layers * sizeof(Layer));
    for (int l = 0; l < network.num_layers; l++) {
        printf("Initializing Layer %d\n", l);
        network.layers[l].num_neurons = neurons_per_layer[l];
        printf("Num Neurons: %d\n", network.layers[l].num_neurons);
        network.layers[l].neurons = (Neuron *)malloc(network.layers[l].num_neurons * sizeof(Neuron));
        printf("Allocating Weights\n");
        network.layers[l].weights = (float **)malloc(network.layers[l].num_neurons * sizeof(float *));
        for (int i = 0; i < network.layers[l].num_neurons; i++) {
            network.layers[l].neurons[i].membrane_potential = 0;
            network.layers[l].neurons[i].voltage_thresh = VOLTAGE_THRESH;
            network.layers[l].neurons[i].decay_rate = DECAY_RATE;
            printf("Allocating Weights for Neuron %d\n", i);
            printf("Num Neurons in Previous Layer: %d\n", l == 0 ? 1 : network.layers[l - 1].num_neurons);
            network.layers[l].weights[i] = (float *)malloc((l == 0 ? 1 : network.layers[l - 1].num_neurons) * sizeof(float));
            printf("Copying Weights\n");
            if (l == 0) {
                print_weights(weights_fc1, 10, 1);
                memcpy(network.layers[l].weights[i], weights_fc1[i], 1 * sizeof(float));
            } else if (l == 1) {
                print_weights(weights_fc2, 2, 10);
                memcpy(network.layers[l].weights[i], weights_fc2[i], 10 * sizeof(float));
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

// Function to classify spike trains based on the firing frequency of the last layer
void classify_spike_trains(int *firing_counts, int num_neurons, FILE *output_file, int sample_index) {
    // Determine the classification based on the neuron with the highest firing frequency
    int max_firing_count = 0;
    int classification = -1;
    for (int i = 0; i < num_neurons; i++) {
        if (firing_counts[i] > max_firing_count) {
            max_firing_count = firing_counts[i];
            classification = i;
        }
    }

    // Output the classification and the firing count to the file
    fprintf(output_file, "Sample %d: Classification = %d, Firing Count = %d\n", sample_index, classification, max_firing_count);
}

// Function to print the weight matrix
void print_weights(float **weights, int rows, int cols) {
    printf("Weights:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", weights[i][j]);
        }
        printf("\n");
    }
}

// Function to print the model overview
void print_model_overview() {
    printf("Model Overview:\n");
    for (int l = 0; l < network.num_layers; l++) {
        printf("Layer %d: %d neurons\n", l, network.layers[l].num_neurons);
        for (int i = 0; i < network.layers[l].num_neurons; i++) {
            printf("  Neuron %d weights: ", i);
            // for (int j = 0; j < (l == 0 ? 1 : network.layers[l - 1].num_neurons); j++) {
            //     printf("%f ", network.layers[l].weights[i][j]);
            // }
            printf("\n");
        }
    }
}
