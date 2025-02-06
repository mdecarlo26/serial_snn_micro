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
#define DECAY_RATE 0.8
#define TIME_WINDOW 20  // Length of the time window in ms
#define MAX_RATE 10      // Maximum spike rate (spikes per time window)
#define NUM_SAMPLES 200  // Number of data samples
// #define DEBUG 1

typedef struct {
    float membrane_potential;
    float voltage_thresh;
    float decay_rate;
} Neuron;

typedef struct {
    Neuron *neurons;
    float **weights;
    int num_neurons;
    int layer_num;
} Layer;

typedef struct {
    Layer *layers;
    int num_layers;
} Network;

Network network;

// Define the global ping pong buffers as pointers to 2D arrays
char **ping_pong_buffer_1;
char **ping_pong_buffer_2;

// Function prototypes
void initialize_network(int neurons_per_layer[], float **weights_fc1, float **weights_fc2);
void free_network();
void set_bit(char **buffer, int x, int y, int value);
int get_bit(const char **buffer, int x, int y);
void update_layer(const char **input, char **output, Layer *layer, int input_size);
void initialize_input_spikes(char **input, int num_neurons);
void classify_spike_trains(int **firing_counts, int num_neurons, FILE *output_file, int sample_index, int num_chunks);
void print_weights(float **weights, int rows, int cols);
void print_model_overview();
void print_neuron_states(Layer *layer);
void print_spike_buffer(const char **buffer, int size);
#ifdef DEBUG
void print_ping_pong_buffers(const char **buffer1, const char **buffer2, int size);
#endif

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

    // Allocate memory for ping pong buffers
    ping_pong_buffer_1 = (char **)malloc(MAX_NEURONS * sizeof(char *));
    ping_pong_buffer_2 = (char **)malloc(MAX_NEURONS * sizeof(char *));
    for (int i = 0; i < MAX_NEURONS; i++) {
        ping_pong_buffer_1[i] = (char *)calloc(TAU, sizeof(char));
        ping_pong_buffer_2[i] = (char *)calloc(TAU, sizeof(char));
    }

    // Print model overview
    print_model_overview();

    // Load data from file
    float data[NUM_SAMPLES];
    load_data("data.txt", data, NUM_SAMPLES);

    // Allocate memory for spike trains
    unsigned char **spike_trains = (unsigned char **)malloc(NUM_SAMPLES * sizeof(unsigned char *));
    for (int i = 0; i < NUM_SAMPLES; i++) {
        spike_trains[i] = (unsigned char *)calloc(TIME_WINDOW, sizeof(unsigned char));
    }

    // Perform rate encoding
    rate_encoding(data, NUM_SAMPLES, TIME_WINDOW, MAX_RATE, spike_trains);
    printf("Encoding Spikes\n");
    // Print the spike trains
    // print_spike_trains(spike_trains, 10, TIME_WINDOW);

    // Process each data point
    FILE *output_file = fopen("model_output.txt", "w");
    if (output_file == NULL) {
        perror("Failed to open output file");
        exit(EXIT_FAILURE);
    }

    printf("Starting Sim\n");
    int num_chunks = TIME_WINDOW / TAU;
    for (int d = 1; d < NUM_SAMPLES; d++) {
        int **firing_counts = (int **)malloc(network.layers[network.num_layers - 1].num_neurons * sizeof(int *));
        for (int i = 0; i < network.layers[network.num_layers - 1].num_neurons; i++) {
            firing_counts[i] = (int *)calloc(num_chunks, sizeof(int));
        }

        // Process each chunk of TAU time steps
        for (int chunk = 0; chunk < TIME_WINDOW; chunk += TAU) {
            int chunk_index = chunk / TAU;

            // Initialize input spikes for the first layer using spike trains
            for (int t = 0; t < TAU; t++) {
                for (int i = 0; i < network.layers[0].num_neurons; i++) {
                    set_bit(ping_pong_buffer_1, i, t, spike_trains[d][chunk + t]);
                }
            }
            #ifdef DEBUG
            print_ping_pong_buffers((const char **)ping_pong_buffer_1, (const char **)ping_pong_buffer_2, network.layers[0].num_neurons);
            #endif

            // Process each layer
            for (int l = 0; l < network.num_layers; l++) {
                int num_neurons = network.layers[l].num_neurons;
                int input_size = (l == 0) ? network.layers[l].num_neurons : network.layers[l - 1].num_neurons;

                // Simulate tau time steps for the current layer
                update_layer((const char **)ping_pong_buffer_1, ping_pong_buffer_2, &network.layers[l], input_size);
                // Swap the buffers
                char **temp = ping_pong_buffer_1;
                ping_pong_buffer_1 = ping_pong_buffer_2;
                ping_pong_buffer_2 = temp;

                #ifdef DEBUG
                print_ping_pong_buffers((const char **)ping_pong_buffer_1, (const char **)ping_pong_buffer_2, network.layers[l].num_neurons);
                #endif
            }

            // Accumulate firing counts for the last layer
            for (int i = 0; i < network.layers[network.num_layers - 1].num_neurons; i++) {
                for (int t = 0; t < TAU; t++) {
                    if (get_bit((const char **)ping_pong_buffer_1, i, t)) {
                        firing_counts[i][chunk_index]++;
                    }
                }
            }
        }

        // Classify the spike train for the current data sample
        classify_spike_trains(firing_counts, network.layers[network.num_layers - 1].num_neurons, output_file, d, num_chunks);

        // Free firing counts memory
        for (int i = 0; i < network.layers[network.num_layers - 1].num_neurons; i++) {
            free(firing_counts[i]);
        }
        free(firing_counts);
    }
    printf("Sim Finished\n");
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

    for (int i = 0; i < NUM_SAMPLES; i++) {
        free(spike_trains[i]);
    }
    free(spike_trains);

    for (int i = 0; i < MAX_NEURONS; i++) {
        free(ping_pong_buffer_1[i]);
        free(ping_pong_buffer_2[i]);
    }
    free(ping_pong_buffer_1);
    free(ping_pong_buffer_2);

    return 0;
}

// Function to initialize weights and neuron properties
void initialize_network(int neurons_per_layer[], float **weights_fc1, float **weights_fc2) {
    network.layers = (Layer *)malloc(network.num_layers * sizeof(Layer));
    for (int l = 0; l < network.num_layers; l++) {
        printf("Initializing Layer %d\n", l);
        network.layers[l].layer_num = l;
        network.layers[l].num_neurons = neurons_per_layer[l];
        network.layers[l].neurons = (Neuron *)malloc(network.layers[l].num_neurons * sizeof(Neuron));
        network.layers[l].weights = (float **)malloc(network.layers[l].num_neurons * sizeof(float *));
        for (int i = 0; i < network.layers[l].num_neurons; i++) {
            network.layers[l].neurons[i].membrane_potential = 0;
            network.layers[l].neurons[i].voltage_thresh = VOLTAGE_THRESH;
            network.layers[l].neurons[i].decay_rate = DECAY_RATE;
            if (l > 0) { // Only allocate weights for layers after the first layer
                network.layers[l].weights[i] = (float *)malloc(network.layers[l - 1].num_neurons * sizeof(float));
                // printf("Allocating Weights for Neuron %d in Layer %d\n", i, l);
                if (l == 1) {
                    memcpy(network.layers[l].weights[i], weights_fc1[i], network.layers[l - 1].num_neurons * sizeof(float));
                } else if (l == 2) {
                    memcpy(network.layers[l].weights[i], weights_fc2[i], network.layers[l - 1].num_neurons * sizeof(float));
                }
            } else {
                network.layers[l].weights[i] = NULL; // No weights for the first layer
            }
        }
    }
}

// Function to free the allocated memory
void free_network() {
    for (int l = 0; l < network.num_layers; l++) {
        for (int i = 0; i < network.layers[l].num_neurons; i++) {
            if (network.layers[l].weights[i] != NULL) {
                free(network.layers[l].weights[i]);
            }
        }
        free(network.layers[l].weights);
        free(network.layers[l].neurons);
    }
    free(network.layers);
}

// Function to set a value in the buffer
void set_bit(char **buffer, int x, int y, int value) {
    buffer[x][y] = value;
}

// Function to get a value from the buffer
int get_bit(const char **buffer, int x, int y) {
    return buffer[x][y];
}

// Function to update the entire layer based on the buffer
void update_layer(const char **input, char **output, Layer *layer, int input_size) {
    for (int i = 0; i < layer->num_neurons; i++) {
        if (layer->neurons[i].membrane_potential)
            layer->neurons[i].membrane_potential *= layer->neurons[i].decay_rate; // Apply decay
        for (int t = 0; t < TAU; t++) {
            float sum = 0;
            int any_fired = 0;
            for (int j = 0; j < input_size; j++) {
                if (get_bit(input, j, t)) {
                    any_fired = 1;
                    if (layer->layer_num == 0) {
                        sum += 1.0; // Input layer
                    } else {
                        sum += layer->weights[i][j]; // Hidden layer
                    }
                }
            }
            layer->neurons[i].membrane_potential += sum; // Increase potential if neuron fired
            if (layer->neurons[i].membrane_potential < 0) {
                layer->neurons[i].membrane_potential = 0; // Floor potential to 0
            }
            #ifdef DEBUG
            printf("Neuron %d, Time %d, Sum: %f, Membrane Potential: %.2f\n", i, t, sum, layer->neurons[i].membrane_potential);
            #endif
            if (!any_fired) {
                set_bit(output, i, t, 0); // Neuron does not fire
                continue;
            }
            if (layer->neurons[i].membrane_potential >= layer->neurons[i].voltage_thresh) {
                set_bit(output, i, t, 1); // Neuron fires
                layer->neurons[i].membrane_potential = 0; // Reset potential
                #ifdef DEBUG
                printf("Neuron %d fires at Time %d\n", i, t);
                #endif
            } else {
                set_bit(output, i, t, 0); // Neuron does not fire
            }
            if (layer->neurons[i].membrane_potential < 0) {
                layer->neurons[i].membrane_potential = 0; // Floor potential to 0
            }
            #ifdef DEBUG
            printf("Updated Membrane Potential for Neuron %d at Time %d: %.2f\n", i, t, layer->neurons[i].membrane_potential);
            #endif
        }
    }
}

// Function to classify spike trains based on the firing frequency of the last layer
void classify_spike_trains(int **firing_counts, int num_neurons, FILE *output_file, int sample_index, int num_chunks) {
    // Determine the classification based on the neuron with the highest firing frequency
    int max_firing_count = 0;
    int classification = -1;
    for (int i = 0; i < num_neurons; i++) {
        int total_firing_count = 0;
        for (int j = 0; j < num_chunks; j++) {
            total_firing_count += firing_counts[i][j];
        }
        if (total_firing_count > max_firing_count) {
            max_firing_count = total_firing_count;
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
        if (l > 0) { // Only print weights for layers after the first layer
            for (int i = 0; i < network.layers[l].num_neurons; i++) {
                printf("  Neuron %d weights: ", i);
                for (int j = 0; j < network.layers[l - 1].num_neurons; j++) {
                    printf("%f ", network.layers[l].weights[i][j]);
                }
                printf("\n");
            }
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
#ifdef DEBUG
void print_ping_pong_buffers(const char **buffer1, const char **buffer2, int size) {
    printf("Ping:\n");
    print_spike_buffer(buffer1, size);
    printf("Pong:\n");
    print_spike_buffer(buffer2, size);
}
#endif
