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
#define TIME_WINDOW 10  // Length of the time window in ms
#define MAX_RATE 10    // Maximum spike rate (spikes per time window)
#define NUM_SAMPLES 200  // Number of data samples

typedef struct {
    float membrane_potential;
    float voltage_thresh;
    float decay_rate;
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

Network network;

// Define the global ping pong buffers as pointers to 2D arrays
char **ping_pong_buffer_1;
char **ping_pong_buffer_2;

// Function prototypes
void initialize_network(int neurons_per_layer[], float **weights_fc1, float **weights_fc2, float *bias_fc1, float *bias_fc2);
void free_network();
void set_bit(char **buffer, int x, int y, int value);
int get_bit(const char **buffer, int x, int y);
void update_layer(const char **input, char **output, Layer *layer, int input_size);
void initialize_input_spikes(char **input, int num_neurons);
void classify_spike_trains(int **firing_counts, int num_neurons, FILE *output_file, int sample_index, int num_chunks);
void print_weights(float **weights, float *bias, int rows, int cols);
void print_model_overview();
void print_neuron_states(Layer *layer);
void print_spike_buffer(const char **buffer, int size);
void print_ping_pong_buffers(const char **buffer1, const char **buffer2, int size);

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

    // Load biases from files
    float *bias_fc1 = (float *)malloc(10 * sizeof(float));
    float *bias_fc2 = (float *)malloc(2 * sizeof(float));
    load_bias("bias_fc1.txt", bias_fc1, 10);
    load_bias("bias_fc2.txt", bias_fc2, 2);
    printf("Biases loaded\n");

    initialize_network(neurons_per_layer, weights_fc1, weights_fc2, bias_fc1, bias_fc2);
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

    // Load initial spikes from CSV file
    float **initial_spikes = (float **)malloc(NUM_SAMPLES * sizeof(float *));
    for (int i = 0; i < NUM_SAMPLES; i++) {
        initial_spikes[i] = (float *)malloc(TIME_WINDOW * sizeof(float));
    }
    load_csv("spikes.csv", initial_spikes, NUM_SAMPLES, TIME_WINDOW);
    printf("Initial spikes loaded\n");

    // Process each data point
    FILE *output_file = fopen("model_output.txt", "w");
    if (output_file == NULL) {
        perror("Failed to open output file");
        exit(EXIT_FAILURE);
    }

    printf("Starting Sim\n");
    int num_chunks = TIME_WINDOW / TAU;
    for (int d = 110; d < NUM_SAMPLES; d++) {
        int **firing_counts = (int **)malloc(network.layers[network.num_layers - 1].num_neurons * sizeof(int *));
        for (int i = 0; i < network.layers[network.num_layers - 1].num_neurons; i++) {
            firing_counts[i] = (int *)calloc(num_chunks, sizeof(int));
        }

        // Process each chunk of TAU time steps
        for (int chunk = 0; chunk < TIME_WINDOW; chunk += TAU) {
            int chunk_index = chunk / TAU;

            // Initialize input spikes for the first layer using initial spikes
            for (int t = 0; t < TAU; t++) {
                for (int i = 0; i < network.layers[0].num_neurons; i++) {
                    set_bit(ping_pong_buffer_1, i, t, initial_spikes[d][chunk + t]);
                }
            }
            // print_ping_pong_buffers((const char **)ping_pong_buffer_1, (const char **)ping_pong_buffer_2, network.layers[0].num_neurons);

            // Print input spikes
            printf("Input spikes at chunk %d:\n", chunk);
            print_spike_buffer((const char **)ping_pong_buffer_1, network.layers[0].num_neurons);

            // Process each layer
            for (int l = 0; l < network.num_layers; l++) {
                int num_neurons = network.layers[l].num_neurons;
                int input_size = (l == 0) ? network.layers[l].num_neurons : network.layers[l - 1].num_neurons;

                // Simulate tau time steps for the current layer
                printf("Simulating Layer %d\n", l);
                update_layer((const char **)ping_pong_buffer_1, ping_pong_buffer_2, &network.layers[l], input_size);
                // Swap the buffers
                char **temp = ping_pong_buffer_1;
                ping_pong_buffer_1 = ping_pong_buffer_2;
                ping_pong_buffer_2 = temp;

                // Print neuron states after processing each layer
                // printf("Neuron states in layer %d after processing:\n", l);
                // print_neuron_states(&network.layers[l]);

                // Print ping pong buffers
                // print_ping_pong_buffers((const char **)ping_pong_buffer_1, (const char **)ping_pong_buffer_2, network.layers[l].num_neurons);
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
        printf("Ouput spikes at sample %d:\n", d);
        print_ping_pong_buffers((const char **)ping_pong_buffer_1, (const char **)ping_pong_buffer_2, network.layers[network.num_layers-1].num_neurons);
        // Classify the spike train for the current data sample
        classify_spike_trains(firing_counts, network.layers[network.num_layers - 1].num_neurons, output_file, d, num_chunks);
        break;

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
        free(initial_spikes[i]);
    }
    free(initial_spikes);

    for (int i = 0; i < MAX_NEURONS; i++) {
        free(ping_pong_buffer_1[i]);
        free(ping_pong_buffer_2[i]);
    }
    free(ping_pong_buffer_1);
    free(ping_pong_buffer_2);
    free(bias_fc1);
    free(bias_fc2);

    return 0;
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
            network.layers[l].neurons[i].membrane_potential = 0;
            network.layers[l].neurons[i].voltage_thresh = VOLTAGE_THRESH;
            network.layers[l].neurons[i].decay_rate = DECAY_RATE;
            if (l > 0) { // Only allocate weights and biases for layers after the first layer
                network.layers[l].weights[i] = (float *)malloc(network.layers[l - 1].num_neurons * sizeof(float));
                if (l == 1) {
                    memcpy(network.layers[l].weights[i], weights_fc1[i], network.layers[l - 1].num_neurons * sizeof(float));
                    network.layers[l].bias[i] = bias_fc1[i];
                } else if (l == 2) {
                    memcpy(network.layers[l].weights[i], weights_fc2[i], network.layers[l - 1].num_neurons * sizeof(float));
                    network.layers[l].bias[i] = bias_fc2[i];
                }
            } else {
                network.layers[l].weights[i] = NULL; // No weights for the first layer
                network.layers[l].bias[i] = 0; // No bias for the first layer
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
        free(network.layers[l].bias);
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

// Function to update the entire layer based on the buffer and bias
void update_layer(const char **input, char **output, Layer *layer, int input_size) {
    for (int i = 0; i < layer->num_neurons; i++) {
        for (int t = 0; t < TAU; t++) {
            float sum = 0;
            int any_fired = 0;
            for (int j = 0; j < input_size; j++) {
                if (get_bit(input, j, t)) {
                    any_fired = 1;
                    if (layer->layer_num == 0) {
                        sum += 1.0; // Input layer
                    } else {
                        sum += layer->weights[i][j] + layer->bias[i]; // Hidden layer
                    }
                }
            }
            layer->neurons[i].membrane_potential += sum; // Increase potential if neuron fired
            // if (layer->layer_num > 0) {
            //     layer->neurons[i].membrane_potential += layer->bias[i]; // Add bias
            // }
            printf("Neuron %d, Time %d, Sum: %f, Membrane Potential: %.2f\n", i, t, sum, layer->neurons[i].membrane_potential);
            if (!any_fired) {
                layer->neurons[i].membrane_potential *= layer->neurons[i].decay_rate; // Apply decay only
                set_bit(output, i, t, 0); // Neuron does not fire
                continue;
            }
            if (layer->neurons[i].membrane_potential >= layer->neurons[i].voltage_thresh) {
                set_bit(output, i, t, 1); // Neuron fires
                layer->neurons[i].membrane_potential -= VOLTAGE_THRESH; // Reset potential
                printf("Neuron %d fires at Time %d\n", i, t);
            } else {
                set_bit(output, i, t, 0); // Neuron does not fire
            }
            layer->neurons[i].membrane_potential *= layer->neurons[i].decay_rate; // Apply decay
            printf("Updated Membrane Potential for Neuron %d at Time %d: %.2f\n", i, t, layer->neurons[i].membrane_potential);
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

// Function to print the weight matrix and biases
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
    printf("Model Overview:\n");
    for (int l = 0; l < network.num_layers; l++) {
        printf("Layer %d: %d neurons\n", l, network.layers[l].num_neurons);
        if (l > 0) { // Only print weights and biases for layers after the first layer
            print_weights(network.layers[l].weights, network.layers[l].bias, network.layers[l].num_neurons, network.layers[l - 1].num_neurons);
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
    printf("Ping:\n");
    print_spike_buffer(buffer1, size);
    printf("Pong:\n");
    print_spike_buffer(buffer2, size);
}
