#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "file_operations.h"
#include "rate_encoding.h"
#include <time.h>
#include <sys/time.h>

#define MAX_LAYERS 10
#define MAX_NEURONS 1000
#define TAU 10
#define VOLTAGE_THRESH 1.0
#define DECAY_RATE 0.95
#define NUM_SAMPLES 10000  // Total dataset size
#define TIME_WINDOW 20         // Temporal steps in spike train
#define INPUT_SIZE 784       // 28x28 flattened images

// #define TIME_WINDOW 10  // Length of the time window in ms
// #define MAX_RATE 10    // Maximum spike rate (spikes per time window)
// #define NUM_SAMPLES 200  // Number of data samples

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
void classify_spike_trains(int **firing_counts, int num_neurons, FILE *output_file, int sample_index, int num_chunks, char* labels);
int heavyside(float x, int threshold);

// Debug Print Functions
void print_weights(float **weights, float *bias, int rows, int cols);
void print_model_overview();
void print_neuron_states(Layer *layer);
void print_spike_buffer(const char **buffer, int size);
void print_ping_pong_buffers(const char **buffer1, const char **buffer2, int size);
void print_firing_counts(int **firing_counts, int num_neurons, int num_chunks);

// Memory allocation functions
char*** allocate_spike_array();
void free_spike_array(char*** spikes);
int validate_spike_data(char ***spikes);
char *allocate_labels(int num_samples);
void free_labels(char *labels);

int main() {
    srand(time(NULL));  // Seed the random number generator

    // Example initialization
    network.num_layers = 3;
    int l1 = INPUT_SIZE;
    int l2 = 256;
    int l3 = 10;
    int neurons_per_layer[] = {l1, l2, l3};

    // Allocate and load weights and biases with correct dimensions:
    float **weights_fc1 = (float **)malloc(l2 * sizeof(float *));
    float **weights_fc2 = (float **)malloc(l3 * sizeof(float *));
    for (int i = 0; i < l2; i++) {
        weights_fc1[i] = (float *)malloc(l1 * sizeof(float));
    }
    for (int i = 0; i < l3; i++) {
        weights_fc2[i] = (float *)malloc(l2 * sizeof(float));
    }
    load_weights("weights_fc1.txt", weights_fc1, l2, l1);
    load_weights("weights_fc2.txt", weights_fc2, l3, l2);
    printf("Weights loaded\n");
    // Load biases from files
    float *bias_fc1 = (float *)malloc(l2 * sizeof(float));
    float *bias_fc2 = (float *)malloc(l3 * sizeof(float));
    load_bias("bias_fc1.txt", bias_fc1, l2);
    load_bias("bias_fc2.txt", bias_fc2, l3);
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
    char*** initial_spikes = allocate_spike_array();
    if (!initial_spikes) return 1;
    char* labels = allocate_labels(NUM_SAMPLES);

    // Read data into allocated arrays
    if (read_spike_data("mnist_input_spikes.csv", initial_spikes) || read_labels("mnist_labels.csv", labels, NUM_SAMPLES)) {
        free_spike_array(initial_spikes);
        free(labels);
        return 1;
    }

    // Load initial spikes from CSV file
    // float **initial_spikes = (float **)malloc(NUM_SAMPLES * sizeof(float *));
    // for (int i = 0; i < NUM_SAMPLES; i++) {
    //     initial_spikes[i] = (float *)malloc(TIME_WINDOW * sizeof(float));
    // }
    // load_csv("spikes.csv", initial_spikes, NUM_SAMPLES, TIME_WINDOW);
    if (!validate_spike_data(initial_spikes)) {
        printf("Invalid spike data\n");
        free_spike_array(initial_spikes);
        free(labels);
        return 1;
    }
    // printf("%d\n", initial_spikes[0][3][0]);

    // Process each data point
    FILE *output_file = fopen("model_output.txt", "w");
    if (output_file == NULL) {
        perror("Failed to open output file");
        exit(EXIT_FAILURE);
    }


    struct timeval start, stop;
    gettimeofday(&start, NULL);

    printf("\033[1;32mStarting Sim\033[0m\n");
    int num_chunks = TIME_WINDOW / TAU;
    // for (int d = 0; d < NUM_SAMPLES; d++) {
    for (int d = 2; d < 3; d++) {
        int **firing_counts = (int **)malloc(network.layers[network.num_layers - 1].num_neurons * sizeof(int *));
        for (int i = 0; i < network.layers[network.num_layers - 1].num_neurons; i++) {
            firing_counts[i] = (int *)calloc(num_chunks, sizeof(int));
        }
        printf("\r\033[KSample: \033[1;37m%d\033[0m/%d", d+1, NUM_SAMPLES);
        fflush(stdout);
        // printf("Processing Sample %d\n", d);
        // printf("Label: %d\n", labels[d]);

        // Process each chunk of TAU time steps
        for (int chunk = 0; chunk < TIME_WINDOW; chunk += TAU) {
            int chunk_index = chunk / TAU;
            // printf("Processing Chunk %d\n", chunk);
            // Initialize input spikes for the first layer from the loaded data
            for (int t = 0; t < TAU; t++) {
                for (int i = 0; i < network.layers[0].num_neurons; i++) {
                    set_bit(ping_pong_buffer_1, i, t, initial_spikes[d][chunk + t][i]);
                }
            }

            // printf("Input spikes at chunk %d:\n", chunk);
            // print_spike_buffer((const char **)ping_pong_buffer_1, network.layers[0].num_neurons);

            // Process each layer sequentially
            for (int l = 0; l < network.num_layers; l++) {
                int input_size = (l == 0) ? network.layers[l].num_neurons : network.layers[l - 1].num_neurons;

                // printf("Simulating Layer %d\n", l);
                update_layer((const char **)ping_pong_buffer_1, ping_pong_buffer_2, &network.layers[l], input_size);

                // Swap the ping-pong buffers for the next layer
                char **temp = ping_pong_buffer_1;
                ping_pong_buffer_1 = ping_pong_buffer_2;
                ping_pong_buffer_2 = temp;
            }

            // Accumulate firing counts for the final layer
            for (int i = 0; i < network.layers[network.num_layers - 1].num_neurons; i++) {
                for (int t = 0; t < TAU; t++) {
                    if (get_bit((const char **)ping_pong_buffer_1, i, t)) {
                        firing_counts[i][chunk_index]++;
                    }
                }
            }
        }
        printf("Output spikes at sample %d:\n", d);
        print_ping_pong_buffers((const char **)ping_pong_buffer_1, (const char **)ping_pong_buffer_2, network.layers[network.num_layers-1].num_neurons);
        printf("\033[1;33mFiring counts for sample %d:\033[0m\n", d);
        print_firing_counts(firing_counts, network.layers[network.num_layers - 1].num_neurons, num_chunks);
        classify_spike_trains(firing_counts, network.layers[network.num_layers - 1].num_neurons, output_file, d, num_chunks, labels);

        // Free firing counts memory for this sample
        for (int i = 0; i < network.layers[network.num_layers - 1].num_neurons; i++) {
            free(firing_counts[i]);
        }
        free(firing_counts);
    }
    printf("\n");
    printf("\033[1;32mSim Finished\033[0m\n");
    fclose(output_file);

    gettimeofday(&stop, NULL);

    // Free the allocated memory
    for (int i = 0; i < 10; i++) {
        free(weights_fc1[i]);
    }
    for (int i = 0; i < 2; i++) {
        free(weights_fc2[i]);
    }
    free(weights_fc1);
    free(weights_fc2);

    printf("Freeing spike array\n");
    free_spike_array(initial_spikes);
    printf("Freeing labels\n");
    free_labels(labels);

    for (int i = 0; i < MAX_NEURONS; i++) {
        free(ping_pong_buffer_1[i]);
        free(ping_pong_buffer_2[i]);
    }
    free(ping_pong_buffer_1);
    free(ping_pong_buffer_2);
    free(bias_fc1);
    free(bias_fc2);

    printf("Simulation took %lu ms\n", (stop.tv_sec - start.tv_sec) * 1000 + (stop.tv_usec - start.tv_usec) / 1000);
    printf("Average time per sample: %ld ms\n", ((stop.tv_sec - start.tv_sec) * 1000 + (stop.tv_usec - start.tv_usec) / 1000) / NUM_SAMPLES);

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
            network.layers[l].neurons[i].delayed_reset = 0;
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

int heaviside(float x, int threshold) {
    return (x > threshold) ? 1 : 0;
}

// Function to update the entire layer based on the buffer and bias
void update_layer(const char **input, char **output, Layer *layer, int input_size) {
    for (int i = 0; i < layer->num_neurons; i++) {
        for (int t = 0; t < TAU; t++) {
            printf("Time step %d\n", t);

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


// Function to classify spike trains based on the firing frequency of the last layer
void classify_spike_trains(int **firing_counts, int num_neurons, FILE *output_file, int sample_index, int num_chunks, char* labels) {
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
    fprintf(output_file, "Sample %d: Classification = %d, Firing Count = %d, Label = %d\n", sample_index, classification, max_firing_count, labels[sample_index]);
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

char*** allocate_spike_array() {
    char *data_block = malloc(NUM_SAMPLES * TIME_WINDOW * INPUT_SIZE * sizeof(char));
    if (!data_block) {
        perror("Memory allocation error for data block");
        return NULL;
    }
    
    // Allocate the array of sample pointers.
    char ***spikes = malloc(NUM_SAMPLES * sizeof(char **));
    if (!spikes) {
        perror("Memory allocation error for spike array");
        free(data_block);
        return NULL;
    }
    
    // For each sample, allocate an array of pointers (one per time step).
    for (int i = 0; i < NUM_SAMPLES; i++) {
        spikes[i] = malloc(TIME_WINDOW * sizeof(char *));
        if (!spikes[i]) {
            perror("Memory allocation error for spike row pointers");
            for (int j = 0; j < i; j++) {
                free(spikes[j]);
            }
            free(spikes);
            free(data_block);
            return NULL;
        }
        // Set each time pointer to the correct offset in the contiguous block.
        for (int j = 0; j < TIME_WINDOW; j++) {
            spikes[i][j] = data_block + (i * TIME_WINDOW * INPUT_SIZE) + (j * INPUT_SIZE);
        }
    }
    
    return spikes;
}

void free_spike_array(char*** spikes) {
    if (spikes) {
        free(spikes[0][0]);  // Free the contiguous data block.
        for (int i = 0; i < NUM_SAMPLES; i++) {
            free(spikes[i]);  // Free each sample's row pointer array.
        }
        free(spikes);  // Free the array of sample pointers.
    }
}

int validate_spike_data(char ***spikes) {
    if (!spikes || !spikes[0] || !spikes[0][0]) {
        fprintf(stderr, "Invalid spike pointer structure\n");
        return 0;
    }

    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < TIME_WINDOW; j++) {
            for (int k = 0; k < INPUT_SIZE; k++) {
                char value = spikes[i][j][k];
                if (value != 0 && value != 1) {
                    fprintf(stderr, "Invalid value at [%d][%d][%d]: %d\n", i, j, k, value);
                    return 0;
                }
            }
        }
    }
    printf("All spike data loaded correctly.\n");
    return 1;
}

char *allocate_labels(int num_samples) {
    char *labels = malloc(num_samples * sizeof(char));
    if (!labels) {
        perror("Failed to allocate memory for labels");
        return NULL;
    }
    return labels;
}

void free_labels(char *labels) {
    free(labels);
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