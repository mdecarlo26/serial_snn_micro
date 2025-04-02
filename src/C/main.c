#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>


#include "define.h"
#include "file_operations.h"
#include "rate_encoding.h"
#include "network.h"
#include "debug.h"

Network network;

// Define the global ping pong buffers as pointers to 2D arrays
char **ping_pong_buffer_1;
char **ping_pong_buffer_2;

// Function prototypes
void initialize_input_spikes(char **input, int num_neurons);
void classify_spike_trains(int **firing_counts, int num_neurons, FILE *output_file, int sample_index, int num_chunks, char* labels);

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
    int l2 = 16;
    int l3 = NUM_CLASSES;
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
    load_weights("../2d_weights_fc1.txt", weights_fc1, l2, l1);
    load_weights("../2d_weights_fc2.txt", weights_fc2, l3, l2);
    printf("Weights loaded\n");
    // Load biases from files
    float *bias_fc1 = (float *)malloc(l2 * sizeof(float));
    float *bias_fc2 = (float *)malloc(l3 * sizeof(float));
    load_bias("../2d_bias_fc1.txt", bias_fc1, l2);
    load_bias("../2d_bias_fc2.txt", bias_fc2, l3);
    printf("Biases loaded\n");

    initialize_network(neurons_per_layer, weights_fc1, weights_fc2, bias_fc1, bias_fc2);
    zero_network();
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
    // if (read_spike_data("../mnist_input_spikes.csv", initial_spikes) || read_labels("../mnist_labels.csv", labels, NUM_SAMPLES)) {
    if (read_spike_data("../input_spikes.csv", initial_spikes) || read_labels("../labels.csv", labels, NUM_SAMPLES)) {
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
    for (int d = 0; d < NUM_SAMPLES; d++) {
    // for (int d = 0; d < 1; d++) {
        int **firing_counts = (int **)malloc(network.layers[network.num_layers - 1].num_neurons * sizeof(int *));
        for (int i = 0; i < network.layers[network.num_layers - 1].num_neurons; i++) {
            firing_counts[i] = (int *)calloc(num_chunks, sizeof(int));
        }
        zero_network();
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
                    // set_bit(ping_pong_buffer_1, network.layers[0].num_neurons-1-i, t, initial_spikes[d][chunk + t][i]);
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
        // printf("Output spikes at sample %d:\n", d);
        // print_ping_pong_buffers((const char **)ping_pong_buffer_1, (const char **)ping_pong_buffer_2, network.layers[network.num_layers-1].num_neurons);
        // printf("\033[1;33mFiring counts for sample %d:\033[0m\n", d);
        // print_firing_counts(firing_counts, network.layers[network.num_layers - 1].num_neurons, num_chunks);
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
