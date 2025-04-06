#include "network.h"
#include "define.h"
#include "debug.h"
#include <stdlib.h>

extern Network network;
static Layer static_layers[MAX_LAYERS];
static Neuron static_neurons[MAX_LAYERS][MAX_NEURONS];
static float* static_weights[MAX_LAYERS][MAX_NEURONS];
static float  static_weight_data[MAX_LAYERS][MAX_NEURONS][MAX_NEURONS];
static float  static_bias[MAX_LAYERS][MAX_NEURONS];
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
                for (int j = 0; j < input_size; j++) {
                    if (get_bit(input, j, t)) { // if incoming spike is present
                        if (layer->layer_num == 0) {
                            sum += 1.0f; // For the input layer, each spike contributes a value of 1
                        } else {
                            sum += layer->weights[i][j];
                        }
                    }
                }
            }
            else{
                if (get_bit(input, i, t)) { // if incoming spike is present
                    sum += 1.0f; // For the input layer, each spike contributes a value of 1
                }
            }
                       
            // printf("Neuron %d: Old Membrane Potential = %f\n", i, layer->neurons[i].membrane_potential);

            float new_mem = 0;
            int reset_signal = heaviside(layer->neurons[i].membrane_potential,layer->neurons[i].voltage_thresh);
            new_mem = layer->neurons[i].decay_rate * layer->neurons[i].membrane_potential + sum - reset_signal * layer->neurons[i].voltage_thresh;
            layer->neurons[i].membrane_potential = new_mem;
            int output_spike = heaviside(layer->neurons[i].membrane_potential, layer->neurons[i].voltage_thresh);
            set_bit(output, i, t, output_spike); // Reset output for this time step
            // printf("Neuron %d: Membrane Potential = %f, Output = %d, Reset: %d, Sum: %d\n", i, layer->neurons[i].membrane_potential, output_spike, reset_signal, sum);
        }
    }
}


// Function to initialize weights, biases, and neuron properties
// void initialize_network(int neurons_per_layer[], float **weights_fc1, float **weights_fc2, float *bias_fc1, float *bias_fc2) {
//     network.layers = (Layer *)malloc(network.num_layers * sizeof(Layer));
//     for (int l = 0; l < network.num_layers; l++) {
//         printf("Initializing Layer %d\n", l);
//         network.layers[l].layer_num = l;
//         network.layers[l].num_neurons = neurons_per_layer[l];
//         network.layers[l].neurons = (Neuron *)malloc(network.layers[l].num_neurons * sizeof(Neuron));
//         network.layers[l].weights = (float **)malloc(network.layers[l].num_neurons * sizeof(float *));
//         network.layers[l].bias = (float *)malloc(network.layers[l].num_neurons * sizeof(float));
//         for (int i = 0; i < network.layers[l].num_neurons; i++) {
//             network.layers[l].neurons[i].voltage_thresh = VOLTAGE_THRESH;
//             network.layers[l].neurons[i].decay_rate = DECAY_RATE;
//             if (l > 0) { // Allocate weights and biases for layers after the input layer
//                 network.layers[l].weights[i] = (float *)malloc(network.layers[l - 1].num_neurons * sizeof(float));
//                 if (l == 1) {
//                     memcpy(network.layers[l].weights[i], weights_fc1[i], network.layers[l - 1].num_neurons * sizeof(float));
//                     network.layers[l].bias[i] = bias_fc1[i];
//                 } else if (l == 2) {
//                     memcpy(network.layers[l].weights[i], weights_fc2[i], network.layers[l - 1].num_neurons * sizeof(float));
//                     network.layers[l].bias[i] = bias_fc2[i];
//                 }
//             } else {
//                 network.layers[l].weights[i] = NULL; // No weights for the input layer
//                 network.layers[l].bias[i] = 0;         // No bias for the input layer
//             }
//         }
//     }
// }

void initialize_network(int neurons_per_layer[], float **weights_fc1, float **weights_fc2, float *bias_fc1, float *bias_fc2) {
    network.layers = static_layers;

    for (int l = 0; l < network.num_layers; l++) {
        printf("Initializing Layer %d\n", l);
        network.layers[l].layer_num = l;
        network.layers[l].num_neurons = neurons_per_layer[l];

        // Assign statically allocated neuron structures
        network.layers[l].neurons = static_neurons[l];

        // Assign statically allocated weight pointers and bias array
        network.layers[l].weights = static_weights[l];
        network.layers[l].bias = static_bias[l];

        for (int i = 0; i < network.layers[l].num_neurons; i++) {
            network.layers[l].neurons[i].voltage_thresh = VOLTAGE_THRESH;
            network.layers[l].neurons[i].decay_rate = DECAY_RATE;

            if (l > 0) {
                // Set weight pointer for this neuron to static buffer
                network.layers[l].weights[i] = static_weight_data[l][i];

                // Copy weights and bias from passed-in arguments
                if (l == 1) {
                    memcpy(network.layers[l].weights[i], weights_fc1[i], network.layers[l - 1].num_neurons * sizeof(float));
                    network.layers[l].bias[i] = bias_fc1[i];
                } else if (l == 2) {
                    memcpy(network.layers[l].weights[i], weights_fc2[i], network.layers[l - 1].num_neurons * sizeof(float));
                    network.layers[l].bias[i] = bias_fc2[i];
                }
            } else {
                network.layers[l].weights[i] = NULL;
                network.layers[l].bias[i] = 0;
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

int classify_inference(int **firing_counts, int num_neurons, int num_chunks){
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
    return classification;
}

int inference(const char **input, char** ping_pong_buffer_1, char** ping_pong_buffer_2){

    static int firing_counts_data[NUM_CLASSES][TIME_WINDOW / TAU] = {0};
    int* firing_counts[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        firing_counts[i] = firing_counts_data[i];
        // Zero the row before use (manual clear)
        for (int j = 0; j < TIME_WINDOW / TAU; j++) {
            firing_counts[i][j] = 0;
        }
    }

    for (int chunk = 0; chunk < TIME_WINDOW; chunk += TAU) {
        int chunk_index = chunk / TAU;
        // printf("Processing Chunk %d\n", chunk);
        // Initialize input spikes for the first layer from the loaded data
        for (int t = 0; t < TAU; t++) {
            for (int i = 0; i < network.layers[0].num_neurons; i++) {
                set_bit(ping_pong_buffer_1, i, t, input[chunk + t][i]);
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

    return classify_inference(firing_counts, network.layers[network.num_layers - 1].num_neurons, TIME_WINDOW / TAU);
}