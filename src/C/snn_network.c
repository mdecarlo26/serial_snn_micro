#include <stdint.h>
#include "snn_network.h"
#include "define.h"
#include <stdlib.h>

extern Snn_Network snn_network;
static Layer static_layers[MAX_LAYERS];
static Neuron static_neurons[MAX_LAYERS][MAX_NEURONS];
// static float* static_weights[MAX_LAYERS][MAX_NEURONS];
// static float  static_weight_data[MAX_LAYERS][MAX_NEURONS][MAX_NEURONS];
// static float  static_bias[MAX_LAYERS][MAX_NEURONS];
extern uint8_t ping_pong_buffer_1[MAX_NEURONS][BITMASK_BYTES];
extern uint8_t ping_pong_buffer_2[MAX_NEURONS][BITMASK_BYTES];

// Function to set a value in the buffer
void set_bit(uint8_t buffer[MAX_NEURONS][BITMASK_BYTES], int neuron_idx, int t, int value) {
    int byte_idx = t / 8;
    int bit_idx = t % 8;
    if (value)
        buffer[neuron_idx][byte_idx] |= (1 << bit_idx);
    else
        buffer[neuron_idx][byte_idx] &= ~(1 << bit_idx);
}

int get_bit(const uint8_t buffer[MAX_NEURONS][BITMASK_BYTES], int neuron_idx, int t) {
    int byte_idx = t / 8;
    int bit_idx = t % 8;
    return (buffer[neuron_idx][byte_idx] >> bit_idx) & 1;
}


int heaviside(float x, int threshold) {
    return (x >= threshold) ? 1 : 0;
}

// Function to update the entire layer based on the buffer and bias
void update_layer(const uint8_t input[MAX_NEURONS][BITMASK_BYTES],
                uint8_t output[MAX_NEURONS][BITMASK_BYTES], Layer *layer, int input_size) {
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

void initialize_network(int neurons_per_layer[],
     const float weights_fc1[HIDDEN_LAYER_1][INPUT_SIZE], const float weights_fc2[NUM_CLASSES][HIDDEN_LAYER_1],
     const float *bias_fc1, const float *bias_fc2) {
    snn_network.layers = static_layers;

    for (int l = 0; l < snn_network.num_layers; l++) {
        // printf("Initializing Layer %d\n", l);
        snn_network.layers[l].layer_num = l;
        snn_network.layers[l].num_neurons = neurons_per_layer[l];

        // Assign statically allocated neuron structures
        snn_network.layers[l].neurons = static_neurons[l];

        // Assign statically allocated weight pointers and bias array
        // snn_network.layers[l].bias = static_bias[l];

        for (int i = 0; i < snn_network.layers[l].num_neurons; i++) {
            snn_network.layers[l].neurons[i].voltage_thresh = VOLTAGE_THRESH;
            snn_network.layers[l].neurons[i].decay_rate = DECAY_RATE;

            if (l > 0) {
                // Set weight pointer for this neuron to static buffer
                // snn_network.layers[l].weights[i] = static_weight_data[l][i];

                // Copy weights and bias from passed-in arguments
                if (l == 1) {
                    snn_network.layers[l].weights = (float **)weights_fc1;
                    // memcpy(snn_network.layers[l].weights[i], weights_fc1[i], snn_network.layers[l - 1].num_neurons * sizeof(float));
                    snn_network.layers[l].bias = bias_fc1;
                } else if (l == 2) {
                    snn_network.layers[l].weights = (float **)weights_fc2;
                    // memcpy(snn_network.layers[l].weights[i], weights_fc2[i], snn_network.layers[l - 1].num_neurons * sizeof(float));
                    snn_network.layers[l].bias = bias_fc2;
                }
            } else {
                snn_network.layers[l].weights = NULL;
                snn_network.layers[l].bias = NULL;
            }
        }
    }
}

void zero_network() {
    for (int l = 0; l < snn_network.num_layers; l++) {
        for (int i = 0; i < snn_network.layers[l].num_neurons; i++) {
            snn_network.layers[l].neurons[i].membrane_potential = 0;
            snn_network.layers[l].neurons[i].delayed_reset = 0;
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

int inference(const uint8_t input[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES],
     uint8_t ping_pong_buffer_1[MAX_NEURONS][BITMASK_BYTES],
     uint8_t ping_pong_buffer_2[MAX_NEURONS][BITMASK_BYTES], int sample_idx){
    zero_network();
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
        int in_spike = 0;
        for (int t = 0; t < TAU; t++) {
            for (int i = 0; i < snn_network.layers[0].num_neurons; i++) {
                in_spike = get_input_spike(input, sample_idx, chunk + t, i);
                // printf("Input spike at sample %d, time %d, neuron %d: %d\n", sample_idx, chunk + t, i, in_spike);
                set_bit(ping_pong_buffer_1, i, t, in_spike);
                // set_bit(ping_pong_buffer_1, snn_network.layers[0].num_neurons-1-i, t, initial_spikes[d][chunk + t][i]);
            }
        }
        // printf("Input spikes at chunk %d:\n", chunk);
        // print_spike_buffer((const char **)ping_pong_buffer_1, snn_network.layers[0].num_neurons);

        // Process each layer sequentially
        for (int l = 0; l < snn_network.num_layers; l++) {
            int input_size = (l == 0) ? snn_network.layers[l].num_neurons : snn_network.layers[l - 1].num_neurons;

            printf("Simulating Layer %d\n", l);
            update_layer(ping_pong_buffer_1, ping_pong_buffer_2, &snn_network.layers[l], input_size);

            // Swap the ping-pong buffers for the next layer
            uint8_t (*temp)[BITMASK_BYTES] = ping_pong_buffer_1;
            ping_pong_buffer_1 = ping_pong_buffer_2;
            ping_pong_buffer_2 = temp;
        }

        // Accumulate firing counts for the final layer
        for (int i = 0; i < snn_network.layers[snn_network.num_layers - 1].num_neurons; i++) {
            for (int t = 0; t < TAU; t++) {
                if (get_bit(ping_pong_buffer_1, i, t)) {
                    firing_counts[i][chunk_index]++;
                }
            }
        }
    }

    return classify_inference(firing_counts, snn_network.layers[snn_network.num_layers - 1].num_neurons, TIME_WINDOW / TAU);
}

void set_input_spike(uint8_t buffer[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES],
                     int sample, int t, int neuron_idx, int value) {
    int byte_idx = neuron_idx / 8;
    int bit_idx  = neuron_idx % 8;

    if (value)
        buffer[sample][t][byte_idx] |= (1 << bit_idx);
    else
        buffer[sample][t][byte_idx] &= ~(1 << bit_idx);
}

int get_input_spike(const uint8_t buffer[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES],
                    int sample, int t, int neuron_idx) {
    int byte_idx = neuron_idx / 8;
    int bit_idx  = neuron_idx % 8;
    return (buffer[sample][t][byte_idx] >> bit_idx) & 1;
}