#include <stdint.h>
#include "snn_network.h"
#include "define.h"
#include <stdlib.h>

extern Snn_Network snn_network;

static Layer static_layers[MAX_LAYERS];
static Neuron static_neurons[MAX_LAYERS][MAX_NEURONS];

static int8_t *fc1_pointer_table[HIDDEN_LAYER_1];
static int8_t *fc2_pointer_table[NUM_CLASSES];

static int8_t *fc1_bias_pointer = NULL;
static int8_t *fc2_bias_pointer = NULL;
static int weights_initialized = 0;

// Backing storage for the two ping-pong buffers:
// Static memory for ping-pong buffers
// Each neuron has BITMASK_BYTES bytes, and there are MAX_NEURONS neurons
// static uint8_t ping_pong_buffer_storage_1[MAX_NEURONS][BITMASK_BYTES] = {0};
// static uint8_t ping_pong_buffer_storage_2[MAX_NEURONS][BITMASK_BYTES] = {0};
static uint8_t ping_pong_buffer_storage_1[TAU][INPUT_BYTES] = {0};
static uint8_t ping_pong_buffer_storage_2[TAU][INPUT_BYTES] = {0};

// Pointers that we can swap:
// static uint8_t (*ping_pong_buffer_1)[BITMASK_BYTES] = ping_pong_buffer_storage_1;
// static uint8_t (*ping_pong_buffer_2)[BITMASK_BYTES] = ping_pong_buffer_storage_2;
static uint8_t (*ping_pong_buffer_1)[INPUT_BYTES] = ping_pong_buffer_storage_1;
static uint8_t (*ping_pong_buffer_2)[INPUT_BYTES] = ping_pong_buffer_storage_2;

// Function to set a value in the buffer
void set_bit(uint8_t buffer[TAU][INPUT_BYTES], int neuron_idx, int t, int value) {
    int byte_idx = neuron_idx / 8;
    int bit_idx = neuron_idx % 8;
    if (value)
        buffer[t][byte_idx] |= (1 << (8-bit_idx));
    else
        buffer[t][byte_idx] &= ~(1 << (8-bit_idx));
}

int get_bit(const uint8_t buffer[TAU][INPUT_BYTES], int neuron_idx, int t) {
    int byte_idx = neuron_idx / 8;
    int bit_idx = neuron_idx % 8;
    return (buffer[t][byte_idx] >> (8-bit_idx)) & 1;
}

// Function to update the entire layer based on the buffer and bias
void update_layer(const uint8_t input[TAU][INPUT_BYTES],
                uint8_t output[TAU][INPUT_BYTES], Layer *layer, int input_size) {
    printf("Layer %d\n", layer->layer_num);
    printf("Number of neurons: %d\n", layer->num_neurons);
    for (int t = 0; t < TAU; t++) {
        for (int i = 0; i < layer->num_neurons; i++) {
            printf("Time step %d: ", t);
#if (Q07_FLAG)
            int32_t sum = layer->bias[i];
#else
            float sum = dequantize_q07(layer->bias[i]);
#endif

            int num_bytes = (input_size + 7) / 8;
            for (int byte_idx = 0; byte_idx < INPUT_BYTES; byte_idx++) {
                uint8_t byte = input[t][byte_idx];
                int base_idx = byte_idx * 8;

                while (byte) {
                    int bit = __builtin_ctz(byte);
                    int j = base_idx + bit;
                    if (j < input_size) {
#if (Q07_FLAG)
                        sum += layer->weights[i][j];
#else
                        sum += dequantize_q07(layer->weights[i][j]);
#endif
                    }
                    byte &= byte - 1;
                }
            }

            int reset_signal = HEAVISIDE(layer->neurons[i].membrane_potential,
                                        layer->neurons[i].voltage_thresh);

#if (LIF)
    #if (Q07_FLAG)
            int32_t new_mem = ((DECAY_FP7 * layer->neurons[i].membrane_potential) >> DECAY_SHIFT)
                            + sum - reset_signal * layer->neurons[i].voltage_thresh;
    #else
            float new_mem = layer->neurons[i].decay_rate * layer->neurons[i].membrane_potential
                            + sum - reset_signal * layer->neurons[i].voltage_thresh;
    #endif
#elif (IF)
    #if (Q07_FLAG)
            int32_t new_mem = layer->neurons[i].membrane_potential + sum
                            - reset_signal * layer->neurons[i].voltage_thresh;
    #else
            float new_mem = layer->neurons[i].membrane_potential + sum
                            - reset_signal * layer->neurons[i].voltage_thresh;
    #endif
#endif

            layer->neurons[i].membrane_potential = new_mem;
            if (reset_signal)
                set_bit(output, i, t, 1);
        }
    }
}

void initialize_network(int neurons_per_layer[],
     const int8_t weights_fc1[HIDDEN_LAYER_1][INPUT_SIZE], const int8_t weights_fc2[NUM_CLASSES][HIDDEN_LAYER_1],
     const int8_t *bias_fc1, const int8_t *bias_fc2) {
    snn_network.layers = static_layers;

    if (!weights_initialized) {
        for (int i = 0; i < HIDDEN_LAYER_1; i++) {
            fc1_pointer_table[i] = (int8_t *)weights_fc1[i];
        }
        for (int i = 0; i < NUM_CLASSES; i++) {
            fc2_pointer_table[i] = (int8_t *)weights_fc2[i];
        }

        fc1_bias_pointer = (int8_t *)bias_fc1;
        fc2_bias_pointer = (int8_t *)bias_fc2;

        weights_initialized = 1;
    }

    for (int l = 0; l < snn_network.num_layers; l++) {
        snn_network.layers[l].layer_num = l;
        snn_network.layers[l].num_neurons = neurons_per_layer[l];
        snn_network.layers[l].neurons = static_neurons[l];

        if (l == 1) {
            snn_network.layers[l].weights = fc1_pointer_table;
            snn_network.layers[l].bias = fc1_bias_pointer;
        } else if (l == 2) {
            snn_network.layers[l].weights = fc2_pointer_table;
            snn_network.layers[l].bias = fc2_bias_pointer;
        } else {
            snn_network.layers[l].weights = NULL;
            snn_network.layers[l].bias = NULL;
        }

        for (int i = 0; i < snn_network.layers[l].num_neurons; i++) {
#if (Q07_FLAG)
            snn_network.layers[l].neurons[i].membrane_potential = 0;
            snn_network.layers[l].neurons[i].voltage_thresh = VOLTAGE_THRESH_FP7;
            snn_network.layers[l].neurons[i].decay_rate = DECAY_FP7;
            snn_network.layers[l].neurons[i].delayed_reset = 0;
#else
            snn_network.layers[l].neurons[i].membrane_potential = 0.0f;
            snn_network.layers[l].neurons[i].voltage_thresh = VOLTAGE_THRESH;
            snn_network.layers[l].neurons[i].decay_rate = DECAY_RATE;
            snn_network.layers[l].neurons[i].delayed_reset = 0.0f;
#endif
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

int inference(const uint8_t input[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES], int sample_idx){
    zero_network();
    static int firing_counts_data[NUM_CLASSES][TIME_WINDOW / TAU] = {0};
    int* firing_counts[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        firing_counts[i] = firing_counts_data[i];
        for (int j = 0; j < TIME_WINDOW / TAU; j++) {
            firing_counts[i][j] = 0;
        }
    }

    printf("Sparsity is the percentage of neurons that are firing in the layer\n");
    for (int chunk = 0; chunk < TIME_WINDOW; chunk += TAU) {
        int chunk_index = chunk / TAU;
        for (int t = 0; t < TAU; t++) {
            for (int i = 0; i < snn_network.layers[0].num_neurons; i++) {
                int in_spike = get_input_spike(input, sample_idx, chunk + t, i);
                set_bit(ping_pong_buffer_1, i, t, in_spike);
            }
        }
        for (int l = 0; l < snn_network.num_layers; l++) {
            float layer_sparsity[TAU];
            int input_size = (l == 0) ? snn_network.layers[l].num_neurons : snn_network.layers[l - 1].num_neurons;

            // compute_buffer_sparsity(ping_pong_buffer_1, input_size, layer_sparsity);

            // printf("Layer %d input sparsity:", l);
            // for (int t = 0; t < TAU; t++) {
            //     printf(" %.2f", layer_sparsity[t]);
            // }
            // printf("\n");

        printf("Chunk %d\n", chunk);
            update_layer(ping_pong_buffer_1, ping_pong_buffer_2, &snn_network.layers[l], input_size);
        printf("Fail\n");

            // Swap pointers
            uint8_t (*temp)[INPUT_BYTES] = ping_pong_buffer_1;
            ping_pong_buffer_1 = ping_pong_buffer_2;
            ping_pong_buffer_2 = temp;
        }

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

// === Quantize float32 to Q0.8 (int8_t) ===
// Range: [-1.0, 0.99609375]
int8_t quantize_q07(float x) {
    if (x > Q07_MAX_FLOAT)  x = Q07_MAX_FLOAT;
    if (x < Q07_MIN_FLOAT)  x = Q07_MIN_FLOAT;

    int32_t scaled = (int32_t)(x * Q07_SCALE + (x >= 0 ? 0.5f : -0.5f));

    if (scaled > Q07_MAX_INT8)  scaled = Q07_MAX_INT8;
    if (scaled < Q07_MIN_INT8)  scaled = Q07_MIN_INT8;

    return (int8_t)scaled;
}

float dequantize_q07(int32_t q) {
    return ((float)q) * Q07_INV_SCALE;
}

void compute_buffer_sparsity(const uint8_t buffer[TAU][INPUT_BYTES],
                             int num_neurons,
                             float sparsity[TAU]) {
    for (int t = 0; t < TAU; t++) {
        int active_spike_count = 0;
        for (int byte = 0; byte < (num_neurons + 7) / 8; byte++) {
            uint8_t val = buffer[t][byte];
            while (val) {
                val &= (val - 1);  // Clear the least significant set bit
                active_spike_count++;
            }
        }
        sparsity[t] = ((float)active_spike_count / num_neurons);
    }
}
