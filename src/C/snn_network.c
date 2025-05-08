#include <stdint.h>
#include "dsp_helper.h"
#include "snn_network.h"
#include "define.h"
#include <stdlib.h>

extern Snn_Network snn_network;

static Layer static_layers[NUM_LAYERS];

static int8_t *fc1_pointer_table[INPUT_SIZE];
static int8_t *fc2_pointer_table[HIDDEN_LAYER_1];

static int8_t *fc1_bias_pointer = NULL;
static int8_t *fc2_bias_pointer = NULL;
static int weights_initialized = 0;

// Backing storage for the two ping-pong buffers:
// Static memory for ping-pong buffers
// Each neuron has BITMASK_BYTES bytes, and there are MAX_NEURONS neurons
static uint8_t ping_pong_buffer_storage_1[TAU][INPUT_BYTES] = {0};
static uint8_t ping_pong_buffer_storage_2[TAU][INPUT_BYTES] = {0};

// Pointers that we can swap:
static uint8_t (*ping_pong_buffer_1)[INPUT_BYTES] = ping_pong_buffer_storage_1;
static uint8_t (*ping_pong_buffer_2)[INPUT_BYTES] = ping_pong_buffer_storage_2;

// Function to update the entire layer based on the buffer and bias
void update_layer(const uint8_t input[TAU][INPUT_BYTES],
                  uint8_t output[TAU][INPUT_BYTES],
                  Layer *layer, int input_size) {
    int num_bytes = (input_size + 7) / 8;
    int N = layer->layer_num;
    // printf("Layer %d: num_neurons = %d, input_size = %d\n", layer->layer_num, layer->num_neurons, input_size);

    // scratch buffers for column and sums
    for (int t = 0; t < TAU; t++) {
#if (Q07_FLAG)
            static int32_t sums[MAX_NEURONS]  __attribute__((aligned(4)));
            memset(sums, 0, layer->num_neurons * sizeof(int32_t));
#else
            static float sums[MAX_NEURONS]  __attribute__((aligned(4)));
            memset(sums, 0, layer->num_neurons * sizeof(float));
#endif

            if (N > 0) {
                // Hidden or output layer: sum over presynaptic spikes
#if (Q07_FLAG)
                vectorize_q7_add_to_q31(
                    layer->bias,
                    sums,
                    layer->num_neurons
                );
#else
                for (int j=0 ; j < input_size; j++) {
                    sums[j] = dequantize_q07(layer->bias[j]);
                }
#endif
                for (int byte_idx = 0; byte_idx < num_bytes; byte_idx++) {
                    uint8_t byte = input[t][byte_idx];
                    int base_idx = byte_idx * 8;

                    while (byte) {
                        int bit = __builtin_ctz(byte);
                        int j = base_idx + bit;
                        if (j < input_size) {
#if (Q07_FLAG)
                            vectorize_q7_add_to_q31(
                                layer->weights[j],
                                sums,
                                layer->num_neurons
                            );
#else
                        for (int i=0 ; i < input_size; i++) {
                            sums[i] = dequantize_q07(layer->weights[i][j]);
                        }
#endif
                        }
                        byte &= byte - 1;  // Clear least significant set bit
                    }
                }

            } else {
                // Input layer: spike from self (i-th input neuron only)
                // This is a bit of a hack, but it works for the input layer
                // and is a bit faster than the alternative of using a separate
                // function to handle the input layer.
                for (int i = 0; i < layer->num_neurons; i++) {
                if (GET_BIT(input[t], i)) {
#if (Q07_FLAG)
                    sums[i] += (1 << DECAY_SHIFT);  // Q0.7 equivalent of +1
#else               
                    sums[i] += 1.0f;
#endif
                }
                }
            }

            for (int i = 0; i < layer->num_neurons; i++) {
            layer->delayed_resets[i]  = HEAVISIDE(layer->membrane_potentials[i],
                                         layer->voltage_thresholds[i]);
#if (LIF)
    #if (Q07_FLAG)
            int32_t new_mem = ((DECAY_FP7 * layer->membrane_potentials[i]) >> DECAY_SHIFT)
                      + sums[i]
                      - layer->delayed_resets[i] * layer->voltage_thresholds[i];
    #else
            float new_mem = (int32_t)(layer->decay_rates[i] * layer->membrane_potentials[i])
                      + sums[i]
                      - layer->delayed_resets[i] * layer->voltage_thresholds[i];
    #endif
#elif (IF)
    #if (Q07_FLAG)
            int32_t new_mem = layer->membrane_potentials[i]
                      + sums[i]
                      - layer->delayed_resets[i] * layer->voltage_thresholds[i];
    #else
            float new_mem = (int32_t)((float)layer->membrane_potentials[i] + (float)sums[i])
                      - layer->delayed_resets[i] * layer->voltage_thresholds[i];
    #endif
#endif

            layer->membrane_potentials[i] = new_mem;
            SET_BIT(output[t], i, layer->delayed_resets[i]);
        }
    }
}

void initialize_network(int neurons_per_layer[],
     const int8_t weights_fc1[INPUT_SIZE][HIDDEN_LAYER_1], const int8_t weights_fc2[HIDDEN_LAYER_1][NUM_CLASSES],
     const int8_t *bias_fc1, const int8_t *bias_fc2) {
    snn_network.layers = static_layers;

    if (!weights_initialized) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            fc1_pointer_table[i] = (int8_t *)weights_fc1[i];
        }
        for (int i = 0; i < HIDDEN_LAYER_1; i++) {
            fc2_pointer_table[i] = (int8_t *)weights_fc2[i];
        }

        fc1_bias_pointer = (int8_t *)bias_fc1;
        fc2_bias_pointer = (int8_t *)bias_fc2;

        weights_initialized = 1;
    }

    for (int l = 0; l < snn_network.num_layers; l++) {
        snn_network.layers[l].layer_num = l;
        snn_network.layers[l].num_neurons = neurons_per_layer[l];


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
#if (Q07_FLAG)
            memset(snn_network.layers[l].membrane_potentials, 0, snn_network.layers[l].num_neurons * sizeof(int32_t));
            memset(snn_network.layers[l].delayed_resets, 0, snn_network.layers[l].num_neurons * sizeof(int32_t));
            memset(snn_network.layers[l].voltage_thresholds, (int16_t)VOLTAGE_THRESH_FP7, snn_network.layers[l].num_neurons * sizeof(int16_t));
            memset(snn_network.layers[l].decay_rates, DECAY_FP7, snn_network.layers[l].num_neurons * sizeof(int16_t));
#else
            memset(snn_network.layers[l].membrane_potentials, 0.0, snn_network.layers[l].num_neurons * sizeof(float));
            memset(snn_network.layers[l].delayed_resets, 0.0, snn_network.layers[l].num_neurons * sizeof(float));
            memset(snn_network.layers[l].voltage_thresholds, VOLTAGE_THRESH, snn_network.layers[l].num_neurons * sizeof(float));
            memset(snn_network.layers[l].decay_rates, DECAY_RATE, snn_network.layers[l].num_neurons * sizeof(float));
#endif
        printf("volt: %d, decay: %d\n", VOLTAGE_THRESH_FP7, DECAY_FP7);
        for (int i = 0; i < snn_network.layers[l].num_neurons; i++) {
            printf("Layer %d, Neuron %d: Decay = %d, Voltage Thresh = %d\n",
                   l, i, snn_network.layers[l].decay_rates[i], snn_network.layers[l].voltage_thresholds[i]);
        }
    }
}

void zero_network() {
    for (int l = 0; l < snn_network.num_layers; l++) {
#if (Q07_FLAG)
        memset(snn_network.layers[l].membrane_potentials, 0, snn_network.layers[l].num_neurons * sizeof(int32_t));
        memset(snn_network.layers[l].delayed_resets, 0, snn_network.layers[l].num_neurons * sizeof(int32_t));
#else
        memset(snn_network.layers[l].membrane_potentials, 0.0, snn_network.layers[l].num_neurons * sizeof(float));
        memset(snn_network.layers[l].delayed_resets, 0.0, snn_network.layers[l].num_neurons * sizeof(float));
#endif
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

    // printf("Sparsity is the percentage of neurons that are firing in the layer\n");
    for (int chunk = 0; chunk < TIME_WINDOW; chunk += TAU) {
        int chunk_index = chunk / TAU;
        for (int t = 0; t < TAU; t++) {
            for (int i = 0; i < snn_network.layers[0].num_neurons; i++) {
                int in_spike = get_input_spike(input, sample_idx, chunk + t, i);
                // set_bit(ping_pong_buffer_1, i, t, in_spike);
                SET_BIT(ping_pong_buffer_1[t], i, in_spike);
            }
        }
        for (int l = 0; l < snn_network.num_layers; l++) {
            int input_size = (l == 0) ? snn_network.layers[l].num_neurons : snn_network.layers[l - 1].num_neurons;

            // float layer_sparsity[TAU];
            // compute_buffer_sparsity(ping_pong_buffer_1, input_size, layer_sparsity);

            // printf("Layer %d input sparsity:", l);
            // for (int t = 0; t < TAU; t++) {
            //     printf(" %.2f", layer_sparsity[t]);
            // }
            // printf("\n");

            update_layer(ping_pong_buffer_1, ping_pong_buffer_2, &snn_network.layers[l], input_size);

            // Swap pointers
            uint8_t (*temp)[INPUT_BYTES] = ping_pong_buffer_1;
            ping_pong_buffer_1 = ping_pong_buffer_2;
            ping_pong_buffer_2 = temp;
        }

        for (int i = 0; i < snn_network.layers[snn_network.num_layers - 1].num_neurons; i++) {
            for (int t = 0; t < TAU; t++) {
                // if (get_bit(ping_pong_buffer_1, i, t)) {
                if (GET_BIT(ping_pong_buffer_1[t], i)) {
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
