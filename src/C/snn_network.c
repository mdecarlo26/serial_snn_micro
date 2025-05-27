#include <stdint.h>
#include "dsp_helper.h"
#include "snn_network.h"
#include "define.h"
#include <stdlib.h>

extern Snn_Network snn_network;

static Layer static_layers[NUM_LAYERS];

// static int8_t *fc1_pointer_table[INPUT_SIZE];
// static int8_t *fc2_pointer_table[HIDDEN_LAYER_1];
static int8_t *fc2_pointer_table[CONV1_NEURONS];

// static int8_t *fc1_bias_pointer = NULL;
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
                  uint8_t       output[TAU][INPUT_BYTES],
                  Layer        *layer,
                  int            input_size)
{
    int num_bytes = (input_size + 7) / 8;

    for (int t = 0; t < TAU; t++) {
        // Aligned scratch for sums
    #if (Q07_FLAG)
        static int32_t sums[MAX_NEURONS]; __attribute__((aligned(4)));
    #else
        static float   sums[MAX_NEURONS]; __attribute__((aligned(4)));
    #endif

        // Zero out sums
        memset(sums, 0, layer->num_neurons * sizeof(sums[0]));
        printf("Layer type: %d, num_neurons: %d\n", layer->type, layer->num_neurons);
        if (layer->type == LAYER_CONV) {
            // --- sparse-driven conv add into sums ---
            conv_sparse_q7_add(
                input[t],                        // bit-packed 28×28
                layer->conv_weights_col,        // [9][4] pre-transposed weights
                layer->conv_bias,               // [4]
                sums                             // [4×676]
            );
        }
        else if (layer->type == LAYER_FC) {
            // --- fully-connected sum over presynaptic spikes ---
        #if (Q07_FLAG)
            // bias broadcast
            vectorize_q7_add_to_q31(
                layer->bias,                    // [num_neurons]
                sums,
                layer->num_neurons
            );
        #else
            for (int j = 0; j < input_size; j++)
                sums[j] = dequantize_q07(layer->bias[j]);
        #endif

            // weight adds
            for (int byte_idx = 0; byte_idx < num_bytes; byte_idx++) {
                uint8_t byte = input[t][byte_idx];
                int     base = byte_idx * 8;

                while (byte) {
                    int bit = __builtin_ctz(byte);
                    int j   = base + bit;
                    byte   &= byte - 1;

                    if (j < input_size) {
                    #if (Q07_FLAG)
                        vectorize_q7_add_to_q31(
                            layer->weights[j],       // [num_neurons]
                            sums,
                            layer->num_neurons
                        );
                    #else
                        for (int i = 0; i < layer->num_neurons; i++)
                            sums[i] += dequantize_q07(layer->weights[j][i]);
                    #endif
                    }
                }
            }
        }
        // (Note: input layer simply leaves sums==0, then LIF will inject +1 per spike if needed)

/*
TODO
Add support for IF
Add support for vectorized float
Add compiler directives for IF and LIF with vectorization
Add compiler directives for float with vectorization
*/
    LIF_STEP:
        // decay: V = (DECAY_FP7 * V) >> DECAY_SHIFT
        vector_scale_q31(
            layer->membrane_potentials,
            layer->decay_rates[0],          // or DECAY_FP7
            DECAY_SHIFT,
            layer->membrane_potentials,
            layer->num_neurons
        );

        // add inputs: V += sums
        vectorize_q31_add_to_q31(
            sums,
            layer->membrane_potentials,
            layer->num_neurons
        );

        // threshold: mask = V >= thresh
        vector_compare_ge_q31(
            layer->membrane_potentials,
            layer->voltage_thresholds,
            layer->delayed_resets,
            layer->num_neurons
        );

        // reset: V[mask] -= thresh
        vector_sub_where_q31(
            layer->delayed_resets,
            layer->voltage_thresholds,
            layer->membrane_potentials,
            layer->num_neurons
        );

        // pack spikes out
        vector_pack_bits(
            layer->delayed_resets,
            output[t],
            layer->num_neurons
        );
    }
}

// void initialize_network(int neurons_per_layer[],
//      const int8_t weights_fc1[INPUT_SIZE][HIDDEN_LAYER_1], const int8_t weights_fc2[HIDDEN_LAYER_1][NUM_CLASSES],
//      const int8_t *bias_fc1, const int8_t *bias_fc2) {
//     snn_network.layers = static_layers;

//     if (!weights_initialized) {
//         for (int i = 0; i < INPUT_SIZE; i++) {
//             fc1_pointer_table[i] = (int8_t *)weights_fc1[i];
//         }
//         for (int i = 0; i < HIDDEN_LAYER_1; i++) {
//             fc2_pointer_table[i] = (int8_t *)weights_fc2[i];
//         }

//         fc1_bias_pointer = (int8_t *)bias_fc1;
//         fc2_bias_pointer = (int8_t *)bias_fc2;

//         weights_initialized = 1;
//     }

//     for (int l = 0; l < snn_network.num_layers; l++) {
//         snn_network.layers[l].layer_num = l;
//         snn_network.layers[l].num_neurons = neurons_per_layer[l];


//         if (l == 1) {
//             snn_network.layers[l].weights = fc1_pointer_table;
//             snn_network.layers[l].bias = fc1_bias_pointer;
//         } else if (l == 2) {
//             snn_network.layers[l].weights = fc2_pointer_table;
//             snn_network.layers[l].bias = fc2_bias_pointer;
//         } else {
//             snn_network.layers[l].weights = NULL;
//             snn_network.layers[l].bias = NULL;
//         }
// #if (Q07_FLAG)
//             memset(snn_network.layers[l].membrane_potentials, 0, snn_network.layers[l].num_neurons * sizeof(int32_t));
//             memset(snn_network.layers[l].delayed_resets, 0, snn_network.layers[l].num_neurons * sizeof(int32_t));
//             // Have to loop throught and not memset since memset doesnt work with non-zero values less than an integer size
//             for (int i = 0; i < snn_network.layers[l].num_neurons; i++) {
//                 snn_network.layers[l].voltage_thresholds[i] = VOLTAGE_THRESH_FP7;
//                 snn_network.layers[l].decay_rates[i] = DECAY_FP7;
//             }
// #else
//             memset(snn_network.layers[l].membrane_potentials, 0.0, snn_network.layers[l].num_neurons * sizeof(float));
//             memset(snn_network.layers[l].delayed_resets, 0.0, snn_network.layers[l].num_neurons * sizeof(float));
//             for (int i = 0; i < snn_network.layers[l].num_neurons; i++) {
//                 snn_network.layers[l].voltage_thresholds[i] = VOLTAGE_THRESH;
//                 snn_network.layers[l].decay_rates[i] = DECAY_RATE;
//             }
// #endif
//     }
// }

void initialize_network(int neurons_per_layer[],
    // new conv layer weights & bias:
    const int8_t conv1_weights_col[CONV1_KERNEL*CONV1_KERNEL][CONV1_FILTERS],
    const int8_t conv1_bias[CONV1_FILTERS],
    // final FC layer (from conv → output):
    const int8_t weights_fc2_data[CONV1_NEURONS][NUM_CLASSES],
    const int8_t bias_fc2[NUM_CLASSES])
{
    // Hook up our static layers array
    snn_network.layers = static_layers;

    // One-time pointer‐table init for FC2
    if (!weights_initialized) {
        // Build pointer table for FC2 (conv1 is used directly)
        for (int i = 0; i < CONV1_NEURONS; i++) {
            fc2_pointer_table[i] = (int8_t *)weights_fc2_data[i];
        }
        fc2_bias_pointer = (int8_t *)bias_fc2;
        weights_initialized = 1;
    }

    // Now configure each layer
    for (int l = 0; l < snn_network.num_layers; l++) {
        Layer *layer = &snn_network.layers[l];
        layer->layer_num    = l;
        layer->num_neurons  = neurons_per_layer[l];

        if (l == 0) {
            // Input layer
            layer->type             = LAYER_INPUT;
            layer->weights          = NULL;
            layer->bias             = NULL;
            layer->conv_weights_col = NULL;
            layer->conv_bias        = NULL;
        }
        else if (l == 1) {
            // Conv layer
            layer->type             = LAYER_CONV;
            layer->conv_weights_col = conv1_weights_col;
            layer->conv_bias        = conv1_bias;
            layer->weights          = NULL;
            layer->bias             = NULL;
        }
        else {
            // Output FC layer
            layer->type             = LAYER_FC;
            layer->weights          = fc2_pointer_table;
            layer->bias             = fc2_bias_pointer;
            layer->conv_weights_col = NULL;
            layer->conv_bias        = NULL;
        }

#if (Q07_FLAG)
        // Initialize all Q0.7 state
        memset(layer->membrane_potentials, 0,
               layer->num_neurons * sizeof(int32_t));
        memset(layer->delayed_resets, 0,
               layer->num_neurons * sizeof(int32_t));
        for (int i = 0; i < layer->num_neurons; i++) {
            layer->voltage_thresholds[i] = VOLTAGE_THRESH_FP7;
            layer->decay_rates[i]        = DECAY_FP7;
        }
#else
        // (float‐mode if you ever need it)
        memset(layer->membrane_potentials, 0,
               layer->num_neurons * sizeof(float));
        memset(layer->delayed_resets, 0,
               layer->num_neurons * sizeof(float));
        for (int i = 0; i < layer->num_neurons; i++) {
            layer->voltage_thresholds[i] = VOLTAGE_THRESH;
            layer->decay_rates[i]        = DECAY_RATE;
        }
#endif
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

void conv_sparse_q7_add(
    const uint8_t *in_bits,
    const int8_t  (*W)[CONV1_FILTERS],
    const int8_t  *bias,
    int32_t       *sums)
{
    const int K     = CONV1_KERNEL;
    const int H_in  = CONV1_H_IN,
              W_in  = CONV1_W_IN;
    const int H_out = CONV1_H_OUT,
              W_out = CONV1_W_OUT;
    const int F     = CONV1_FILTERS;
    const int patches = H_out * W_out;
    const int input_bytes = (H_in*W_in + 7)/8;

    // 1) zero & bias broadcast
    memset(sums, 0, F*patches*sizeof(int32_t));
    for (int f = 0; f < F; f++) {
        int32_t b = bias[f];
        int32_t *row = sums + f*patches;
        for (int p = 0; p < patches; p++)
            row[p] = b;
    }

    // 2) event-driven: for each input spike
    for (int byte_i = 0, bit_base = 0; byte_i < input_bytes; byte_i++, bit_base += 8) {
        uint8_t byte = in_bits[byte_i];
        while (byte) {
            int bit = __builtin_ctz(byte);
            byte &= byte - 1;
            int idx = bit_base + bit;
            int y0  = idx / W_in, x0 = idx % W_in;

            // slide 3×3 kernel
            for (int ky = 0; ky < K; ky++) {
                int out_y = y0 - ky;
                if (out_y < 0 || out_y >= H_out) continue;

                for (int kx = 0; kx < K; kx++) {
                    int out_x = x0 - kx;
                    if (out_x < 0 || out_x >= W_out) continue;

                    // compute flat k_idx & patch_base
                    int k_idx     = ky*K + kx;                   // 0..8
                    int patch     = out_y*W_out + out_x;         // 0..675
                    int32_t *sump = sums + (patch*F);            // &sums[patch*4]

                    // vectorized 4-at-a-time add:
                    vectorize_q7_add_to_q31(
                      &W[k_idx][0],    // srcA: int8_t[4] (weights for this k,pos)
                      sump,            // dst:   int32_t[4]
                      F                // 4
                    );
                }
            }
        }
    }
}
