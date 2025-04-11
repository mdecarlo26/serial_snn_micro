#include <Arduino.h>
#include "SnnLib.h"

Snn_Network snn_network;

// 3. Fully static memory for ping-pong buffers
static uint8_t ping_pong_buffer_1[MAX_NEURONS][BITMASK_BYTES] = {0};
static uint8_t ping_pong_buffer_2[MAX_NEURONS][BITMASK_BYTES] = {0};

// 6. Fully static memory for labels
char labels[NUM_SAMPLES] = {0};

// srand((unsigned int)time(NULL));



const float* weights_fc1[HIDDEN_LAYER_1];
const float* weights_fc2[NUM_CLASSES];



static uint8_t initial_spikes[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES] = {0};


void setup() {
    Serial.begin(96000);
    while (!Serial) {
        ; // Wait for serial port to connect. Needed for native USB port only
    }
    snn_network.num_layers = NUM_LAYERS;
    int neurons_per_layer[] = {INPUT_SIZE, HIDDEN_LAYER_1, NUM_CLASSES};

    for (int i = 0; i < HIDDEN_LAYER_1; i++) {
        weights_fc1[i] = weights_fc1_data[i];
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        weights_fc2[i] = weights_fc2_data[i];
    }

    initialize_network(neurons_per_layer, weights_fc1, weights_fc2, bias_fc1, bias_fc2);
    zero_network();

    labels[0] = label;

    rate_encoding_3d(input_data, NUM_SAMPLES, TIME_WINDOW, INPUT_SIZE, initial_spikes);
}

void loop() {
    int d = 0;
    int classification = inference(initial_spikes, ping_pong_buffer_1, ping_pong_buffer_2, d);
    Serial.print("Sample 0: Classification = ");
    Serial.print(classification);
    Serial.print(", Label = ");
    Serial.println(labels[d]);
    delay(1000); // Delay for 1 second before the next iteration
}