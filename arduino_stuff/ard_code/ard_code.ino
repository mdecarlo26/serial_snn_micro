#include <Arduino.h>
#include "SnnLib.h"

Snn_Network snn_network;

// 3. Fully static memory for ping-pong buffers
char *ping_pong_buffer_1_data[MAX_NEURONS];
char *ping_pong_buffer_2_data[MAX_NEURONS];

char **ping_pong_buffer_1 = ping_pong_buffer_1_data;
char **ping_pong_buffer_2 = ping_pong_buffer_2_data;

static char ping_pong_buffer_1_blocks[MAX_NEURONS][TAU] = {0};
static char ping_pong_buffer_2_blocks[MAX_NEURONS][TAU] = {0};

// 6. Fully static memory for labels
char labels[NUM_SAMPLES] = {0};

// srand((unsigned int)time(NULL));



float* weights_fc1[HIDDEN_LAYER_1];
float* weights_fc2[NUM_CLASSES];



static char initial_spikes_data[NUM_SAMPLES][TIME_WINDOW][INPUT_SIZE] = {0};
char* initial_spikes_pointers_2d[NUM_SAMPLES][TIME_WINDOW];
char** initial_spikes[NUM_SAMPLES];


void setup() {
    Serial.begin(96000);
    while (!Serial) {
        ; // Wait for serial port to connect. Needed for native USB port only
    }
    snn_network.num_layers = NUM_LAYERS;
    int neurons_per_layer[] = {INPUT_SIZE, HIDDEN_LAYER_1, NUM_CLASSES};
    for (int i = 0; i < MAX_NEURONS; i++) {
        ping_pong_buffer_1_data[i] = ping_pong_buffer_1_blocks[i];
        ping_pong_buffer_2_data[i] = ping_pong_buffer_2_blocks[i];
    }

    for (int i = 0; i < HIDDEN_LAYER_1; i++) {
        weights_fc1[i] = weights_fc1_data[i];
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        weights_fc2[i] = weights_fc2_data[i];
    }

    initialize_network(neurons_per_layer, weights_fc1, weights_fc2, bias_fc1, bias_fc2);
    zero_network();
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < TIME_WINDOW; j++) {
            initial_spikes_pointers_2d[i][j] = initial_spikes_data[i][j];
        }
        initial_spikes[i] = initial_spikes_pointers_2d[i];
    }

    labels[0] = label;

    rate_encoding_3d(input_data, NUM_SAMPLES, TIME_WINDOW, INPUT_SIZE, initial_spikes);
}

void loop() {
    int classification = inference(initial_spikes[0], ping_pong_buffer_1, ping_pong_buffer_2);
    Serial.print("Sample 0: Classification = ");
    Serial.print(classification);
    Serial.print(", Label = ");
    Serial.println(labels[0]);
    delay(1000); // Delay for 1 second before the next iteration
}