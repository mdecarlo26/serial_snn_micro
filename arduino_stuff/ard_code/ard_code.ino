#include <Arduino.h>
#include "SnnLib.h"

Snn_Network snn_network;



// 6. Fully static memory for labels
char labels[NUM_SAMPLES] = {0};

// srand((unsigned int)time(NULL));

int counter = 0;

static uint8_t initial_spikes[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES] = {0};


void setup() {
    // Serial.begin(96000);
    // while (!Serial) {
    //     ; // Wait for serial port to connect. Needed for native USB port only
    // }
    // Serial.println("--Starting SNN--");
    snn_network.num_layers = NUM_LAYERS;
    int neurons_per_layer[] = {INPUT_SIZE, HIDDEN_LAYER_1, NUM_CLASSES};

    initialize_network(neurons_per_layer, weights_fc1_data, weights_fc2_data, bias_fc1, bias_fc2);
    zero_network();

    labels[0] = label;

    rate_encoding_3d(input_data, NUM_SAMPLES, TIME_WINDOW, INPUT_SIZE, initial_spikes);
}

void loop() {
    unsigned long start = millis();
    int d = 0;
    int classification = inference(initial_spikes, d);
    unsigned long end = millis();
    counter++;

    // Serial.print("Iteration Num ");
    // Serial.println(counter);
    // Serial.print("Sample ");
    // Serial.print(d);
    // Serial.print(": Classification = ");
    // Serial.print(classification);
    // Serial.print(", Label = ");
    // Serial.println((int)labels[d]);
    // Serial.print("Elapsed time (s): ");
    // Serial.println((float)(end - start)/1000.0);
    delay(1000); // Delay for 1 second before the next iteration


}