#include <Arduino.h>                         // must come first
#include <TensorFlowLite.h>                  // Arduino wrapper for TFLM
#include "model.h"                           // from: xxd -i model.tflite > model.h

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// Tune this arena size to your model’s tensor requirements
constexpr int kTensorArenaSize = 10 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// TFLM globals
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = nullptr;
  tflite::AllOpsResolver resolver;
  tflite::MicroInterpreter* interpreter = nullptr;

  TfLiteTensor* input_tensor = nullptr;
  TfLiteTensor* output_tensor = nullptr;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  // 1. Load the model
  model = ::tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while(true);
  }

  // 2. Build the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // 3. Allocate memory for tensors
  interpreter->AllocateTensors();
  Serial.print("Input is ");
Serial.println(input_tensor->type == kTfLiteFloat32 ? "FLOAT32" :
               input_tensor->type == kTfLiteUInt8  ? "UINT8"  :
               input_tensor->type == kTfLiteInt8   ? "INT8"   :
                                                     "??");
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    while(true);
  }

  // 4. Obtain pointers to the I/O tensors
  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.println("TFLM interpreter ready!");
}

void loop() {
  // Fill input with dummy mid-point data (INT8) — replace with your real data
  for (int i = 0; i < input_tensor->bytes; i++) {
    input_tensor->data.int8[i] = input[i];
  }

  long int start = millis();
  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    delay(1000);
    return;
  }
  long int end = millis();

  // Print output bytes
  int label_idx = 0;
  int max_tmp = 0;
  Serial.print("Output (uint8): ");
  for (int i = 0; i < output_tensor->bytes; i++) {
    if (output_tensor->data.int8[i] > max_tmp){
      label_idx = i;
      max_tmp = output_tensor->data.int8[i];
    }
    
  }
  Serial.print(label_idx);
    Serial.print(' ');
  Serial.println();
  Serial.print("Inference time: ");
  Serial.print(end - start);
  Serial.println(" ms");
  delay(1000);
}
