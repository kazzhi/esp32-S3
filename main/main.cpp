/*
 * SPDX-FileCopyrightText: 2010-2022 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: CC0-1.0
 */
#define TF_LITE_STATIC_MEMORY

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h> // For memcpy


#include "model.h"  // Your quantized model
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <dirent.h>


#define IMAGE_WIDTH 64
#define IMAGE_HEIGHT 64
#define IMAGE_CHANNELS 1

constexpr int kTensorArenaSize = 0;

uint8_t* load_image_from_spiffs(const char* filename, int* width, int* height, int* channels) {
    FILE* f = fopen(filename, "r");
    if (f == NULL) {
        printf("Error opening image file: %s\n", filename);
        return NULL;
    }

    int comp; // Number of components in the image
    uint8_t* image_data = stbi_load_from_file(filename, width, height, &comp, *channels); // Force grayscale

    if (image_data == NULL) {
        printf("Error loading image data from file: %s\n", filename);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return image_data;
}

bool read_image_file(const char *filename, uint8_t *buffer) {
    char filepath[64];
    snprintf(filepath, sizeof(filepath), "%s/%s", IMAGE_DIR, filename);

    FILE *file = fopen(filepath, "rb");
    if (!file) {
        ESP_LOGE("SPIFFS", "Failed to open file: %s", filepath);
        return false;
    }

    size_t bytes_read = fread(buffer, 1, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS, file);
    fclose(file);

    if (bytes_read != IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS) {
        ESP_LOGE("SPIFFS", "Error: Read %d bytes instead of %d", bytes_read, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS);
        return false;
    }

    ESP_LOGI("SPIFFS", "Successfully read image: %s", filename);
    return true;
}

uint8_t tensor_arena[kTensorArenaSize];

void run_inference(tflite::MicroInterpreter *interpreter, uint8_t *image_data) {
    // Get model input tensor
    TfLiteTensor *input_tensor = interpreter->input(0);

    // Ensure input size matches
    if (input_tensor->dims->data[1] != IMAGE_WIDTH || input_tensor->dims->data[2] != IMAGE_HEIGHT) {
        ESP_LOGE("TFLM", "Model input size mismatch. Expected %dx%d, but got %dx%d",
                 IMAGE_WIDTH, IMAGE_HEIGHT, input_tensor->dims->data[1], input_tensor->dims->data[2]);
        return;
    }

    // Copy image data into input tensor
    memcpy(input_tensor->data.uint8, image_data, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS);

    int64_t start_time = esp_timer_get_time();
    
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE("TFLM", "Error running inference");
        return;
    }

    int64_t end_time = esp_timer_get_time();
    ESP_LOGI("TFLM", "Inference time: %lld microseconds", end_time - start_time);

    // Get model output tensor
    TfLiteTensor *output_tensor = interpreter->output(0);
    ESP_LOGI("TFLM", "Inference Output: %d", output_tensor->data.uint8[0]);  // Modify based on model output
}

void process_images(tflite::MicroInterpreter *interpreter) {
    struct dirent *entry;
    DIR *dir = opendir(IMAGE_DIR);
    if (!dir) {
        ESP_LOGE("SPIFFS", "Failed to open image directory: %s", IMAGE_DIR);
        return;
    }

    uint8_t image_buffer[IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS];

    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".bin")) { // Assuming binary image files
            ESP_LOGI("SPIFFS", "Processing image: %s", entry->d_name);
            if (read_image_file(entry->d_name, image_buffer)) {
                run_inference(interpreter, image_buffer);
            }
        }
    }

    closedir(dir);
}

extern "C" void app_main() {
    mount_spiffs();

    // Load TFLite Model
    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE("TFLM", "Model schema version mismatch");
        return;
    }

    // Set up the resolver
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D(esp_nn_conv2d);
    resolver.AddDepthwiseConv2D(esp_nn_depthwise_conv2d);
    resolver.AddSoftmax();
    resolver.AddFullyConnected();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();

    // Set up the interpreter
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE("TFLM", "Failed to allocate tensors");
        return;
    }

    ESP_LOGI("TFLM", "Model successfully loaded!");

    // Run inference on SPIFFS images
    process_images(&interpreter);
}


