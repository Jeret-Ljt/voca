# TFLite quantized inference example
#
# Based on:
# https://www.tensorflow.org/lite/performance/post_training_integer_quant
# https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor.QuantizationParams

import numpy as np
import tensorflow as tf

# Location of tflite model file (float32 or int8 quantized)
model_path = "ds_graph/deepspeech-0.5.0-models/output_graph.tflite"
# Processed features (copy from Edge Impulse project)
features = np.zeros([16, 19, 26])

  
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Allocate tensors
interpreter.allocate_tensors()

# Print the input and output details of the model
print()
print("Input details:")
print(input_details)
print()
print("Output details:")
print(output_details)
print()

# Convert features to NumPy array
np_features = np.array(features)

# If the expected input type is int8 (quantized model), rescale data
input_type = input_details[0]['dtype']
if input_type == np.int8:
    input_scale, input_zero_point = input_details[0]['quantization']
    print("Input scale:", input_scale)
    print("Input zero point:", input_zero_point)
    print()
    np_features = (np_features / input_scale) + input_zero_point
    np_features = np.around(np_features)
    
# Convert features to NumPy array of expected type
np_features = np_features.astype(input_type)

# Add dimension to input sample (TFLite model expects (# samples, data))
np_features = np.expand_dims(np_features, axis=0)

# Create input tensor out of raw features
interpreter.set_tensor(input_details[0]['index'], np_features)

# Run inference
interpreter.invoke()

# output_details[0]['index'] = the index which provides the input
output = interpreter.get_tensor(output_details[0]['index'])

# If the output type is int8 (quantized model), rescale data
output_type = output_details[0]['dtype']
if output_type == np.int8:
    output_scale, output_zero_point = output_details[0]['quantization']
    print("Raw output scores:", output)
    print("Output scale:", output_scale)
    print("Output zero point:", output_zero_point)
    print()
    output = output_scale * (output.astype(np.float32) - output_zero_point)

# Print the results of inference
print("Inference output:", output)
