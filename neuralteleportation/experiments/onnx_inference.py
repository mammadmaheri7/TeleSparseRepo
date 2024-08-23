import onnx
import onnxruntime as ort
import numpy as np

# Load the input data
input_data = np.load('input_data.npy')

# Load the ONNX models
model_1_path = 'model_weights_cob_activation_norm_teleported.onnx'
model_2_path = 'model_weights_cob_activation_norm.onnx'

# Verify the models (optional, checks for correctness of the model structure)
onnx_model_1 = onnx.load(model_1_path)
onnx_model_2 = onnx.load(model_2_path)

onnx.checker.check_model(onnx_model_1)
onnx.checker.check_model(onnx_model_2)

# Create ONNX Runtime sessions
ort_session_1 = ort.InferenceSession(model_1_path)
ort_session_2 = ort.InferenceSession(model_2_path)

# Get input name for the model (assume single input)
input_name_1 = ort_session_1.get_inputs()[0].name
input_name_2 = ort_session_2.get_inputs()[0].name

# Run inference on both models
output_1 = ort_session_1.run(None, {input_name_1: input_data})
output_2 = ort_session_2.run(None, {input_name_2: input_data})

# Since output_1 and output_2 are lists (for models with multiple outputs), extract the outputs
# Assuming both models have the same output structure
output_1 = output_1[0]
output_2 = output_2[0]

# Compare outputs
comparison = np.isclose(output_1, output_2, atol=1e-5)  # Adjust atol for your precision needs
difference = np.abs(output_1 - output_2)
max_difference = np.max(difference)
mean_difference = np.mean(difference)

# Display detailed comparison
print("Output comparison (element-wise):")
print(comparison)

print("\nDetailed Differences:")
print("Max difference:", max_difference)
print("Mean difference:", mean_difference)

print("\nIf differences exist, their values:")
if not np.all(comparison):
    print(difference[~comparison])

print("\nOutput 1:")
print(output_1)

print("\nOutput 2:")
print(output_2)
