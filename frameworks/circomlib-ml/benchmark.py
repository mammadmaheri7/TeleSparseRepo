import numpy as np
import tensorflow as tf

import subprocess, os, argparse, sys, re
import concurrent.futures, json, threading, psutil, time

import pandas as pd

from tensorflow.keras import layers, models
# copied from models_to_h5.ipynb
def ResNet20Cifar100(num_classes=100):
    from tensorflow.keras import layers, models
    inputs = tf.keras.Input(shape=(32, 32, 3))  # Assuming input image size is 32x32x3

    # Initial Conv Layer
    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='valid', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # # Layer 1 Block 1
    # residual = x
    # x = layers.Conv2D(16, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    # x = layers.Conv2D(16, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # # x = layers.Add()([x, residual])
    # x = layers.ReLU()(x)

    # # Layer 1 Block 2
    # residual = x
    # x = layers.Conv2D(16, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    # x = layers.Conv2D(16, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # # x = layers.Add()([x, residual])
    # x = layers.ReLU()(x)

    # # Layer 1 Block 3
    # residual = x
    # x = layers.Conv2D(16, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    # x = layers.Conv2D(16, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # # x = layers.Add()([x, residual])
    # x = layers.ReLU()(x)

    # # Layer 2 Block 1 (with downsampling)
    # residual = layers.Conv2D(32, kernel_size=1, strides=2, use_bias=False)(x)
    # residual = layers.BatchNormalization()(residual)
    # x = layers.Conv2D(32, kernel_size=3, strides=2, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    # x = layers.Conv2D(32, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # # x = layers.Add()([x, residual])
    # x = layers.ReLU()(x)

    # # Layer 2 Block 2
    # residual = x
    # x = layers.Conv2D(32, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    # x = layers.Conv2D(32, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # # x = layers.Add()([x, residual])
    # x = layers.ReLU()(x)

    # # Layer 2 Block 3
    # residual = x
    # x = layers.Conv2D(32, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    # x = layers.Conv2D(32, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # # x = layers.Add()([x, residual])
    # x = layers.ReLU()(x)

    # # Layer 3 Block 1 (with downsampling)
    # residual = layers.Conv2D(64, kernel_size=1, strides=2, use_bias=False)(x)
    # residual = layers.BatchNormalization()(residual)
    # x = layers.Conv2D(64, kernel_size=3, strides=2, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    # x = layers.Conv2D(64, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # # x = layers.Add()([x, residual])
    # x = layers.ReLU()(x)

    # # Layer 3 Block 2
    # residual = x
    # x = layers.Conv2D(64, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    # x = layers.Conv2D(64, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # # x = layers.Add()([x, residual])
    # x = layers.ReLU()(x)

    # # Layer 3 Block 3
    # residual = x
    # x = layers.Conv2D(64, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    # x = layers.Conv2D(64, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # # x = layers.Add()([x, residual])
    # x = layers.ReLU()(x)

    # Pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)

    # Create model
    model = models.Model(inputs, outputs)
    return model

# Create and compile model
model = ResNet20Cifar100()


p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

params = {"784_56_10": 44543,
            "196_25_10": 5185,
            "196_24_14_10": 5228,
            "28_6_16_10_5":5142,
            "14_5_11_80_10_3": 4966,
            "28_6_16_120_84_10_5": 44426}

accuracys = {"784_56_10": 0.9740,
            "196_25_10": 0.9541,
            "196_24_14_10": 0.9556,
            "14_5_11_80_10_3": 0.9556,
            "28_6_16_120_84_10_5": 0.9856}

arch_folders = {"28_6_16_10_5": "input-conv2d-conv2d-dense/",
                "14_5_11_80_10_3": "input-conv2d-conv2d-dense-dense/",
                "28_6_16_120_84_10_5": "input-conv2d-conv2d-dense-dense-dense/"}

def get_predictions_tf(model, test_images, batch_size=256):
    predictions = []
    for i in range(0, len(test_images), batch_size):
        batch = test_images[i:i+batch_size]
        pred = model.predict(batch, verbose=0)
        predictions.extend(np.argmax(pred, axis=1))
    return predictions

def transfer_weights(layers, model, scalar = 36):
    weights = []
    biases = []
    for ind in range(len(layers)-1):
        w = [[int(model.weights[ind * 2].numpy()[i][j]*10**scalar) for j in range(layers[ind+1])] for i in range(layers[ind])]
        b = [int(model.weights[ind * 2 + 1].numpy()[i]*10**(scalar * 2)) for i in range(layers[ind+1])]
        #b = [0 for i in range(layers[ind+1])]
        weights.append(w)
        biases.append(b)

    return weights, biases

def relu_mod(x):
    return x if x < p // 2 else 0

def DenseInt(nInputs, nOutputs, n, input, weights, bias):
    #print (len(input), nInputs)
    
    Input = [str(input[i] % p) for i in range(nInputs)]
    Weights = [[str(weights[i][j] % p) for j in range(nOutputs)] for i in range(nInputs)]
    Bias = [str(bias[i] % p) for i in range(nOutputs)]
    
    out = [0 for _ in range(nOutputs)]
    remainder = [None for _ in range(nOutputs)]
    
    for j in range(nOutputs):
        for i in range(nInputs):
            out[j] += input[i] * weights[i][j]
        out[j] += bias[j]

        remainder[j] = str(out[j] % n)
        out[j] = out[j] // n % p
        
    return Input, Weights, Bias, out, remainder

def prepare_input_json(layers, weights, biases, x_in, scalar=36, relu = False):
    relu_outs = []
    dense_weights = []
    dense_biases = []
    dense_outs = []
    dense_remainders = []
    x_ins = []

    out = x_in
    for ind in range(len(weights)):
        nInputs = layers[ind]
        nOutputs = layers[ind + 1]
        #print (nInputs, nOutputs)
        x_in, w, b, out, rem = DenseInt(nInputs, nOutputs, 10 ** scalar, 
                                     out, weights[ind], biases[ind])
        
        dense_outs.append(out)
        if relu:
            out = [x if x < p//2 else 0 for x in out]
            relu_outs.append([str(x) if x !=0 else 0 for x in out ])

        #print (out)
        dense_weights.append(w)
        dense_biases.append(b)
        
        dense_remainders.append(rem)
        x_ins.append(x_in)

    
    dense_outs = [[str(x) for x in sub] for sub in dense_outs]
        
    return x_ins[0], dense_weights, dense_biases, dense_outs, dense_remainders, relu_outs, np.argmax(out)

def monitor_memory(pid, freq = 0.01):
    p = psutil.Process(pid)
    max_memory = 0
    while True:
        try:
            mem = p.memory_info().rss / (1024 * 1024)
            max_memory = max(max_memory, mem)
        except psutil.NoSuchProcess:
            break  # Process has finished
        time.sleep(freq)  # Poll every second
        
    #print(f"Maximum memory used: {max_memory} MB")
    return max_memory

def execute_and_monitor(command, show = False):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(monitor_memory, process.pid)
        stdout, stderr = process.communicate()
        max_memory = future.result()
    if show:
        print(f"Maximum memory used: {max_memory} MB")
    return stdout, stderr, max_memory

def dnn_datasets():
    # Load TensorFlow MNIST data
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    np.random.seed(7)
    idx = np.random.permutation(len(test_images))
    test_images = test_images[idx]
    test_labels = test_labels[idx]

    # Normalize and flatten the images
    train_images_tf = train_images.reshape((-1, 28*28)) / 255.0
    test_images_tf = test_images.reshape((-1, 28*28)) / 255.0

    # Resize for 14 * 14 images
    train_images_tf_reshaped = tf.reshape(train_images_tf, [-1, 28, 28, 1])  # Reshape to [num_samples, height, width, channels]
    test_images_tf_reshaped = tf.reshape(test_images_tf, [-1, 28, 28, 1])

    # Downsample images
    train_images_tf_downsampled = tf.image.resize(train_images_tf_reshaped, [14, 14], method='bilinear')
    test_images_tf_downsampled = tf.image.resize(test_images_tf_reshaped, [14, 14], method='bilinear')

    # Flatten the images back to [num_samples, 14*14]
    train_images_tf_downsampled = tf.reshape(train_images_tf_downsampled, [-1, 14*14])
    test_images_tf_downsampled = tf.reshape(test_images_tf_downsampled, [-1, 14*14])

    return test_images_tf, test_images_tf_downsampled

def Conv2DInt(nRows, nCols, nChannels, nFilters, kernelSize, strides, n, input, weights, bias):
    Input = [[[str(input[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    Weights = [[[[str(weights[i][j][k][l] % p) for l in range(nFilters)] for k in range(nChannels)] for j in range(kernelSize)] for i in range(kernelSize)]
    Bias = [str(bias[i] % p) for i in range(nFilters)]
    out = [[[0 for _ in range(nFilters)] for _ in range((nCols - kernelSize)//strides + 1)] for _ in range((nRows - kernelSize)//strides + 1)]
    remainder = [[[None for _ in range(nFilters)] for _ in range((nCols - kernelSize)//strides + 1)] for _ in range((nRows - kernelSize)//strides + 1)]
    for i in range((nRows - kernelSize)//strides + 1):
        for j in range((nCols - kernelSize)//strides + 1):
            for m in range(nFilters):
                for k in range(nChannels):
                    for x in range(kernelSize):
                        for y in range(kernelSize):
                            out[i][j][m] += input[i*strides+x][j*strides+y][k] * weights[x][y][k][m]
                out[i][j][m] += bias[m]
                remainder[i][j][m] = str(out[i][j][m] % n)
                out[i][j][m] = str(out[i][j][m] // n % p)
    return Input, Weights, Bias, out, remainder

def AveragePooling2DInt (nRows, nCols, nChannels, poolSize, strides, input):
    Input = [[[str(input[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    out = [[[0 for _ in range(nChannels)] for _ in range((nCols-poolSize)//strides + 1)] for _ in range((nRows-poolSize)//strides + 1)]
    remainder = [[[None for _ in range(nChannels)] for _ in range((nCols-poolSize)//strides + 1)] for _ in range((nRows-poolSize)//strides + 1)]
    for i in range((nRows-poolSize)//strides + 1):
        for j in range((nCols-poolSize)//strides + 1):
            for k in range(nChannels):
                for x in range(poolSize):
                    for y in range(poolSize):
                        out[i][j][k] += input[i*strides+x][j*strides+y][k]
                remainder[i][j][k] = str(out[i][j][k] % poolSize**2 % p)
                out[i][j][k] = str(out[i][j][k] // poolSize**2 % p)
    return Input, out, remainder

def DenseInt_(nInputs, nOutputs, n, input, weights, bias):
    Input = [str(input[i] % p) for i in range(nInputs)]
    Weights = [[str(weights[i][j] % p) for j in range(nOutputs)] for i in range(nInputs)]
    Bias = [str(bias[i] % p) for i in range(nOutputs)]
    out = [0 for _ in range(nOutputs)]
    remainder = [None for _ in range(nOutputs)]
    for j in range(nOutputs):
        for i in range(nInputs):
            out[j] += input[i] * weights[i][j]
        out[j] += bias[j]
        remainder[j] = str(out[j] % n)
        out[j] = str(out[j] // n % p)
    return Input, Weights, Bias, out, remainder

def prepare_input_json_cnn(X, layers, input_path, scalar = 18):
    kernal_size = layers[-1]
    
    X_in = [[[int(X[i][j][0]*(10 ** scalar))] for j in range(layers[0])] for i in range(layers[0])]
    conv2d_1_weights = [[[[int(model.layers[1].weights[0][i][j][k][l]*(10 ** scalar)) for l in range(layers[1])] for k in range(1)] for j in range(kernal_size)] for i in range(kernal_size)]
    conv2d_1_bias = [int(model.layers[1].weights[1][i]*(10 ** (2 * scalar))) for i in range(layers[1])]

    X_in, conv2d_1_weights, conv2d_1_bias, conv2d_1_out, conv2d_1_remainder = Conv2DInt(layers[0], layers[0], 1, layers[1], kernal_size, 1, 10**18, X_in, conv2d_1_weights, conv2d_1_bias)

    out_conv2d_1 = layers[0] - kernal_size + 1
    relu_1_in = [[[int(conv2d_1_out[i][j][k]) for k in range(layers[1])] for j in range(out_conv2d_1)] for i in range(out_conv2d_1)]
    relu_1_out = [[[str(relu_1_in[i][j][k]) if relu_1_in[i][j][k] < p//2 else 0 for k in range(layers[1])] for j in range(out_conv2d_1)] for i in range(out_conv2d_1)]

    avg2d_1_in = [[[int(relu_1_out[i][j][k]) for k in range(layers[1])] for j in range(out_conv2d_1)] for i in range(out_conv2d_1)]

    _, avg2d_1_out, avg2d_1_remainder = AveragePooling2DInt(out_conv2d_1, out_conv2d_1, layers[1], 2, 2, avg2d_1_in)

    out_conv2d_1 = int(out_conv2d_1/2)
    #print (model.layers[4].weights[0])
    conv2d_2_in = [[[int(avg2d_1_out[i][j][k]) for k in range(layers[1])] for j in range(out_conv2d_1)] for i in range(out_conv2d_1)]
    conv2d_2_weights = [[[[int(model.layers[4].weights[0][i][j][k][l]*(10 ** scalar)) for l in range(layers[2])] for k in range(layers[1])] for j in range(kernal_size)] for i in range(kernal_size)]
    conv2d_2_bias = [int(model.layers[4].weights[1][i]*(10 ** (scalar * 2))) for i in range(layers[2])]

    _, conv2d_2_weights, conv2d_2_bias, conv2d_2_out, conv2d_2_remainder = Conv2DInt(out_conv2d_1, out_conv2d_1, layers[1], layers[2], kernal_size, 1, 10**scalar, conv2d_2_in, conv2d_2_weights, conv2d_2_bias)

    out_conv2d_2 = out_conv2d_1 - kernal_size + 1
    relu_2_in = [[[int(conv2d_2_out[i][j][k]) for k in range(layers[2])] for j in range(out_conv2d_2)] for i in range(out_conv2d_2)]
    relu_2_out = [[[str(relu_2_in[i][j][k]) if relu_2_in[i][j][k] < p//2 else 0 for k in range(layers[2])] for j in range(out_conv2d_2)] for i in range(out_conv2d_2)]


    avg2d_2_in = [[[int(relu_2_out[i][j][k]) if int(relu_2_out[i][j][k]) < p//2 else int(relu_2_out[i][j][k]) - p for k in range(layers[2])] for j in range(out_conv2d_2)] for i in range(out_conv2d_2)]

    _, avg2d_2_out, avg2d_2_remainder = AveragePooling2DInt(out_conv2d_2, out_conv2d_2, layers[2], 2, 2, avg2d_2_in)

    out_conv2d_2 = int(out_conv2d_2/2)
    flatten_out = [avg2d_2_out[i][j][k] for i in range(out_conv2d_2) for j in range(out_conv2d_2) for k in range(layers[2])]

    out_flat = out_conv2d_2 ** 2 * layers[2]

    dense_in = [int(flatten_out[i]) if int(flatten_out[i]) < p//2 else int(flatten_out[i]) - p for i in range(out_flat)]
    dense_weights = [[int(model.layers[8].weights[0][i][j]*(10 ** scalar)) for j in range(layers[3])] for i in range(out_flat)]
    dense_bias = [int(model.layers[8].weights[1][i]*(10 ** (scalar * 2))) for i in range(layers[3])]

    _, dense_weights, dense_bias, dense_out, dense_remainder = DenseInt_(out_flat, layers[3], 10**scalar, dense_in, dense_weights, dense_bias)


    in_json = {
        "in": X_in,
        "conv2d_weights": conv2d_1_weights,
        "conv2d_bias": conv2d_1_bias,
        "conv2d_out": conv2d_1_out,
        "conv2d_remainder": conv2d_1_remainder,
        "conv2d_re_lu_out": relu_1_out,
        "average_pooling2d_out": avg2d_1_out,
        "average_pooling2d_remainder": avg2d_1_remainder,
        "conv2d_1_weights": conv2d_2_weights,
        "conv2d_1_bias": conv2d_2_bias,
        "conv2d_1_out": conv2d_2_out,
        "conv2d_1_remainder": conv2d_2_remainder,
        "conv2d_1_re_lu_out": relu_2_out,
        "average_pooling2d_1_out": avg2d_2_out,
        "average_pooling2d_1_remainder": avg2d_2_remainder,
        "flatten_out": flatten_out,
        "dense_weights": dense_weights,
        "dense_bias": dense_bias,
        "dense_out": dense_out,
        "dense_remainder": dense_remainder
    }

    out = [int(x) for x in dense_out]

    if len(layers) >= 6:
        dense_1_weights = [[int(model.layers[10].weights[0][i][j]*(10**scalar)) for j in range(layers[4])] for i in range(layers[3])]
        dense_1_bias = [int(model.layers[10].weights[1][i]*(10**(scalar * 2))) for i in range(layers[4])]

        re_lu_3_out = [dense_out[i] if int(dense_out[i]) < p//2 else 0 for i in range(layers[3])]
        dense_1_in = [int(re_lu_3_out[i]) for i in range(layers[3])]

        _, dense_1_weights, dense_1_bias, dense_1_out, dense_1_remainder = DenseInt_(layers[3], layers[4], 10**scalar, dense_1_in, dense_1_weights, dense_1_bias)

        #dense_1_out = [str(x) for x in dense_1_out]

        new_entries = {
            "dense_re_lu_out":re_lu_3_out,
            "dense_1_weights":dense_1_weights,
            "dense_1_bias":dense_1_bias,
            "dense_1_out":dense_1_out,
            "dense_1_remainder":dense_1_remainder
        }
        # Append new entries to in_json
        in_json.update(new_entries)
        out = [int(x) for x in dense_1_out]

    if len(layers) >= 7:
        #print (layers[5], layers[4])
        #print ('model:', model.layers[12].weights)
        dense_2_weights = [[int(model.layers[12].weights[0][i][j]*(10**scalar)) for j in range(layers[5])] for i in range(layers[4])]
        dense_2_bias = [int(model.layers[12].weights[1][i]*(10**(scalar * 2))) for i in range(layers[5])]

        re_lu_4_out = [dense_1_out[i] if int(dense_1_out[i]) < p//2 else 0 for i in range(layers[4])]
        dense_2_in = [int(re_lu_4_out[i]) for i in range(layers[4])]

        _, dense_2_weights, dense_2_bias, dense_2_out, dense_2_remainder = DenseInt_(layers[4], layers[5], 10**scalar, dense_2_in, dense_2_weights, dense_2_bias)

        #dense_1_out = [str(x) for x in dense_1_out]

        new_entries = {
            "dense_1_re_lu_out":re_lu_4_out,
            "dense_2_weights":dense_2_weights,
            "dense_2_bias":dense_2_bias,
            "dense_2_out":dense_2_out,
            "dense_2_remainder":dense_2_remainder
        }
        # Append new entries to in_json
        in_json.update(new_entries)
        out = [int(x) for x in dense_2_out]

    out = [x if x < p//2 else 0 for x in out]
    pred = np.argmax(out)

    with open(input_path, "w") as f:
        json.dump(in_json, f)

    return pred

def prepare(model, layers):
    if layers[0] == 196:
        _, test_images = dnn_datasets()
    elif layers[0] == 784:
        test_images, _ = dnn_datasets()

    predictions_tf = get_predictions_tf(model, test_images)

    return predictions_tf, test_images

def prepare_cnn(model, layers, model_name=None):
    # check if model name is not None
    if model_name is not None and model_name=='resnet20':
        # TODO: load cifar100 dataset
        test_images = np.random.rand(256*2, 32, 32, 3)
    elif layers[0] == 14:
        _, test_images = cnn_datasets()
    elif layers[0] == 28:
        test_images, _ = cnn_datasets()

    predictions_tf = get_predictions_tf(model, test_images)

    return predictions_tf, test_images

def cnn_datasets():
    # Load TensorFlow MNIST data
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images_tf = train_images / 255.0
    test_images_tf = test_images / 255.0
    train_images_tf = train_images_tf.reshape(train_images.shape[0], 28, 28, 1)
    test_images_tf = test_images_tf.reshape(test_images.shape[0], 28, 28, 1)

    train_images_tf_14 = tf.image.resize(train_images_tf, [14, 14]).numpy()
    test_images_tf_14 = tf.image.resize(test_images_tf, [14, 14]).numpy()

    return test_images_tf, test_images_tf_14

def gen_model_dnn(layers, model_in_path):
    if len(layers) == 3:
        inputs = tf.keras.layers.Input(shape=(layers[0],))
        out = tf.keras.layers.Dense(layers[1], activation = 'relu')(inputs)
        out = tf.keras.layers.Dense(layers[2])(out)

        model = tf.keras.Model(inputs, out)

    elif len(layers) == 4:
        inputs = tf.keras.layers.Input(shape=(layers[0],))
        out = tf.keras.layers.Dense(layers[1], activation = 'relu')(inputs)
        out = tf.keras.layers.Dense(layers[2], activation = 'relu')(out)
        out = tf.keras.layers.Dense(layers[3])(out)

        model = tf.keras.Model(inputs, out)
    else:
        print ("Layers not Support")
        return None
    
    model.load_weights(model_in_path)
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    return model

# @ TODO: Hardcoded
def gen_model_cnn(layers, model_in_path):
    kernal_size = layers[-1]

    # Define the LeNet model in TensorFlow
    inputs = tf.keras.layers.Input(shape=(layers[0],layers[0],1))
    out = tf.keras.layers.Conv2D(layers[1],kernal_size, use_bias = True)(inputs)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.AveragePooling2D()(out)
    out = tf.keras.layers.Conv2D(layers[2],kernal_size, use_bias = True)(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.AveragePooling2D()(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(layers[3])(out)

    if len(layers) >= 6:
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Dense(layers[4])(out)

    if len(layers) >= 7:
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Dense(layers[5])(out)

    model = tf.keras.Model(inputs, out)

    model.load_weights(model_in_path)

    # Compile the model
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    return model

def load_csv():
    csv_path = '../../benchmarks/benchmark_results.csv'

    columns = ['Framework', 'Architecture', '# Layers', '# Parameters', 'Testing Size', 'Accuracy Loss (%)', 
            'Avg Memory Usage (MB)', 'Std Memory Usage', 'Avg Proving Time (s)', 'Std Proving Time', 'Notes']

    # Check if the CSV file exists
    if not os.path.isfile(csv_path):
        # Create a DataFrame with the specified columns
        df = pd.DataFrame(columns=columns)
        # Save the DataFrame as a CSV file
        df.to_csv(csv_path, index=False)
    else:
        print(f"File '{csv_path}' already exists.")

    df = pd.read_csv(csv_path)
    return df

def benchmark_dnn(test_images, predictions, weights, biases, layers, model_name, tmp_folder, input_path, zkey, veri_key, verify = False, save=False):
    loss = 0

    target_circom = "_".join(str(x) for x in layers) + '.circom'

    json_folder = tmp_folder + target_circom[:-7] + "_js/"
    wit_json_file = json_folder + "generate_witness.js"
    wasm_file = json_folder + target_circom[:-7] + ".wasm"
    input_path = tmp_folder + "input.json"
    wit_file = tmp_folder + "witness.wtns"

    mem_usage = []
    time_cost = []
    benchmark_start_time = time.time()

    for i in range(len(test_images)):
        cost = 0
        X = test_images[i:i+1]
        print ("process for image ",i)
        start_time = time.time()
        X_in = [int(x*1e36) for x in X[0]]
        x_in, dense_weights, dense_biases, dense_outs, dense_remainders, relu_outs, _ = prepare_input_json(layers, weights, biases, X_in, scalar=36, relu=True)

        in_json = {
            "in": x_in,
            "Dense32weights": dense_weights[0],
            "Dense32bias": dense_biases[0],
            "Dense32out": dense_outs[0],
            "Dense32remainder": dense_remainders[0],
            "ReLUout": relu_outs[0], 
            "Dense21weights": dense_weights[1],
            "Dense21bias": dense_biases[1],
            "Dense21out": dense_outs[1],
            "Dense21remainder": dense_remainders[1]
        }

        if len(layers) == 4:
            new_entries = {
                "ReLUout2": relu_outs[1],
                "Dense10weights": dense_weights[2],
                "Dense10bias": dense_biases[2],
                "Dense10out": dense_outs[2],
                "Dense10remainder": dense_remainders[2]
            }
            # Append new entries to in_json
            in_json.update(new_entries)

        with open(input_path, "w") as f:
            json.dump(in_json, f)



        commands = [['node', wit_json_file, wasm_file, input_path, wit_file],
                    ['snarkjs', 'groth16', 'prove', zkey, wit_file, tmp_folder+'proof.json', tmp_folder+'public.json']]

        for command in commands:
            # compute the time of the prove command (second command)
            if command == commands[1]:
                start_time = time.time()

            stdout, _, usage = execute_and_monitor(command)
            
            if command == commands[1]:
                time_prove = time.time() - start_time

            # print ('command:', command)
            # print (stdout)

            if "ERROR" in stdout:
                print ('command:', command)
                print (stdout)
                return
            cost += usage

        if verify:
            command = ['snarkjs', 'groth16', 'verify',veri_key, tmp_folder+'public.json', tmp_folder+'proof.json']
            subprocess.run(command)

        out = load_and_convert_json_to_int(tmp_folder+'public.json')
        out = [x if x < p//2 else 0 for x in out]
        pred = np.argmax(out)

        if pred != predictions[i]:
            loss += 1
            print ("Loss happens on index", i)

        mem_usage.append(cost)
        # time_cost.append(time.time() - start_time)
        time_cost.append(time_prove)
    
    print ("Total time:", time.time() - benchmark_start_time)

    layers = model_name.split("_")
    arch = "Input" + (len(layers)-1) * "-Dense"
    new_row = {
        'Framework': ['circomlib-ml (tensorflow)'],
        'Architecture': [f'{arch} ({"x".join(layers)})'],
        '# Layers': [len(layers)],
        '# Parameters': [params[model_name]],
        'Testing Size': [len(mem_usage)],
        'Accuracy Loss (%)': [loss/len(mem_usage) * 100],
        'Avg Memory Usage (MB)': [sum(mem_usage) / len(mem_usage)],
        'Std Memory Usage': [pd.Series(mem_usage).std()],
        'Avg Proving Time (s)': [sum(time_cost) / len(time_cost)],
        'Std Proving Time': [pd.Series(time_cost).std()]
    }

    new_row_df = pd.DataFrame(new_row)
    print (new_row_df)

    if save:
        df = load_csv()
        df = pd.concat([df, new_row_df], ignore_index=True)
        csv_path = '../../benchmarks/benchmark_results.csv'
        df.to_csv(csv_path, index=False)

    return

def benchmark_cnn(test_images, predictions, layers, model_name, tmp_folder, input_path, zkey, veri_key, save=False, verify = False):
    loss = 0

    if model_name == 'resnet20':
        target_circom = "./golden_circuits/resnet20.circom"
        json_folder = tmp_folder + "resnet20_js/"

    else:
        target_circom = "_".join(str(x) for x in layers) + '.circom'
        json_folder = tmp_folder + target_circom[:-7] + "_js/"

    wit_json_file = json_folder + "generate_witness.js"
    if model_name == 'resnet20':
        wasm_file = json_folder + "resnet20.wasm"
    else:
        wasm_file = json_folder + target_circom[:-7] + ".wasm"

    # IMPORTANT
    if model_name == 'resnet20':
        input_path = "./golden_circuits/final_input_resnet20.json"
        # combine circuit.json and output.json to create input_path.json
        circuit_json = json.load(open("./golden_circuits/circuit_resnet20.json"))
        input_json = json.load(open("./golden_circuits/output_resnet20.json"))
        final_json = {**circuit_json, **input_json}
        # add key "in" to final_json
        X = test_images[0]
        scalar = 18
        h = 32
        nChannels = 3
        # print shape of X
        print("X shape:", X.shape)
        # X_in = [[[int(X[i][j][0]*(10 ** scalar))] for j in range(h)] for i in range(h)]
        X_in = [[[int(X[i][j][k]*(10 ** scalar)) for k in range(nChannels)] for j in range(h)] for i in range(h)]
        Input = [[[str(X_in[i][j][k] % p) for k in range(nChannels)] for j in range(h)] for i in range(h)]
        # final_json["in"] = Input

        tmp = [[[5.2698129172783725e+17, 7.750428947251168e+17, 7.152722868762548e+17], [9.07657758381487e+17, 2.754033986626251e+17, 7.081363260302025e+17], [7.561117785279668e+17, 2.388378570519426e+17, 1.712196870895325e+17], [2.2894011519523638e+17, 7.684752159579995e+17, 9.734818220249738e+17], [9.167435064933158e+17, 7.538189333707511e+17, 6.378766292577887e+17], [5.2338335722853176e+16, 8.569385731439962e+17, 7.391486809439786e+17], [5.503001746652112e+17, 4.4522066263853574e+17, 2.7276385923817792e+17], [8.063442890757902e+17, 7.943756520144972e+17, 4.3634526105200864e+17], [6.204586051175665e+17, 5.194492131481318e+17, 7.353919096806839e+17], [6.351130952860873e+17, 8.242041729035414e+17, 7.039678265421943e+17], [6.403545132047631e+17, 9.25941884106042e+17, 8.213976833871292e+17], [3.881542227234902e+16, 1.1940200804152224e+17, 6.71959009611083e+17], [2.4781895165167434e+17, 2.188174353379534e+17, 9.87228018883116e+17], [7.982037213067764e+17, 2.8053773174016605e+17, 2.0158684949161933e+17], [3.17437721144053e+17, 8.80210676204319e+17, 6.496208690443547e+17], [7.924587385196266e+17, 6.079999747938962e+17, 4.3843343334663034e+17], [5.8258770867917416e+16, 5.7644446454884435e+17, 2.2449726221842435e+17], [7.56678043016884e+16, 1.823279277363452e+17, 5.3629765461065626e+17], [8.973891348871507e+17, 9.175227941799044e+17, 3.3027699110460954e+17], [2.5656724727043923e+17, 5.605145017124945e+17, 4.315793266765514e+17], [7.181843212028101e+17, 9.73839999655388e+17, 9.063422612714239e+17], [7.618097311804063e+17, 8.74654892827384e+17, 8.679327589959409e+17], [9.646413649026165e+16, 7.812275769348383e+17, 6.975194157399359e+17], [9.062838269220502e+17, 2.1887884311964934e+17, 2.3932157073328186e+17], [7.015019015317692e+17, 8.903660413042853e+17, 4.6723451555693075e+17], [5.131078722894559e+17, 4.107245054781381e+17, 4.8605905275059936e+17], [6.701609867957454e+17, 5.376977086113145e+17, 5.300562566747755e+17], [2.698466344052355e+17, 5.1357400036137946e+17, 8.731625961060541e+17], [3.4535490698848326e+17, 1.3918477698255604e+16, 9.533053182027465e+17], [3.7549423209692934e+17, 2.3647448249695846e+17, 3.908162603723305e+17], [8.356618475741876e+17, 8.860422812124104e+17, 4.360866130200548e+17], [7.79547588384647e+16, 4.72090434814182e+17, 5.671048160664124e+17]], [[4.533318803690017e+17, 5.0453182096050285e+17, 7.890459717394753e+17], [4.901555774550592e+17, 6.347385521143142e+17, 3.495007082044678e+17], [4.6415746528355336e+16, 7.56868094483209e+17, 8.034570581718454e+17], [7.464188254626491e+16, 3.757830996590438e+17, 2.979832054556384e+17], [4.9178639995822317e+17, 1.2267585689518467e+17, 4.155651062485124e+17], [2.542683019868167e+17, 8.452113329262422e+16, 9.658024553876596e+17], [7.009626918866929e+16, 4.533527994185882e+16, 5.664797473645473e+17], [8.956841675905605e+17, 6.112587881118981e+16, 2.7091609402298666e+17], [1.5366225048008365e+17, 4.571805321477598e+17, 7.98918339685556e+17], [9.652804701542291e+17, 1.7110013642344278e+17, 3.5033564055196774e+17], [1.7114232208052493e+17, 3.678848120407199e+17, 2.0322042994594614e+17], [2.261174670334376e+16, 8.486696604161156e+17, 6.745224565689134e+17], [1.751916244689844e+17, 4.411431885263708e+17, 3.240037451902285e+17], [4.1426059003081875e+17, 3.2101017984228954e+17, 2.5493700850386547e+17], [6.604695670667706e+17, 3.827261908095032e+17, 4.309412724475067e+17], [1.5710632776784317e+17, 7.358330260420622e+17, 3.157080392057043e+16], [9.51872316975515e+17, 8.611136263051485e+17, 6.851790129842345e+17], [7.597299366953142e+17, 5.668199507811892e+17, 6.4679368061003384e+16], [9.134936811335831e+17, 5.470019822466648e+17, 3.825806701306246e+17], [6.74545858756595e+17, 4.925819960233835e+16, 6.139209548652824e+17], [1.1632357327224352e+17, 2.5162585571151565e+17, 4.61299284955912e+17], [4.8716140760460826e+17, 2.112982310786592e+17, 4.2999961817221997e+17], [5.0243323546885555e+17, 6.36973677156956e+17, 2.230997462945842e+17], [1.4047088655943374e+17, 4.770538381146831e+17, 9.169683328652068e+17], [8.083233084470657e+17, 5.916128182322902e+17, 7.428502637574385e+17], [5.911641008264599e+17, 6.325378004232865e+17, 5.030073490067491e+17], [1.2035629824203253e+17, 6.035475440885633e+17, 1.5973369392275395e+17], [7.219780510255308e+17, 2.2627359488642028e+16, 1.0010842204890946e+17], [5.1995304935662765e+17, 9.370710614384637e+17, 2.7462485834239824e+16], [2.9225492883766246e+17, 8.62401192781506e+17, 7.276607886161134e+17], [2.935475754982726e+17, 3.3449881169851757e+17, 4.853569312832674e+17], [3.291922118209154e+17, 5.119881053523903e+17, 4.24376618356461e+17]], [[4.883424494101696e+16, 3.8717588832933357e+17, 7.038350952400954e+17], [7.208484263961636e+17, 4.010983852949857e+17, 8.785401867047729e+17], [7.586627173560576e+17, 3.5572500671632557e+17, 9.448090594093317e+17], [9.678196796632099e+17, 7.089852603478066e+17, 8.254030972759313e+17], [3.7550918810529344e+17, 1.8871573505263216e+17, 2.47168335951062e+17], [6.719250110676714e+17, 1.3891629655925154e+17, 8.84892982768725e+17], [3.601071033039713e+17, 7.553890651979853e+17, 4.471492264604614e+17], [6.588834983441402e+17, 7.591916786489677e+17, 9.414107843677359e+17], [2.5744528848990512e+17, 9.687473359989505e+17, 2.4360162286378374e+17], [5.459109440723755e+17, 3.725169360479581e+17, 2.5845883042282157e+17], [6.735246711968137e+17, 2.5903381774478883e+17, 6.122847016072832e+17], [8.341999040963735e+17, 4724287036361074.0, 3.722486973825261e+17], [5.304956993275911e+17, 6.149735249036086e+17, 1.1367288196061254e+17], [8.701650210305409e+17, 1.1047647104599112e+17, 9.10097353550526e+17], [4.230624037757559e+17, 6.978261581567313e+17, 6.894394535023016e+17], [2.6521532168627472e+17, 8.081654809185126e+17, 7.191768964051347e+17], [7.830893626370153e+17, 9.760967282263954e+17, 7.379343184032296e+17], [8.189880815807685e+17, 8.255017922768749e+17, 3.112783930797588e+17], [8.178110030412477e+17, 2.4106132057767104e+17, 3.929609012163311e+17], [8.22407994491128e+17, 4.8078128712790656e+17, 9.734343379126167e+17], [1.8090520342136208e+17, 6.060444473489873e+17, 4.965029864184197e+17], [7.291980090913185e+17, 9.244152766054383e+17, 1.4519109915200835e+17], [3.4034185976293286e+17, 6.609366699745631e+17, 1.62953940880846e+17], [6.840896735886405e+17, 2.248710987186665e+17, 5.020190873247019e+17], [6.546232745484521e+17, 3.565861675628705e+17, 3.2857746980745184e+17], [4.764334303211832e+17, 1.4360044067730104e+17, 2.290397952980059e+17], [2.4261334459895334e+17, 5.479963485338335e+17, 9.441607522408024e+16], [2.039001640638648e+17, 7.324550398431749e+17, 1.0199074433947964e+16], [4.790230403692555e+17, 4.212183135558344e+17, 2.8340417996830147e+17], [7.258128614419835e+17, 5.859339372192252e+17, 2.4797999581054163e+17], [3.279872489444119e+17, 8.510160490079784e+17, 4.9915477791238406e+17], [3.332810962501251e+17, 2.876804346218793e+17, 5.469093056801844e+17]], [[3.8497276004027994e+17, 1.1369941531548488e+17, 4.314662340903089e+17], [3.574398745800843e+17, 5.581315963041985e+17, 8.618251447638176e+17], [3.750697088364693e+17, 7.241785252951117e+16, 2.1485103931331984e+17], [4.2186757182751744e+17, 7.482771851530778e+16, 7.41689151151541e+17], [9.598493441166652e+17, 2.1110379019687066e+17, 7.03512370285099e+17], [4.5265191593933984e+17, 7.175922211285481e+17, 3.6161609253840264e+16], [1.2064293764534939e+17, 1.5948727407243955e+17, 2.4632223071578285e+17], [3.03014081805013e+17, 7.78379185071342e+17, 8.122543095166665e+17], [5.3664770062735443e+17, 9.758482283652851e+17, 7.4977820728793e+17], [2.662084587826482e+17, 3.3019396750501606e+17, 8.156525859509312e+17], [9.567209842718876e+17, 2.3209518802396234e+17, 4.0890360096510803e+17], [6.914854639402601e+17, 7.882810534195471e+17, 4.295247999269447e+16], [6.623079754379629e+17, 9101761714063294.0, 2.4807697057841504e+17], [7.604035347728943e+17, 9.511264414512562e+17, 8.614138753816348e+17], [2.6882843739868013e+17, 4.146407791278016e+17, 5.611618287853217e+17], [2.709080423344341e+17, 4.116830029150876e+17, 8.81634270114393e+17], [2.4337576843170096e+17, 8.019754060465737e+17, 8.933896720884323e+17], [1.2722015790227914e+17, 8.635541940984997e+17, 9.611954612386138e+17], [3.289578607945216e+17, 6.62656883049157e+17, 1.456975767914882e+17], [7.313898950423447e+17, 5.983957685468716e+17, 5.1403582195347475e+17], [3.115806550986077e+17, 4.409688359877775e+17, 5.866903968002145e+17], [5.825766163873356e+17, 1.3939736255799373e+17, 8.143628433884946e+17], [6.808807264101258e+16, 9.858347937686867e+17, 7.610370658977573e+17], [4.361124731003445e+17, 1.305796339355132e+17, 6.537684299303511e+17], [9.465097535851858e+17, 9.223514474730149e+17, 6.700525278250026e+17], [7.660890240414403e+17, 9.638608466060744e+17, 4.094528891162468e+17], [7.445024396815625e+17, 6.669176200523571e+17, 9.298083145809334e+17], [5.67694611585356e+17, 4.260898500724295e+17, 7.74524097193175e+17], [1.578127945596255e+17, 2.9721948997351136e+17, 9.889926645698396e+17], [6.83364399469047e+17, 6.538296552693988e+17, 8.447807539792142e+17], [2.1902909952924144e+17, 2.634928795281054e+17, 6.421933340399132e+17], [427129267528414.4, 6.51846716936077e+17, 7.313299063079715e+16]], [[5.907969655489085e+17, 3.8841217108373094e+17, 8.386892378945294e+17], [9.681401195481117e+17, 2.834388534255785e+17, 2.293452005047888e+17], [9.726769642690611e+17, 4.778385955156268e+17, 7.807772600699831e+17], [3.337378924038004e+17, 1.7144412958366272e+17, 1.6507153480418035e+17], [5.0594461072369856e+17, 8.759032666173085e+17, 3.237646130776853e+17], [4.15410158291747e+17, 6.536424850347729e+17, 9955471732194354.0], [3.0434324229296915e+17, 7.517314155084e+17, 8.28573457569165e+16], [6.386944212542326e+17, 8.234130239482024e+17, 4.768903617375439e+17], [5774500989777498.0, 8.193757964810673e+17, 7.214509881254235e+17], [2.383284735124791e+17, 4.201362546702973e+17, 9.207491260035766e+17], [6.961789779313167e+17, 7.165901937151674e+17, 4.707657300990874e+17], [8.811739752900899e+17, 6.454068738666084e+17, 1.6459628362827472e+17], [8.359961826297413e+16, 5.985619639232477e+17, 5.7645584331902266e+17], [9.995810000474875e+17, 3.392219981095621e+17, 8.100746960620564e+17], [6.800390473316635e+17, 2.171372546232213e+17, 5.287439701685115e+16], [9.309404619572259e+17, 2.3813553436502656e+17, 9.885997569806267e+17], [6.069263344604717e+17, 1.328597686566747e+17, 7.320396277834972e+17], [2.032117966850855e+17, 9.600314163686574e+17, 5.465502075907862e+17], [6.808439192733699e+17, 4.148754381169355e+17, 1.1678821329875533e+17], [8.778929007910586e+17, 1.7029717718626714e+16, 6.279605405822405e+17], [4.2989913573552915e+17, 5.470136059938775e+17, 3.599820169197217e+17], [3.6779515254392006e+17, 5.246851251236795e+16, 6.221441402379546e+17], [4.6338574915796435e+17, 6.819098908503485e+16, 5.856758040082e+16], [5.633328349541514e+17, 9.067070097027649e+17, 3.3839414359034148e+16], [8.068151012914956e+17, 2.6837051102220688e+17, 2.526982686614383e+17], [7.487877198803766e+17, 2.7974222468152256e+17, 8.748931491558024e+17], [7.716992905536431e+17, 6.277107296496187e+17, 1.7782500794648048e+16], [8.863186660964328e+17, 6.643088857835305e+17, 2.4014415745969885e+17], [3.906814579036851e+17, 8.00033330522028e+17, 4.9364781726979757e+17], [2.710328327944004e+17, 1.812606338194367e+17, 1.5666488403545786e+16], [1.8925297831897824e+17, 4.729900633753027e+17, 9.581828077627142e+17], [1.82986670895931e+17, 8.681596249029527e+17, 4.9965622127399514e+17]], [[8.322653440731387e+16, 4.941821473004855e+17, 3.041005282373085e+17], [6.588356355220367e+17, 4.528551513313385e+17, 4.862575064652164e+16], [4.371361496546432e+17, 2.9772890006955334e+17, 2.0271937850379174e+17], [3.0134587375406086e+17, 4.995024389159047e+17, 5.7006709833058797e+17], [5.776768533092283e+17, 5.897952731869876e+17, 2.1750738329647356e+16], [5.6267367127252685e+17, 7.446356218167703e+17, 7.730234121236024e+17], [9.051944311353576e+17, 1.2434687511542397e+17, 8.159436328048074e+17], [9.706258713640634e+17, 7.422778927359706e+17, 5.2319909387674656e+17], [4.0628604212533325e+17, 6.053263537642916e+17, 9.708700271582318e+17], [3.904146530935276e+17, 5.546045549326877e+17, 2.8395536741477744e+17], [7.952617493206213e+17, 8.419935372274684e+17, 7.563089154486764e+17], [4.5901942797987674e+17, 3.8092683756720166e+17, 6.111437067819556e+17], [6.636861387319004e+16, 8.973762140798091e+16, 1.132652664215772e+16], [5.6018608660417946e+17, 1.3628923768571933e+17, 9.719439151140178e+17], [8.543820240911876e+17, 4.64492144366376e+17, 8.859582617314237e+17], [7.949623409260698e+17, 1.2140390814294922e+17, 4.928407039187289e+17], [4.301836589362462e+16, 9.414771509468091e+17, 5.3049167275738586e+17], [8.887458400128276e+17, 6.264528482479959e+17, 5.653991576085998e+17], [5.108428147355365e+17, 2.5445978833222384e+17, 7.246059052956287e+17], [1.722500181479858e+17, 5.659319116369242e+17, 1.8550709951568246e+17], [7.29706973627675e+17, 3.096957789963802e+17, 4.3583251317283034e+17], [1.6781677675185757e+17, 4.700906417638772e+17, 2.968153180854043e+17], [2.4136358622721853e+17, 6.302318261923214e+17, 9.741829277820584e+17], [2.07630419901207e+17, 4.0698237732777466e+17, 3.123496564836511e+17], [9.599210662535547e+17, 2.64460203943789e+17, 4.4542511171374176e+17], [7.814742302922102e+17, 8.109065833246639e+17, 6.272118283675676e+17], [4.0028630981180736e+17, 6.800151817466546e+17, 1.0708092594650875e+17], [7.36467570210715e+17, 4.112810121704875e+17, 1.2651804375977038e+17], [7.964561662174735e+17, 2.574676134994407e+17, 9.080571945233737e+17], [1.8076739235352723e+17, 7.638919537330913e+17, 5.0264802989822944e+17], [4.392341370747443e+16, 1.8508734466639277e+17, 4.538562584916371e+17], [4.937347194399543e+17, 4.607133648396417e+17, 1.8539603251441827e+17]], [[5.6048487058427296e+17, 4.9376370289762003e+17, 3.362856277973104e+17], [3.1091733863080616e+16, 2.0444641546988986e+17, 3.33290779033858e+17], [1.6904233799792435e+17, 4.5540935803063597e+17, 1.2201715595539941e+17], [6.665135144878326e+17, 2.5099783042680067e+17, 1.569277258476983e+17], [6.411601458241445e+17, 3.7924741013905594e+17, 5.82951764303076e+17], [8.18212449832453e+17, 6.21093141364144e+17, 5.425125401172794e+17], [5.140804421883256e+17, 5.776161359178765e+17, 1.981434432550677e+17], [4.835462766232045e+17, 2.2245311658506896e+17, 9.409687941948385e+17], [6.94898473194753e+17, 2.57395818097089e+17, 4.042596560602537e+17], [2.095596479701941e+17, 4.951299520177107e+17, 3.547754162272768e+17], [7.028977113473865e+17, 3.603938355130142e+17, 7.10848219736338e+17], [7.992891042397748e+17, 4.1341733095242714e+17, 7.842454497846659e+17], [8.760639918690852e+17, 9.401545067744922e+17, 3.0191751975958746e+17], [3.341176088525977e+17, 4.6385366008210336e+17, 8.103611157187224e+17], [7.002272706756376e+17, 4.7929105467047386e+17, 3.454054103575872e+17], [8.842465976598222e+17, 6.757373674085563e+17, 7.422085005345026e+16], [1.4157114453799568e+17, 7.384081391133367e+17, 6.217485321802961e+17], [4.4315862554589715e+17, 6.602940010220704e+17, 4.235739767051536e+17], [7.444683134969829e+17, 5.4136775481337126e+17, 4.976650813139237e+17], [8.829207238118383e+17, 3.157946169289524e+17, 7.051149182316187e+17], [2.0951782226600035e+17, 9.001914636814582e+17, 3.953096742142993e+17], [3.532533634475759e+17, 7.122652989767671e+17, 3.1104218299603725e+17], [4.38082190637941e+17, 9.093286413034967e+17, 1.5645265860936285e+17], [8.222342516332687e+17, 4.089803629674744e+17, 4.3094321399989843e+17], [2.4301176461879293e+17, 2.5839804767049878e+17, 2.2893613379521494e+17], [6.415398020200306e+17, 5.65183436085353e+17, 9.326974464342487e+17], [6.446933470015625e+17, 8.64468235029757e+17, 2.182821485289226e+17], [9.945223850395978e+17, 3.64849762735416e+17, 2.626941911587679e+17], [1.1103533927146514e+17, 5.561036014821283e+17, 5.245172425905408e+17], [3.8640465332988115e+17, 3.8774960804866355e+17, 3.21656533313639e+17], [2.0892882079868048e+17, 8.597818439119365e+16, 8.801940128375782e+17], [6.124160411117806e+17, 8.81181145097691e+17, 9.34225662700083e+17]], [[6.787250632866084e+17, 6.102339970758088e+17, 2.6208486262535146e+17], [2.2899292126902106e+17, 7.374343512447903e+17, 6.335914236536773e+17], [9.386047695586673e+17, 7.279225394537425e+17, 4.196696825891488e+17], [3.699742330368827e+17, 6.780187692417702e+17, 9.822963604394739e+17], [8.651798002810738e+17, 2.5891640884211686e+17, 1.5903994520663955e+17], [6.866272908073156e+17, 8.994189653395368e+16, 9.034142488749386e+17], [8.827070639986762e+17, 1.221696946898906e+17, 7.326128993981111e+17], [8.905410397950267e+17, 7.853076655440579e+17, 3.328029402895533e+17], [6.673870760586131e+17, 4.721719451981856e+17, 3.960342094664642e+17], [8.035761702976877e+17, 8.782734165918376e+17, 8.286509420312257e+17], [1.7653716346671533e+17, 4.315935074153374e+17, 5.7279925622695795e+17], [6.814890934713988e+17, 6.042478589885201e+17, 9.523680209745088e+17], [4.210262751039155e+17, 7.442437592472311e+17, 8.37053767572335e+17], [7.601522965455896e+17, 5.958327529548411e+17, 7.029554844652751e+17], [1.2533599947447461e+17, 8.159005546849274e+17, 6.794219214672462e+17], [6.06538837101617e+17, 1.5140821402806026e+17, 7.776800638833773e+17], [9.641340594505073e+17, 8.071464467632315e+17, 1.147790965434412e+17], [1.5755947990934928e+17, 2.4791780465662573e+17, 4.388206594898608e+17], [9.173092778411645e+17, 4.153152015586249e+17, 6.694082979484095e+17], [1.972646865771199e+16, 5.329515484908116e+17, 1.533582766611491e+17], [3.035267538117558e+17, 1.3090992178957627e+17, 8.769516011228608e+17], [7.603052753574771e+17, 1.2532392278400494e+17, 1.2387529547330134e+17], [7.022042228082836e+17, 7.286087285450071e+17, 9.656572252474419e+17], [1.3052093643305507e+17, 8.942044214623044e+17, 3.828553169260528e+17], [9.91143267056019e+17, 4.921018454810657e+17, 8.352035651533175e+17], [2.179887196073305e+16, 9.861018031435686e+16, 5.737035856515094e+17], [1.4067192145381979e+17, 8.11743823080279e+17, 3.416096082477408e+17], [3.78934569051355e+17, 4.6831768149141e+17, 5.3170590819646304e+17], [5.52403272552676e+17, 8.596481405895889e+17, 7.939568872389514e+16], [2.4044374453188522e+17, 6.719255113517293e+17, 2.0253175472015184e+17], [8.763419842248484e+17, 9.38168117275946e+16, 5.926698822416339e+17], [7.117733016576172e+17, 4.575881139236202e+17, 7.731198018560872e+17]], [[8.99366563046522e+17, 7.33629803625314e+17, 5.976143850798899e+17], [4.130402300639381e+17, 6.780905367578066e+17, 1.0538090343559725e+17], [5.519451922370707e+17, 1.0979010795973298e+16, 8697584210830334.0], [7.188684022655881e+17, 2.9813476824054874e+17, 7.402405257316259e+17], [9.931813974967506e+17, 5.155546241001253e+17, 2.6764971152975824e+17], [9.466192568929848e+17, 2.710317857300316e+17, 2.2225063274658595e+17], [4.394103268225056e+17, 8.744859914082198e+17, 9.51760352581575e+17], [1.6684626918008928e+17, 1.4039608756590683e+17, 6.568517454462519e+16], [1.5001734299461712e+17, 4.352573598926892e+17, 3.1222109963815616e+17], [4.614094560664095e+17, 8.123552005138204e+17, 8.528101070494817e+17], [2.181876739176438e+17, 4.215908309270571e+17, 8.464724907156522e+17], [5.495002116031546e+17, 6.054545788615325e+17, 6.2135728787222e+17], [4.649346211475985e+17, 6.410489822146216e+17, 6.385270993641325e+17], [3.974419672114598e+17, 2.908436390954795e+17, 6.691752600896937e+17], [4.712122018595839e+17, 2.2478675307692054e+17, 5.188234034154538e+17], [5.708829906293111e+16, 3.464471473883095e+17, 3.1752752245113946e+17], [7.707498429635276e+17, 7.002180030695512e+17, 5.245572456773244e+17], [4.255373146394647e+16, 7.961620545162147e+17, 1.995890739770334e+17], [3.0638722833400224e+17, 3.370707320300893e+17, 3.937976583265327e+17], [8.634876385327612e+17, 8.020438163965115e+16, 4.938574513909333e+17], [3.4864223986385536e+17, 4.591456542857468e+17, 7.452437005161491e+17], [1.0627902965199066e+17, 4.255265618017823e+17, 3.210762082218749e+17], [2.7567014589456685e+17, 9.940223652592562e+17, 4.14630083751109e+17], [7.01660941010615e+17, 4.4003513490876314e+17, 2.178153066819869e+17], [4.36224604607657e+17, 7.914301421702263e+17, 2.9044737425176294e+17], [8.510999034896494e+17, 1.653745099858004e+17, 5.332148039919401e+17], [5.3691732509862675e+17, 8.579471876758452e+17, 3.061694841582284e+17], [7.318533720448618e+17, 1.786021415713931e+17, 9.89349559449562e+17], [5.4454748229678483e+17, 2.4340746323154848e+17, 8.314237407142853e+17], [4.506151301560024e+16, 8.065435058137111e+17, 8.836447762098184e+17], [9.35286320941193e+17, 4.38036871430935e+17, 8.124128983909377e+17], [9.927399522965654e+17, 3.3759315664624644e+16, 5.2064219659611496e+16]], [[5.241159666589893e+17, 3.1720044458262596e+16, 5.977197101392187e+17], [4.6889352073854086e+17, 7.843044957464251e+17, 6.611032846423149e+17], [7.603922162612256e+17, 3.948984738430339e+16, 2.1223118677007814e+17], [6.181811859366543e+17, 4.021855313054075e+17, 4.0840737820495475e+17], [9.348414521103244e+17, 7.187160750582445e+17, 3.939418582393694e+16], [5.085990079756419e+17, 8.00217718639817e+17, 9.576018317396276e+17], [2683881084904360.5, 6.264524875755217e+17, 6.375797404312977e+17], [8.035071097484933e+17, 4.775770881455863e+17, 2.6720835066228944e+17], [3.2047124192099283e+17, 2.4361122120307987e+17, 9.835349912839913e+17], [3.151957367774851e+17, 7.671111022580259e+17, 1.849008931715622e+17], [4.9198001593362496e+17, 1.2022639782964206e+17, 9.038463855490214e+17], [9.619641792633181e+17, 7.073423882363423e+17, 3.5391784295586125e+17], [4.287461584582439e+17, 2.964004782192331e+17, 5.734981063316992e+17], [5.0916781384548474e+17, 1.6104166288456256e+17, 8.330894806296141e+17], [8.26280638437236e+17, 5.0991391887182944e+17, 4.718498148488082e+16], [6.344894062454228e+17, 7.904358411791951e+17, 1.380865747684269e+17], [1.8629997414720288e+17, 5.667855550479688e+17, 2.6249980077476288e+17], [6.284622060561555e+17, 5.720004136997824e+17, 4.133760730205849e+16], [8.137551194766636e+17, 4.395393898596395e+17, 7.090698451530587e+17], [7.631584076025417e+17, 3.7052933439455046e+17, 9.912572930643852e+17], [2.534516804942336e+17, 3.3328739365121372e+16, 2.5733462918090976e+16], [2.500218081929325e+17, 3.720540640493621e+17, 1.101427050558741e+17], [8.911070263914385e+17, 3.542308808646597e+17, 4.2373887688982906e+17], [9.510005701621233e+17, 9.49398051692574e+17, 4.3318276703330496e+17], [4.2952695598581005e+17, 3.0183576183772966e+17, 5.619692935219822e+16], [8.959770304742579e+17, 9.2170658974704e+16, 5.497937773941415e+17], [9.924917215688e+17, 1.6159113789348035e+17, 9.959371275152822e+17], [1.4417027499297952e+17, 7.116885011311041e+17, 1.6547086963088675e+17], [9.515528709732959e+17, 5.988942474293107e+17, 6.899767607501541e+17], [1.0999769054751574e+17, 6.246927041434456e+17, 9.255483211245274e+17], [2.897550879135391e+17, 1.925459166162221e+17, 3.464777351538248e+17], [4.2605881945450586e+17, 7.959818418755813e+17, 3.516023306208374e+17]], [[9.994971737532554e+16, 7.191578381035548e+17, 7.800471111292307e+16], [2.330072013473976e+17, 9.593866102727864e+17, 8.113798775494569e+17], [5.733327549270949e+17, 5.0457940941029786e+17, 9.816546377569742e+17], [3.31668206482736e+17, 9.090888211916582e+17, 8.859847022331889e+17], [5.871980808546029e+17, 1.7478663228666413e+17, 3.4787319487005965e+17], [6.267480911468874e+17, 4.874281155215971e+17, 1.7575114189703667e+17], [4.3903630363421875e+17, 8.767789527264251e+17, 9.850875390386141e+17], [5.809662960189494e+17, 2.0150875137093728e+17, 9.514046384857933e+16], [3.692784387511753e+17, 8.307604573383998e+16, 8.129578080125576e+17], [2.8553064208510282e+17, 3.970699560799179e+17, 4.330058467895711e+17], [5.015499396124822e+17, 5.865301356646074e+17, 4.230138146379043e+17], [4.49171221753371e+17, 7.667573478187542e+17, 5.462755713464853e+17], [5.77518565614359e+17, 1.0358159676768363e+17, 3.300129576770776e+17], [8.826459292330612e+17, 9.975418496190967e+17, 5.866146026591279e+17], [3.859499317229753e+17, 9489209717313818.0, 4.441428731339456e+17], [3.024608254495824e+17, 9.297793529633814e+17, 6.327153639214202e+17], [4.8630442142210074e+17, 7.977183154386673e+17, 9.31487881641552e+17], [1.896074545494718e+17, 9.449052417911978e+17, 9.2957070901956e+17], [1.093917612982529e+17, 6.891213096361463e+17, 2.2925726917103683e+17], [8.47632003948105e+17, 3.398956789525529e+17, 9.70167091631412e+17], [8.065415730490211e+16, 1.1587016866294219e+17, 6.611077943839713e+17], [7.462009156656695e+17, 5.734070028407662e+17, 9.092952823090232e+17], [5.800070329912817e+17, 4.509016084035783e+17, 4.7642716745604563e+17], [6.271067623242158e+17, 2.771402367042861e+17, 3.991377039593751e+17], [1.941111883711829e+17, 8.356071337364618e+17, 2.3288012654754365e+17], [9.830492065906744e+17, 8.882828835329155e+17, 2.5245518747326733e+17], [6.53392187500254e+17, 8.119633350694277e+17, 3.077489576584098e+17], [1.941203169901058e+17, 6.724881874505641e+17, 2.4832344013051856e+17], [3.859799313184489e+17, 9.624179198284443e+17, 1.1910934300012344e+17], [4.078242536075738e+16, 9.452419999709641e+17, 4.700617176344871e+17], [2.6249489145048288e+17, 3.2291708579333677e+17, 8.936653919732264e+17], [5.0527409064460525e+17, 8.805004782374482e+17, 8.430762379813923e+17]], [[8.949690877223791e+17, 4.5031885191180166e+17, 9.524358342635747e+17], [8.365458269996369e+17, 2.2135798259765494e+17, 2.932297739056222e+16], [5.116299497334589e+17, 9.831423686808909e+17, 3.7460385380410477e+17], [8.979848651257126e+17, 3.774408504804606e+17, 6.833838332314289e+17], [4.6143911020398995e+17, 1.5102958773765062e+17, 9.367337766852154e+17], [5.14209335993299e+17, 4.0029562399261645e+17, 5.572838594419692e+17], [2.187194020130745e+17, 7.605273981223334e+17, 1.4525486888916182e+17], [3.1159936009328384e+17, 3.3491995611711635e+17, 8.757893025705953e+17], [2.640666711180194e+17, 8.227061857200302e+17, 5.6147266321728806e+17], [4.066599840066789e+17, 7.153080153617484e+17, 9.914049076353743e+17], [4.932035668915097e+17, 1.185439577075309e+16, 9.954650042516814e+17], [2.9679005766975354e+17, 3.599743035511088e+17, 4.211257754875173e+17], [6.422196230750766e+17, 5.044116197474225e+17, 9.417802195807213e+17], [2.3125911775488595e+17, 7.388475664019346e+16, 3.300447051972502e+17], [9.692701622782249e+17, 2.9275934850355046e+17, 6.028300565696072e+17], [3.90557799905679e+17, 2.1839034022079318e+17, 6.103014864499478e+16], [6.558263498696954e+17, 2.410203649507009e+17, 4.778894994487931e+17], [2.1372468890363504e+17, 9.435926178104e+17, 2.5920067637902954e+17], [6.808297730038995e+17, 5.3508997274981216e+17, 6.377392969620831e+17], [2.386435512036571e+17, 3.710425062256759e+17, 9.754087409140696e+17], [8.72059297219519e+17, 2.9221811701490976e+17, 8.221367797886454e+17], [4.19116995606331e+17, 2.928781990876472e+17, 7.522430304616367e+17], [7.422804059592453e+17, 5.914192526535854e+17, 4.336472264651622e+17], [2.4354118770530253e+17, 4.19335682051186e+17, 7.303801454758491e+17], [7.388147048782298e+17, 6.775662186618877e+17, 1.9571479915233892e+16], [6.899875960912205e+16, 6.37855783452632e+17, 6.265367876205724e+17], [5.205429796516204e+17, 1.9384770299780774e+17, 8.536218059391862e+17], [1.2982042757679446e+17, 9.675465800407676e+17, 7.179461986879676e+17], [9250514972021006.0, 8.338936459788645e+17, 3.815043366241099e+17], [9.859502914118593e+17, 9.947852585287585e+17, 9.42862890693548e+17], [9.273144133493898e+17, 4.6865281943478886e+17, 4.568912752691492e+17], [2.5246612538516346e+17, 4.8672587640148314e+17, 4.6046679262278714e+17]], [[8.45023269143976e+17, 6.96569092939526e+17, 9.022622117924908e+17], [1.8778522022781574e+17, 8.90317688544445e+17, 8.541128013838184e+17], [1.3508905865008669e+17, 2.518400835338881e+17, 2.6330267360140912e+17], [2.2105543357513325e+17, 5.073569723314159e+17, 6.861384098450231e+17], [5.968347787558698e+17, 6.04709158742186e+16, 1.5299450051953456e+17], [1.5850161993671898e+17, 8.720103923917853e+17, 2.827265189773035e+17], [9.313625853629747e+17, 4.8700773376041574e+17, 9.870062896407419e+17], [8.400665023051005e+17, 3.834886675499462e+17, 1.5052501464349622e+17], [7.47055341846407e+17, 1.6668305132291117e+17, 7.34421940695858e+17], [5.214651270752457e+17, 2.241850377935354e+17, 8.455897494521368e+17], [3.869780860600608e+17, 5.90862486079459e+17, 7.4235958941193e+17], [4.265931233291257e+17, 5.914678739658875e+17, 1.2912322821394395e+17], [9.096324385156333e+17, 1.451116643241578e+17, 9.523174279539228e+17], [4.0212970207048506e+17, 5.993917660381096e+17, 6.349758497588009e+17], [9.191002642643841e+17, 2.4638466903872948e+16, 6.453897550200138e+17], [9.874356793286638e+17, 9.607491696696282e+17, 5.823384711466332e+17], [9.88330605025312e+17, 7.901455759058949e+17, 5.038413911669104e+17], [6.151546896216617e+17, 4.8533452387907085e+17, 8.998736169784492e+17], [8.571921132654346e+17, 4.1906942048819264e+17, 1.1619012875848755e+17], [7.641016919188529e+17, 5.5440640076526483e+17, 8.60709262813009e+17], [4.626724001683301e+17, 3.7173314009342016e+17, 2.7562014235674458e+17], [2.026990686298811e+17, 1.2393696325428416e+16, 6.122913712284017e+17], [4.03322064050723e+17, 2.5362203547053984e+17, 5.88594751809297e+17], [3.629018795166331e+17, 2.2379044716082707e+17, 9.301510840028758e+17], [4.117570380096258e+17, 9.114630831769802e+17, 3.509143113079305e+16], [5.837491203722239e+17, 2.4741418336287846e+17, 9.346870084724411e+17], [1.9788821543552947e+17, 2.1350931694127674e+17, 6.262993994553475e+17], [2.4202811029077942e+17, 8.808360323683456e+17, 6.801936573252337e+17], [7.276534602410725e+17, 4.548225866403429e+17, 3.7293578329134426e+17], [1.738770346580589e+17, 9.65857528042332e+17, 4.177820793803199e+17], [1.0548234364210262e+16, 7.099934656176582e+17, 9.491926637463037e+17], [3.314704698574915e+17, 8.558207308413687e+17, 9.27347730921214e+17]], [[6.274241640194966e+16, 9.080302074269102e+17, 8.537807963265221e+17], [9.626896360807899e+17, 4.212782016314499e+17, 3.222692544195263e+17], [3.415234214640449e+17, 3.057645499873075e+17, 2.8377107742333462e+17], [5.99363505110614e+17, 2.0464370840577616e+16, 4.08893290941763e+16], [1.5050710331713124e+16, 4.6416952027169344e+17, 2.6788749651645962e+17], [3.329070270625045e+17, 9.861468092031044e+17, 5.1199602915091046e+17], [5.878054614186052e+17, 8.723464506974272e+17, 5.400711411688347e+17], [7.997329182297548e+17, 3.501565948530092e+17, 7.446957980279753e+17], [3.5089766121707332e+16, 7.877188083461213e+17, 7.387917762881737e+17], [1.1105963394091645e+17, 1.9462915444438112e+17, 5.472497170762339e+17], [4.134918397485897e+17, 8.943722596939145e+17, 1.1354206043885683e+17], [8.302848501488759e+17, 9.267755689707204e+17, 5.4933293601112134e+17], [3.0451929573816115e+17, 1.5509431161752397e+17, 2.694655853295189e+17], [2.390760018499626e+17, 8.780225841057407e+17, 9.23141174078898e+17], [2.4228175742203683e+17, 8.882978695083649e+17, 6.674518012350228e+17], [1.9309091549798608e+17, 9.307277689660164e+17, 8.883101100742462e+16], [1.9102340640089456e+17, 1.178184409272477e+17, 6.652814807099233e+17], [9.158553249054368e+17, 4.2667088110113485e+17, 8.078735881134664e+17], [5.077799871742672e+16, 8.264585305526988e+17, 5.179239555141859e+17], [7.744409980857362e+17, 9.120004696591206e+17, 6.677200216265242e+17], [1.926863106593506e+17, 1.2873690112006741e+17, 2.8138305774260896e+17], [4.9078864540980915e+17, 8.253276425321021e+17, 9.64064815072822e+17], [4.1436480505585016e+16, 6.904251183102692e+17, 3.191142424239718e+17], [3.757354983546348e+17, 2.082716626467853e+17, 7.631605306984852e+17], [7.915230764132151e+17, 2.5654872838053766e+17, 4.709901792774178e+17], [7.015693850937252e+17, 1.9813258473801776e+17, 3.989197508473441e+16], [5.0962891370133146e+17, 5.0213897714826906e+17, 6.161307601581381e+17], [4.310075429704716e+17, 8.95955251118301e+17, 1.226349761535156e+17], [4.402198684366718e+17, 8.666434055239697e+17, 3.256034668520068e+17], [1.9407244181497728e+17, 3.2764968517781357e+17, 2.9677494899356115e+17], [1.9044782243602608e+17, 2.0358138009919923e+17, 4.5802503995398534e+17], [1.3522814945578144e+17, 4.257776782931699e+17, 9.563829984027683e+17]], [[9.340928798937672e+17, 6.694726843124099e+17, 3.3417394041772045e+17], [4.3204784936808904e+16, 4.448847897665579e+17, 4.782943819883464e+17], [1.5015312413894266e+17, 3.5279142318649005e+17, 6.677776734995698e+17], [1.4650628501105856e+17, 6.670176390748713e+17, 2.523831102442553e+17], [8.429873709511426e+16, 8.856191385062331e+17, 7.278862516270879e+17], [1.3159808477781576e+17, 4.9546220135570176e+17, 3.290421695222704e+17], [1.561403579266274e+17, 8.819694476602452e+17, 7.988075239562863e+17], [3.684687671859399e+17, 5.49816035163364e+17, 1.3341308203163605e+17], [1.6060199463998826e+17, 1.2140230218812542e+17, 1.6817417275816947e+17], [6.290296339982338e+16, 2.9553025198636986e+17, 4.807901008907248e+17], [6.681871613435546e+17, 5.268267465773324e+17, 2.612173240511051e+17], [9.709438739430175e+17, 1.78020492439554e+17, 7.16505557075273e+17], [6.642290623725988e+17, 3.163580591576003e+17, 2.757316414062727e+16], [9.082188596521943e+17, 9.066726695228206e+17, 8.037028950620861e+17], [5.858486209021828e+17, 8.548886315009682e+17, 2.5717920029934982e+17], [8.526381826111388e+17, 1.9378157308954058e+17, 1.3204409706545317e+17], [3.729663854703225e+16, 8.479819268721875e+17, 5.420711042751483e+17], [2.8570193774978824e+16, 7.777820488448488e+17, 5.863084776850417e+17], [8.03419000331392e+17, 2.3796542974276157e+17, 9.22526291821888e+17], [9.58181347186591e+17, 9582694180756968.0, 3.840823395027388e+17], [6.784109123481134e+17, 8.034233425545993e+17, 9.499401069818624e+17], [7.595569171161551e+17, 5.938962396171948e+17, 3.5083145966179784e+16], [9.882438794831217e+17, 2.5101891467311478e+17, 5.0727553839479514e+17], [8.261572085689965e+17, 3.050092923848474e+17, 3.305548484259486e+17], [1.1428537636215986e+17, 9.998093392318885e+16, 5.4668354211997203e+17], [3.145558174703638e+17, 7.13905309231275e+17, 9.958719931320758e+16], [1.773869062856126e+17, 7.592355290278963e+17, 4.86857042139362e+17], [7.56704891985314e+17, 5.306807705149443e+17, 8.582611278757088e+17], [1.8452357831211574e+17, 4.188530538756674e+17, 1.7104146074229776e+17], [1.6549021894007987e+17, 7.869271163822378e+16, 2.766308935824726e+17], [7.799067540345485e+17, 7.79871042186165e+17, 8.871944892436334e+17], [5.5721912813513197e+17, 5.2708726016587994e+17, 7.581481549465644e+17]], [[1.2097124104173662e+17, 5.450845439176388e+16, 1.3203773332441904e+17], [4.595708510380677e+17, 6.382403359700673e+16, 4.0817424889398e+17], [7.444173737375666e+17, 5.712015492403426e+17, 3.0559568673856228e+16], [9.721897144681064e+17, 5.41060243359459e+17, 8.594880140712535e+17], [3.493371096784598e+17, 4.44104368222519e+17, 6.542292947767427e+17], [1.3960014114944896e+17, 8.658212725313311e+17, 2.94198481012583e+17], [3.904628457955127e+17, 9.483518058659956e+17, 1.3418279605075589e+17], [8.823765911362006e+17, 1.5659189907892746e+17, 2.0039869055295888e+17], [6.339416493021562e+17, 5.156510042242526e+17, 2.3210225313895693e+17], [4.218786752822232e+17, 8.296939648661939e+17, 1.720485643436197e+17], [3.198892379790279e+17, 3.4059797205082976e+17, 5.92451442746439e+17], [6.826510683778328e+17, 8.785390923186674e+17, 2.6668451149333117e+17], [2.6638701173683098e+17, 2.211650038977888e+17, 1.0908077798722582e+17], [5.0645014061962424e+16, 7.451844011554433e+17, 9.800304848315379e+17], [3.605962954188149e+17, 2.0556468656133165e+17, 1.9989111222903667e+17], [1.3554677752439203e+17, 7.715012031743434e+17, 2.118495526025671e+17], [4.7075967285246803e+17, 5.3960696998975386e+17, 7.865039062841416e+17], [9.348606371876251e+17, 7885400177313873.0, 8.683077079102258e+17], [5.821536826914945e+17, 8.907266887011351e+17, 1.7073660981054717e+17], [6.940059433046657e+17, 6.170801340764963e+17, 1.5068234918096422e+17], [6.457371233536786e+17, 5.5001193173275245e+17, 8.037370725464786e+17], [4.338149447581996e+17, 7.176148098491866e+17, 8.045015430509471e+17], [6.858712597329827e+17, 4.3375413109854586e+17, 5.447112903636893e+17], [6.929466347391896e+17, 1.93203823737895e+17, 1.0527787093938912e+17], [5.206216299977957e+17, 9.070763569134062e+17, 7.583178318839401e+17], [1.2439968813898672e+17, 3.939735088076035e+17, 5.102962776532366e+17], [5.187253409583523e+17, 6.003087984133143e+17, 6.689731191228116e+17], [2.0912650720510816e+17, 2.088362450258152e+16, 7.94452810271448e+17], [2.301617195482526e+17, 5.638554619004054e+17, 2.5534903736809224e+16], [7.158077018926286e+17, 2.793419466303243e+17, 1.5668657551872333e+17], [1.1192369979362315e+17, 7.368461367811249e+17, 3.9981994852677005e+17], [3.169746097365044e+17, 5.502542315541947e+16, 9.885691573687692e+17]], [[5.4500877260796666e+17, 8.186611892844835e+17, 1.1660341066306202e+17], [8.902411057393687e+17, 1.7852525434091514e+17, 8.761355776148326e+17], [6.149909553799345e+17, 9.213059313982358e+17, 1.5264481637125306e+17], [6.636263662546519e+17, 8.8615969930541e+16, 3.6937643137719514e+17], [3.8633595950963226e+17, 4.5765717569599706e+17, 2.428004622160127e+17], [7.712444462804768e+17, 4.696925069422573e+17, 1.7363081237437485e+17], [4.842060964285422e+17, 6.662468949823597e+17, 9.050641549231117e+17], [7.478245607038587e+17, 4.080547522818582e+17, 7.148220159972296e+16], [4.1979952030201683e+17, 2.03954499593879e+17, 9.008207784973777e+17], [8.428145309938025e+17, 3.869153673766344e+17, 9.887995244398981e+17], [4.804562811018924e+17, 4.033061403992202e+17, 3.7349503279418e+17], [9.59439809082836e+16, 7.898649994717978e+17, 2.3964992426444387e+17], [6.007055643420211e+17, 3.884955738549677e+17, 4.6936125791047955e+17], [6.484786930009048e+17, 3.954764821921775e+17, 7.509581892167904e+17], [4.2517311372766264e+16, 7.799352560872297e+17, 5.180180596929266e+17], [8.486402815098234e+17, 7.087935031811589e+17, 3.644044315599778e+17], [5.0003234649871264e+17, 2.8992403230888864e+17, 4.821210054783465e+16], [7.703505886462584e+17, 5.52369834929849e+17, 4.51617615764214e+17], [5.265945654805192e+17, 1.196737714790278e+17, 9.640274533247254e+17], [4.754689165740942e+17, 9.677331710550086e+17, 8.534543692984328e+17], [4.885139437969266e+16, 5.202764057724223e+17, 5.278849909772135e+17], [7.922468965786248e+17, 5.776170042989766e+17, 3.878144550733774e+17], [2.9332794241737404e+16, 7.578457982389307e+17, 2.050270694472438e+17], [1.1580700638235918e+17, 2.657248560154163e+17, 8.495727097643757e+16], [6.327274823626171e+17, 3.874331411069548e+17, 4.884623719136417e+17], [1.9332139454059382e+17, 2.2045548971148355e+17, 6.695687032484064e+17], [3.766211888740486e+17, 9.185212529558728e+16, 6.591318730362784e+17], [7.678808689016438e+17, 1.7071623710392826e+17, 9.692716191251484e+17], [7.909472720312119e+17, 7.748644190081347e+17, 3.199767911349973e+17], [3.055032485969318e+17, 5.293106544308923e+16, 4.784830265599972e+17], [4.9326617735227405e+17, 4.3648315916182336e+17, 5.768893758935077e+17], [1.877076048281525e+17, 8.710552107872342e+17, 3.0015500726676525e+17]], [[7.137251931540325e+17, 4.7852072135199155e+17, 4.367890124455174e+17], [5.3838868086421357e+17, 6.440212188664651e+16, 5.005776321484565e+17], [7.945161678598058e+17, 5.3359143829726464e+17, 9.36619388382711e+17], [4.4596137641540243e+17, 5.283378072814221e+17, 4.667427488915724e+17], [5.971621906101339e+17, 1.1060230240299674e+16, 2.497240629987638e+17], [3.641577191983477e+17, 8.933382229595168e+17, 6.536144225008393e+17], [6.126755004079877e+17, 1.7813398356450794e+17, 7.479733170461499e+17], [1.3817556408873944e+17, 2.0830099538089564e+16, 1.0907340864595182e+17], [8.596586399324452e+17, 7.450785820506061e+17, 3.527308045176798e+17], [5.019249350801048e+17, 4.844998408359957e+17, 6.778647367088113e+17], [2.420225654464957e+17, 4.5256349852552205e+17, 8.960078215605848e+17], [7.330195218679881e+17, 2.424944995375076e+17, 4.540990192188152e+17], [8.349098426748092e+17, 5.679384826181418e+17, 1.3975923391092437e+17], [2.2011693206317296e+17, 8.827377079190199e+17, 6.476142698728726e+17], [4.9839061627681704e+16, 2.636726777231059e+17, 4.56378620058315e+17], [1.7620102203396048e+17, 8.925710941849194e+17, 7.299143215606464e+17], [4.768160995638813e+17, 9.993393290926419e+17, 1.2927439366219395e+17], [3.925928827537367e+17, 7.047963872452892e+17, 9.578075671427997e+17], [1.586930113906907e+17, 3.4835990648862116e+16, 1.6817911048315005e+17], [6.282218024016604e+17, 9.809681014751516e+17, 8.08659863374283e+17], [5.793963039095436e+17, 6.384178317827197e+17, 5.0177012469051194e+17], [7.058398397060932e+17, 8.561421770303726e+17, 6.119805562437606e+17], [6.977336642017032e+17, 5.78962876957992e+17, 5.416505772744217e+17], [9.650519880024928e+17, 4.423382395128944e+17, 1.1155331020557202e+17], [4.6143278961561043e+17, 2.6783689522978828e+16, 2.272384034493835e+16], [6.349930690536248e+17, 2.45678928678158e+16, 4.1556389342368416e+17], [4.6079305814162765e+17, 6.942101870658089e+17, 3.965744360722634e+16], [9.562696252017137e+17, 8.247803090876634e+17, 4.094596350358008e+17], [6.10842288914929e+17, 7.260808275679223e+17, 8.522715629950764e+17], [5.075640516849781e+17, 9.54034326465615e+17, 6.530929503631429e+17], [6.978797199855565e+17, 7.942439286230034e+17, 5.988307026427412e+17], [6.724659916338636e+17, 9.041956563510382e+17, 8.037849959872874e+17]], [[2.3776246762654397e+17, 4.381581168936437e+17, 5.375283061306101e+17], [1.8296263894673747e+17, 3.385175733805278e+17, 4.796756757953462e+17], [5.3110139406989766e+17, 8.048450277049533e+17, 1.071397686410458e+16], [6.674852398738976e+17, 7.382935059664096e+17, 7.619318109408379e+17], [8.691371483559606e+17, 3.729568304264501e+17, 1.0711548476363808e+17], [9.475724161604079e+17, 8.253278652863227e+17, 6.419662722453055e+17], [6.449334602287904e+17, 7.824890758736959e+17, 5.362331691481952e+17], [6.156561900890732e+17, 3.8287289298402304e+17, 7.174092108077732e+17], [9.763524483980662e+17, 1.679280557747893e+17, 9.884196811739697e+17], [1.313259544442692e+17, 7.468379014462112e+16, 6.19234388109735e+17], [4.086001459684591e+17, 4.223968520679866e+17, 2.5241553024952755e+17], [1.7877755965557962e+16, 7.072759457906204e+17, 3.1488911132721786e+17], [8.452915822789048e+17, 1.0119907155006269e+17, 8.147484787660576e+17], [4.574177710729458e+17, 7.148426726874949e+17, 2.5045772823563296e+17], [5.948032989512813e+17, 1.005022667319686e+17, 7.053350130490406e+17], [7.470913887111421e+17, 7.372791848559843e+17, 2.2057317584364368e+17], [1.4830163379504592e+17, 1.864108797250542e+17, 5.5886813135363936e+17], [2.4287374964851728e+17, 1.823856900317955e+17, 7.853444585752716e+17], [3.1425036809956e+17, 2.3390636578359338e+17, 8.18759866168523e+17], [4.290712128130123e+17, 3.919265950203714e+17, 4.512893177494735e+17], [8.59617475900848e+17, 2.6767565129775395e+17, 8.856868421570025e+17], [3.5103480887568525e+17, 5.434803404444505e+17, 9.603034884091226e+17], [4.3491196237985754e+17, 3.6051304948979315e+17, 7.093244500655483e+17], [2.1854039346357212e+16, 2.6854353965698218e+17, 3.5720651564601645e+17], [2.784082706212756e+17, 1.9680685989968972e+16, 3.864047327427003e+17], [8.083584431643077e+17, 8.243686272860362e+16, 9.812770932579794e+17], [6.195126415555547e+17, 6.396541468754597e+17, 6.170957891983523e+17], [7.702702289416253e+17, 8.127283225774193e+17, 9.676313676667558e+17], [7.048931032713462e+17, 9.607252766573044e+17, 9.053497838097806e+17], [2.2919848048372592e+17, 7.719829652739485e+17, 1.656348605449054e+16], [7.824362638679796e+17, 7.369930253329293e+17, 4.879371478978141e+17], [4.5571518036931904e+17, 9.614596967232428e+17, 5.976169188934345e+17]], [[3.339466452037684e+17, 1.698986146523962e+17, 8.389289651789679e+17], [9.354833633390365e+17, 5.348051730758015e+17, 4.296858742172504e+17], [8.478835516801298e+17, 1.7438954627772298e+17, 2.374920282829136e+17], [2.3631995465061894e+17, 3.008961016657123e+17, 8.666489216270856e+16], [4.286225499017566e+17, 4.2843046522693894e+17, 5.772090387574063e+17], [9.267958338364554e+17, 4.9036753022114477e+17, 4.7522308555503654e+17], [1.545081661448291e+17, 2.952642595562103e+17, 9.195142486374487e+17], [5.193662922180038e+17, 1.2796902828087464e+17, 4.948121245284811e+17], [5.747362898568732e+17, 5.3950449131199494e+17, 6.67888142904724e+17], [5.6759497124045734e+17, 7.262432354774107e+17, 8.179800630622794e+17], [6.012315001918761e+17, 1.8665306566035578e+17, 7.649432492767809e+17], [1.623277236000672e+17, 9.747898978361997e+17, 1.9797238751999856e+17], [3.2377356876820864e+17, 7.550721453179124e+17, 1.8314654160486666e+17], [7.763953797071822e+17, 8.493267196025321e+17, 1.9382647139871843e+17], [6.642069810868529e+17, 7.429403231655351e+17, 4.4735358876291656e+16], [9.149654976810949e+17, 1.8550628558724934e+17, 2.5841695022616874e+17], [3.3362439782053754e+17, 2.4223927678476675e+17, 6.419864861941352e+17], [4.5235093379474976e+17, 3.6342229918008384e+17, 6.373876887527707e+17], [8.485967715098515e+17, 9.60665994930577e+17, 4.16619141014426e+17], [7.246355489203515e+17, 9.801851345238934e+17, 4.954515299777482e+17], [9.194347838782185e+17, 7.691358451875274e+16, 9.638250151363123e+17], [2.3571287131939155e+17, 6.438906914825431e+17, 3.9816105938961197e+17], [6.410598896392317e+17, 1.3672087025772018e+17, 7.157776855055708e+17], [7.541133115126606e+17, 3.3755575047503226e+17, 4.181430541287097e+17], [6.919216073221669e+17, 5.882103885929354e+17, 7.56930400656126e+16], [6.543926292253921e+17, 2.8368410529834298e+17, 2.4648642388809274e+17], [5.93540514197305e+17, 9.870893190790253e+16, 7.00895140497045e+17], [1.0238521819101987e+17, 9.616543073348886e+17, 4.645078779271826e+17], [9.412151678408276e+17, 4.161531166029774e+17, 7.43367243126148e+17], [7.238390382413371e+17, 7.865961261993601e+17, 6.8111658649419e+17], [7.328749755938468e+17, 4.914630546464799e+17, 3.5103771125562374e+17], [7.618735451369558e+17, 6.245331521383543e+17, 5.138456024955017e+17]], [[5.336382757188022e+17, 5.4620997207241075e+17, 2.5006910882203437e+17], [6.87003157181869e+17, 6.18865088942454e+17, 6.188249010985364e+17], [2.2721080596403254e+17, 7.22883845941617e+17, 7.70703975042104e+17], [2.6416709585826237e+17, 1.4900071110310707e+17, 6.551540734998033e+17], [3.38984113352629e+17, 4.8589709847875674e+17, 3.382648433949582e+17], [9.846088675779253e+16, 9.959473820503028e+17, 4.6023117824634406e+17], [8.350167829081098e+17, 1.4725391370619e+17, 4.8350786224855336e+16], [6.758475807854053e+17, 1.92993864808242e+17, 6.710544915528388e+17], [8.37237187527248e+17, 3.761357830067111e+17, 2.8013321111030355e+17], [2.241190257145973e+17, 3.058575554190651e+17, 1.9254438379663584e+17], [7.327627343870317e+17, 7.281167034790205e+17, 7.370683322960718e+17], [4.6205418106501805e+17, 6.81680430135797e+17, 2.941144953453354e+17], [1.8818203640881037e+17, 7.393619292131426e+16, 4.292879779334241e+17], [4.9462461383440794e+17, 5.5576672530788595e+17, 8.424049510822292e+17], [9.484283816646492e+17, 6.619081376941453e+17, 3.519773742021943e+17], [7.803331846448221e+17, 3.798476613795977e+17, 8.255300971626917e+17], [1.5073729047951114e+17, 3.329343929162696e+17, 7.223104322892882e+17], [4.017863601778473e+17, 4.067293713749335e+17, 3.141395757105669e+17], [9.20228276674914e+17, 3.066287532435966e+17, 3.2418793994033468e+16], [2.3499320534551072e+17, 9.610853711166735e+17, 5.644652697174786e+17], [9.767761793981007e+17, 4.2860913557530054e+17, 2.9636485715965754e+17], [2.4310358751582685e+17, 1.317758983372903e+17, 7.537931304944841e+17], [2.686309921542831e+17, 5.1355058659861094e+17, 4.6074471883548474e+17], [8.167095942482583e+17, 5.654630344676963e+17, 3.286315528987137e+17], [5.931600099250382e+17, 4.2570485687571245e+17, 3.5894184564719744e+17], [5.159537589403591e+17, 8.72910644615662e+17, 3039100420805796.0], [6.445338308693382e+17, 7.757700347117998e+17, 9.531530672137244e+17], [2.2674619629581418e+17, 6.1108277923856536e+16, 5.933573886533492e+17], [1.0829485501666536e+17, 9.983391189437505e+17, 9.65077530502154e+17], [8.246853228518972e+17, 3.425511129613471e+17, 8.403251187091556e+17], [3.5912287448906374e+17, 3.8708149376741384e+16, 7.016659522979941e+17], [4.968738648557437e+17, 6.635953373264396e+17, 4.4995151980826496e+17]], [[3.228066257789335e+17, 8.086206872813606e+16, 9.425825536111712e+17], [3.7256284845713363e+17, 7.17116961143758e+17, 6.780611955359224e+17], [3.0514546990092704e+17, 9.736314201018651e+17, 5.856670905020655e+17], [4.57405547044619e+17, 6.2180467926698e+17, 6.739029425785217e+17], [2.5488381319956832e+17, 1.593047880570171e+17, 2.5885213409927632e+17], [9.233517705267139e+17, 2.308123331454347e+17, 2.0992913926019184e+17], [5.2407799724607686e+17, 7.194877063322985e+17, 4.544641691150082e+17], [9.566214764732067e+17, 3.6732517119498566e+17, 6.183813128884731e+17], [7.772117919381669e+17, 2.0999792907463667e+17, 5.266116147085822e+17], [1.7972320287163168e+17, 1.2312244125314864e+17, 5.1678413338924154e+17], [6.916557965811744e+17, 9.208662185562102e+17, 2.5990673623419892e+16], [8.867386103935224e+17, 5.5797437619632384e+17, 4.6778827115081645e+17], [6.059714252220816e+16, 6.83247704988136e+17, 1.3794871722748403e+17], [6.576230796473159e+16, 2.6906133444420554e+17, 8.28572070645476e+17], [5.519337145948536e+17, 1.9831090316475197e+17, 9.924458285480863e+17], [7.452232825975299e+17, 2.15366810558128e+17, 3.160684861169231e+17], [1.2639642757540882e+17, 1.0713719879667016e+17, 9.799523414140119e+17], [2.3205716612260474e+17, 5.674821803240504e+17, 3.401010120671202e+17], [2.637924602657231e+17, 4.71208433744835e+17, 2.0764350743025962e+17], [9.03561449617571e+17, 6.399562810669699e+17, 3.455503332692378e+16], [9.738123295290993e+17, 7.423298225942344e+17, 6.85184915948396e+17], [2.2558858836938934e+17, 6.179289834680837e+17, 1.8739187789155632e+17], [5.71946579432006e+17, 3.675090270884404e+17, 8.808732568406698e+17], [4.2182381227859686e+17, 2.5780617055587696e+17, 4.2831411521469696e+17], [8.714922002263244e+17, 3.456142130964673e+17, 3.086036638670984e+17], [3.28032260603017e+17, 8.403884856361914e+17, 3.2907161480508243e+17], [5.026476723294737e+16, 2.071293633941671e+17, 7.694073057323009e+17], [7.685285055804722e+17, 7.997769333274387e+17, 5.137595971603512e+17], [2.8897316464820666e+17, 9.787293756845865e+17, 6043563736453983.0], [9.380313411051666e+17, 7.636048876037098e+17, 2.158289444753977e+17], [7.319850894820416e+17, 1.9498133103078374e+17, 5.7187567389807e+17], [8.072541723106587e+17, 3.9674791249398214e+17, 9.886763714547178e+17]], [[7.553603807317869e+17, 5.766563279900333e+17, 3.511372951080659e+17], [4.074572327518321e+17, 4.9252626238132794e+17, 5.821544025138758e+16], [6.423522718900379e+17, 2.0952007732576528e+17, 3.7630697932829606e+17], [7.201279254202194e+17, 9.517799605482264e+17, 6.641788086154153e+17], [3.7625874334071155e+17, 3.4694176898593485e+17, 7.858201981154978e+16], [2.8487390827402624e+17, 1.7417212976727235e+17, 7.145013522143284e+17], [3.655246365704433e+17, 1.4555263710658163e+17, 1.188930212471373e+17], [2.512802582398056e+17, 5.1781523496152096e+17, 3.358287568107039e+17], [6.949675884389702e+17, 7.257817451585556e+17, 1.569149902855137e+17], [1.9903899632609946e+17, 5.675364049793229e+17, 6.494324345097852e+17], [3.906884782676822e+17, 6.169753039876569e+16, 9.732924565195278e+16], [1.187206847419986e+17, 4.1733612159758714e+17, 5.648830144973762e+17], [6.378408637907716e+17, 2.681403808601357e+17, 6.496642914544175e+17], [7.127419152329331e+17, 4.3774473789438216e+16, 1.5078635975238154e+17], [9.373214049328119e+17, 5.953846538049846e+17, 2.415912518235266e+17], [7.914176842355435e+16, 7.550294310141432e+17, 3.681389694794801e+17], [7.842486034902764e+17, 4.7094616402925037e+17, 4.505595403449991e+16], [1.9429981110551254e+17, 9.735147504013211e+17, 1.9745388379700422e+17], [6.2645093851507736e+16, 8.767041272168536e+17, 9.297795941696795e+16], [7.741546909863713e+17, 6.392036390920908e+16, 5.4579268484787336e+16], [4.863961046849771e+17, 8.423035843482708e+17, 7.318303566018268e+17], [5.4136949905337517e+17, 2.8103797455440426e+17, 3.442881252142995e+17], [9.663295532912536e+17, 9.921478498439973e+16, 5.4060145931715706e+17], [6.621019680913112e+17, 4.946849253169563e+17, 5.4844121719368346e+17], [6.06006731117303e+17, 4.109156267367555e+17, 4.318890583775898e+17], [2.5880759122250864e+17, 1.5037697912755898e+17, 1.788158066178277e+17], [6.700567204126262e+17, 1.6725384902249274e+17, 9.099699124908948e+17], [8.572578233732731e+16, 1.1805659738072504e+17, 3.454376913379532e+17], [1.8358554466227117e+17, 4.528386230442774e+17, 5.881291158152024e+17], [4.52038664097528e+17, 8.184277025182522e+17, 2.1901140467970458e+17], [9.015184131459738e+17, 5.3382557402124544e+17, 5.4905560093032646e+17], [2.3559270301704883e+17, 3.208313629915571e+16, 4.243873632994083e+17]], [[7.037578519660328e+17, 5.442015507653989e+17, 5.742387453958621e+17], [6.206842314705382e+17, 9.937648428280266e+16, 1.3434838963144358e+17], [4.3468506873921344e+17, 4.038432747469445e+16, 4.434257324774394e+16], [1.3377913938830333e+17, 3.166085698961235e+17, 9.966465047128978e+17], [5.771849920781713e+17, 1.6959123187740467e+17, 8.727218545760173e+17], [4.256682944363597e+17, 6.464341509908389e+17, 7.550377660778685e+17], [4.238608540325801e+17, 2.3786549919948352e+17, 2.7782267173831888e+17], [3.127664977703964e+17, 2.3003314177416856e+16, 1.6526637771976538e+17], [5.532331017719771e+16, 9.947536352257687e+17, 8.203887671334559e+17], [8.458019863823984e+16, 8.549101950779848e+17, 9.958954683195005e+17], [2.8124581329021516e+16, 4.7278802138491674e+17, 7.34261453118572e+16], [9.487515990357463e+17, 6.538119934146226e+17, 4.759577387846066e+17], [3.584231144750053e+17, 4.3100390426523974e+17, 1.685920469777842e+17], [3.663580064986637e+17, 3.1981884228503354e+17, 6.316572912401274e+17], [1.7893685629366474e+17, 6.925281370143837e+17, 9.762786662902622e+16], [8.909162662180998e+17, 4.392420464099541e+17, 8.148821652362729e+17], [6.959209061139872e+17, 7.251557333424786e+17, 8.120681688489112e+17], [2.0999498075028544e+16, 3.309601637362959e+17, 9.803030727209322e+17], [6.270648456592637e+17, 8.415407503002049e+17, 9.298444185924831e+17], [7.205704488623266e+16, 7.527645106935243e+16, 3.7210566618147846e+17], [9.385374942685425e+17, 6.843817009830515e+17, 5.28784121640382e+17], [9.157811155569746e+17, 2.2913888898760938e+17, 3.372474545701487e+16], [3.6043244175926445e+17, 8.605339176986772e+17, 5.130555673974688e+17], [3.5826958507303994e+17, 8.951503566934893e+17, 3.396905268545808e+17], [3.1701981765837075e+17, 5.5422262683474976e+17, 5.824529220338794e+17], [8.127752555860342e+17, 3.764896308166544e+17, 1.0523559549908323e+17], [820618242059478.8, 1.516836728805888e+17, 7.307914367954083e+17], [6.904286959406207e+17, 9.831445463899793e+17, 8.300258178849912e+17], [8.919899601006373e+17, 8.300512267196598e+17, 6.441163638084541e+17], [6.118373465069871e+17, 9.096288639824375e+17, 1.6277242148862836e+16], [9.821613080538943e+17, 4.5439255637778374e+17, 1.4035572216205772e+16], [1.8537793073012522e+17, 2.45624154769555e+17, 4.243525190151531e+16]], [[3.317811617818116e+17, 6.191000591467343e+17, 7.831789650057267e+17], [7.038796067776252e+17, 9.446931428511497e+17, 9.826067964457165e+17], [3.397172935813619e+16, 7.73820152164696e+17, 3.087293823758286e+17], [7.44008076173439e+17, 7.823488164957601e+17, 2.8645169905181978e+17], [9.132923824178703e+17, 3.628908277944368e+17, 7.260702122493065e+17], [5.5403097534968026e+17, 1.464596909918531e+17, 6.426368056804798e+16], [1.3890240261200526e+17, 1.002187254765755e+17, 1.9980424395921702e+17], [2.1820549506345853e+17, 9.665453620873554e+17, 4.598533512519489e+17], [2.567161735301592e+17, 5.0985181168607386e+17, 3.372729185993424e+17], [9.960950210330257e+17, 9.734385679454549e+16, 2.0331195063575302e+17], [8.71781050028017e+17, 2.249244872025735e+17, 1.4168235125258578e+17], [7.241408971067309e+17, 5.601805077510938e+17, 6.822441199289839e+17], [7.140374878695374e+17, 2.830351579945911e+16, 2.6022325485308618e+17], [3.654485906294893e+17, 5.0550661494360824e+16, 4.0544992623764864e+17], [1.638253623242586e+17, 2.5321659635514547e+17, 8.870849068578842e+17], [2.188863569960342e+16, 7.438230231302446e+17, 1104827226325322.2], [6.918923207568788e+17, 3.914968863391226e+17, 7.839765410941436e+17], [4.537143425222231e+17, 9.056440947640891e+17, 7.241535010616707e+16], [3.995952792761302e+17, 9.462850050994573e+17, 1.4259225659617891e+17], [9.11810247507853e+17, 4.3367320147021274e+17, 8.460046486050655e+17], [5.755583333173578e+17, 6.508191538536091e+17, 5.221042174536152e+17], [9.124633062119761e+17, 5.912418322476579e+17, 1.537802637236375e+17], [6.975687740754781e+17, 8.982475673020019e+17, 5.99821983567093e+17], [2.4808355752659795e+17, 9.989949446304974e+17, 8.757529156574276e+17], [9.13103494584205e+17, 5.4916602128998214e+17, 7.341835415776495e+17], [4.9736428134493894e+17, 7.18941909379115e+17, 9.441989731964612e+17], [1.4366935125318948e+16, 9.610968613294469e+16, 8.170918999519158e+17], [4.5507729063911386e+17, 7.525676486444602e+17, 1.457794084795827e+16], [4.298931944446735e+17, 6.142606702589615e+17, 5.890581336656997e+17], [3.9697049244235725e+17, 6.238523309372042e+17, 5.415028054393541e+17], [6.31511968772614e+17, 9.176436894708916e+17, 2.2798718750185576e+16], [1.1422071708536008e+17, 6.5386773344695e+16, 9.454160649220593e+17]], [[2.7281021811738525e+17, 1.5341091616116653e+17, 3.865847659134185e+17], [3.428986056810067e+17, 3.0243821208769485e+17, 3.8495045549803117e+17], [8.974929693032876e+17, 3.6482515036466643e+17, 9.11257764908564e+17], [4.6353837185822976e+17, 1.5297239593543533e+17, 1.5990978238000774e+17], [3.0999512773382308e+16, 4.7285593462802266e+17, 4.7313984288626534e+17], [5.877119909879415e+17, 4.192394940577171e+17, 7.6615704306068e+17], [6.023316350736982e+17, 5.0142846896856806e+17, 5.416173816450667e+17], [5515950442917750.0, 5.473561469748278e+16, 8.461251341297006e+17], [7.199887303193379e+17, 3.6187899831371494e+17, 1.794075659711415e+17], [8.6077549409641e+17, 6.221918594632657e+17, 8.760255922781818e+17], [4.489667890265968e+17, 7.335667310351442e+17, 2.7146246005812758e+17], [6.425911681034281e+17, 2.94812021514214e+17, 8.300677713452961e+17], [9.097915604849512e+17, 2.2121045400837114e+17, 3.626467807258222e+17], [7.927357188311923e+17, 7.956953684987149e+17, 8.521097506038131e+17], [2.567447160498325e+17, 6.379775267199095e+16, 1.6981062141797942e+17], [6.477659030516612e+17, 2.21791520827786e+17, 8.728654432497915e+17], [6.088812648016823e+17, 7.935101567576042e+17, 8.1035533400873e+16], [3.421081822225641e+17, 7.862664285107415e+17, 1.317303699594834e+17], [4.6499843777733664e+17, 6.865023084308375e+17, 8.414462265620438e+17], [7.463661838932641e+17, 2.7079923936601226e+17, 5.241409223932541e+17], [6.84510539886395e+17, 7.539498247794579e+17, 4.171349928676914e+17], [7.872515622545969e+17, 6.898095240195004e+17, 6.064132301665048e+17], [7.892279671711615e+17, 3.663140785535518e+17, 7.74160702476204e+17], [1.0234146087457096e+16, 7.025672345904406e+17, 2.1023527809627053e+17], [2.2585932236447126e+17, 3.931500817853051e+17, 9.398609898966246e+17], [2.204012052508586e+16, 4.870587852127293e+17, 2.6697925984455027e+17], [8.397690939091148e+17, 5.016828210405948e+17, 9.434713033240381e+17], [8.611974309832888e+17, 2.652456295287764e+17, 8.020133962701742e+17], [1.0708842860253942e+17, 2.5120153501379693e+17, 9.124714417890367e+17], [7.692040452973358e+16, 9.828390665541898e+16, 5.232126093167486e+17], [3.8768807177489894e+17, 1.7207369910701008e+17, 5.1583053499430464e+17], [2.3307886144169808e+17, 1.468220434936759e+17, 9.604969897450616e+17]], [[1.8428123903657834e+17, 3.347172768033679e+16, 6.5820617783272e+17], [1.3458655641169414e+17, 2.9295044630137344e+17, 2.177389770814625e+17], [3.5200923316507526e+17, 7.66688515177067e+17, 4.9773299572667104e+17], [4.088923905064001e+17, 4.54774266688713e+17, 8.430651463470303e+17], [2.5768715337967674e+17, 3.6596717209078675e+17, 4.6439148599085875e+17], [4.500929017141385e+17, 1.031053339247383e+17, 4.581892982588354e+17], [8.552251635223507e+17, 1.7033421662949966e+16, 6.774522893882605e+17], [9.884731974570264e+17, 5.108658240025233e+17, 2.3763849848278797e+17], [8.841914957512932e+17, 1.8696245590840733e+17, 5.0942357292583085e+17], [1.4213896294158235e+17, 4.761441589286194e+17, 6.759282857320397e+17], [5.233807519210121e+17, 1.9242008541158928e+17, 1.0776302446724829e+17], [9.14926903956359e+17, 5.329264696843893e+17, 4.9825076915777976e+16], [5.652591520686831e+17, 1.4590088342911133e+17, 2.5158633007236275e+17], [1.4978076738389568e+17, 4.726929343599839e+16, 5.857394597853555e+17], [6.243474427983332e+17, 5.562545374835185e+17, 1.9694034346467837e+17], [2.971137395788954e+17, 8.888592988909151e+17, 8.522027531997143e+17], [6.05508590824859e+17, 3.0012128241837114e+17, 3.793458568617454e+17], [9.06862958776721e+17, 5.3673013926654176e+17, 6.79424068731236e+17], [4.8867485574686355e+17, 3.588774566845061e+17, 9.948650000099034e+17], [2.1889895624094634e+17, 6.509913275419205e+17, 7.62003333186299e+17], [7.771385097407611e+17, 8.487054475542967e+17, 5.715405163439366e+17], [2.6611746657553904e+17, 8.531139210003024e+16, 1.277674933622448e+17], [1.1831233472613634e+17, 3.244588149374216e+17, 8.149646587153544e+17], [3.01681613638609e+17, 6.881612913472424e+17, 6.07203487053936e+17], [8.661690899486605e+17, 8.63940093718851e+17, 7.255838990939653e+17], [7.521653372353226e+17, 1.2679917145573405e+17, 6.668313847232732e+16], [6.452475039747363e+16, 7.338196692047106e+16, 9.774284826994039e+17], [4.622999909342196e+16, 4.66829356504496e+17, 3.9329307891684115e+17], [1.4634266504221293e+17, 4.9220527404185024e+17, 7.186670919697533e+17], [2.4861225839983738e+17, 2.8098638409058064e+17, 2.8671495666093062e+17], [2.4878568765206544e+17, 1.5246844149721018e+17, 3.211165606237066e+17], [1.4162202704538984e+17, 7.587453166001656e+16, 8.819690104434881e+17]], [[7.26578514175112e+17, 3.4155586659119776e+17, 1.7471708478703496e+16], [6.013997205748763e+17, 5.246939137191475e+17, 3.738163783969752e+17], [8.426645085657905e+17, 9.2593536097474e+17, 6.992292537425393e+17], [7.747472008564251e+17, 9.875442878981668e+17, 5.7482423231754266e+17], [8.181477479188855e+17, 9.783707434870656e+17, 4.796449549389132e+17], [8.185082344014216e+17, 4.356885086352719e+17, 1.461724268481538e+17], [7.909180578207515e+17, 7.44707440206685e+17, 6.100335732596728e+17], [5.889148222114047e+17, 3.964876855749533e+16, 9.549614508734716e+17], [4.8959997396991616e+17, 3.7456859242688376e+16, 7.527320421132933e+17], [1.256406034930907e+17, 5.590360701450222e+17, 1.7435308763043312e+17], [5.4018063488846016e+17, 2.105606907018742e+17, 5.95705469582582e+17], [2.9764138251423876e+16, 7.595004227557979e+17, 1.2091284315760564e+16], [8.904169071126253e+17, 3.645802441697845e+17, 6.039833965439535e+17], [5.766561306744307e+17, 8.897061093940047e+17, 8.476450286352128e+17], [1.4825903122668614e+17, 4.260255838940293e+17, 6.967367201175896e+17], [4.155711891870639e+16, 1.549000460142026e+17, 5.650443467971467e+17], [8.516463294402031e+17, 9.007651339500383e+17, 3.972227483904598e+17], [8.472981113603597e+17, 2.5862699073200268e+16, 5.981244885977966e+17], [8.522974843145427e+17, 4.666605506795677e+17, 8.855457939176375e+17], [8.067081747794723e+17, 2.8183605594021133e+17, 6.777296514852347e+17], [8.891811384362094e+17, 2.2191931781788067e+17, 7.905837927885972e+17], [4.522613884443361e+16, 5.4709790934130995e+17, 9.706215720471844e+17], [4.8075364811363475e+17, 7.949008701113652e+17, 2.2742542719774716e+16], [3.460395654478492e+17, 8.006156133841579e+16, 9.831523517208438e+17], [8.337152900493455e+17, 5.3807234484505094e+17, 3.051988897385255e+17], [4.28106698731786e+17, 7.40787358502082e+17, 1.5579614394107376e+17], [4.571262210586572e+17, 2.1551739522497482e+17, 8.804193611068883e+17], [5.422156207939474e+17, 2.6995166949739603e+17, 2.293559680895798e+17], [7.453944778926097e+17, 6.311016257104462e+17, 8.89628665938276e+17], [6.391759514846185e+16, 2.1040345339415133e+17, 6.605934112387894e+17], [8.086647238568329e+17, 2.7420888881042106e+17, 3.51155641108545e+17], [6.305489642102026e+17, 4.595771582146314e+17, 2.8061064194627104e+17]], [[6.320688196405254e+17, 2.2010465916116982e+17, 7.13451901149605e+17], [1.401191869614139e+17, 8.823939962717943e+17, 7.243412946112634e+17], [6.5044862081168664e+16, 8.879614652036799e+17, 5.137663936368715e+17], [8.906591648341062e+17, 4.6523437198673434e+17, 3.0702257315114444e+16], [8.324411387226077e+17, 7.318984995268943e+17, 5.000254426981743e+16], [9.592541909351949e+17, 9.877636979050641e+17, 1.5067978518612714e+17], [5.216144077820928e+16, 2.287156949106355e+17, 6.928177491546836e+17], [6.280665695745586e+17, 6.342356622046048e+17, 1.7462016544838032e+16], [5.1038818405463706e+17, 7.982647329848865e+17, 5.1588142254704384e+17], [8.177558687762892e+17, 5.184794546262912e+17, 9.723813702223391e+17], [3.054821205372971e+17, 4.7097763861952966e+17, 1.472371579362991e+17], [7.916688950076948e+17, 3.695828367668189e+17, 7.838161881069901e+17], [8.048971404707974e+17, 9.669509015198718e+16, 6.544447865700831e+16], [7.088448352766508e+16, 5.773573269406525e+17, 2.8468370210940032e+17], [4.2726246933198374e+17, 9.762661903855429e+17, 4.568502833440932e+17], [4.928780632031291e+17, 4.7927879734488557e+17, 1.425553527440091e+17], [8.115213475510019e+17, 4.90651234513632e+17, 2.7103976677777587e+17], [4.148030480950674e+17, 2.7942513088281386e+17, 9.7960710989148e+17], [9.424584892147917e+17, 6.47136438233622e+17, 1.0245341908861349e+17], [2.044565156766812e+17, 4.76594720590195e+17, 7.953698114505172e+17], [7.10040680740692e+17, 1.1246113853805584e+16, 1.3187726703019221e+17], [8.326762040801967e+17, 6.726810326068925e+17, 5.036742408218475e+16], [4.313819738046212e+17, 8.975914947307542e+17, 5.832271357406634e+17], [9.628358234995394e+16, 7.330327848104627e+17, 7.101981607565734e+17], [1.6201704339764877e+17, 6.562919535497184e+17, 8.602139344178697e+17], [5.7292087270017165e+17, 5.20057534845122e+16, 7.732691463131343e+17], [1.0423678026452598e+17, 3.965304166828215e+17, 9.487160601676867e+17], [6.090843251131293e+17, 4.4671307639722515e+17, 4.8193056123905965e+17], [6.013598989540841e+17, 9.774586847050354e+17, 2.622887763324321e+17], [2.1428580209398074e+17, 3.1037517663996486e+17, 5.141396242229594e+17], [7.789434993609263e+17, 6.135304208877586e+17, 1.0029432194514576e+17], [7.330322659123278e+16, 1.884174509282832e+16, 8.654044773552584e+17]], [[3.9655864048169984e+17, 4.283664507059365e+17, 1.7237060332202947e+17], [6.122403501912151e+17, 4.3307582335886816e+17, 7.035818765155332e+17], [8.923781902667834e+17, 5.207034401208248e+17, 9.993352234587676e+17], [8.460924870178145e+17, 5.995919776088431e+17, 3.068755925309288e+17], [1.7064323995042906e+17, 6.582950452076806e+17, 9.05198551281302e+16], [3.826436925357538e+17, 8.850874791324998e+17, 6.428393617652224e+17], [6.255997305890397e+17, 2.4203995962123104e+17, 8.855582153090035e+17], [8.962018202845611e+16, 2.2105755346523446e+17, 3.43661544700946e+17], [5.5348528615711456e+17, 6.16565811967732e+17, 2.5963596204128826e+17], [8.225830765110075e+17, 5.6758807291441024e+17, 3.8952729558599866e+17], [7.187581914893363e+17, 8.695363514842358e+17, 4.075630477205687e+17], [9.434895996432088e+17, 7.603803002889236e+17, 7.332814469885474e+16], [5.565272546396609e+17, 3.679120378538014e+16, 6.947669280287904e+17], [3.658126060552428e+17, 4.190816917694047e+17, 5.646851728802619e+17], [2.8276037435970624e+16, 2.6634066133690947e+17, 7.401234695826519e+17], [3.149285298700325e+17, 1.3863727277843274e+17, 8.651454889727084e+17], [2.248690270140107e+17, 1.1543031987388174e+17, 1.1326200600955306e+17], [4.825575170482105e+17, 7655462773584198.0, 3804559435995558.0], [2.3974964144854938e+17, 7.108786038591423e+17, 2.1271793915128688e+17], [8.950092741726395e+17, 4.4018438035809606e+17, 7.225126240197853e+17], [8.16710483846032e+17, 5.2779991081193043e+17, 8.769992759882595e+16], [3.988183730560618e+17, 2.128282017973878e+17, 3.7988540784481984e+17], [6.816636860265381e+17, 5.061707852840415e+17, 9.937128633267098e+17], [5.956113880599423e+17, 8.774783136584796e+17, 8.876567641958322e+17], [5.291143113069073e+17, 4.416256220070387e+17, 6.568397094710095e+17], [5.344852884061231e+17, 5.795634301644605e+17, 8.973790370829206e+17], [5.422071800655042e+17, 7.358819367452041e+17, 6.826059425454609e+17], [9.362556146841007e+17, 5.892393544064188e+17, 2.1375148228888963e+17], [1.9581522132455053e+17, 6.382217044519667e+17, 9.718013150740368e+16], [5.0073604914412915e+17, 1.7010898944635043e+17, 2.99541303696065e+17], [1.187367809890838e+16, 1.1200735905799653e+17, 9.479926941585988e+17], [7.022973232063425e+17, 7.100244181883409e+17, 2.7663703270241514e+17]], [[3.218392276942762e+17, 9.874118121097704e+17, 1.3706126113896432e+16], [3.1390128887874944e+17, 4.383737027189213e+17, 1.5605893798720493e+17], [1.1912802968400138e+16, 4.585667985043945e+17, 9.226527278325887e+17], [9.815183982100557e+17, 7.370711591857282e+16, 4.5534934802956474e+17], [9.397567854676394e+17, 5.329570806957563e+17, 6.541923658618996e+17], [3.234205725502294e+17, 8.815610050952666e+17, 4.110512174426021e+17], [3.6558613137837446e+17, 3.403389182231668e+17, 1.6317350592040448e+17], [8.09369284702951e+17, 7.159731570912168e+17, 8.223022665610493e+17], [4.246967779594464e+17, 8.682455075925556e+17, 3.6691981963413414e+17], [2.991331534127677e+17, 5.335632786937149e+17, 2268285903519573.5], [1.0291484957670571e+17, 6.05772604531683e+17, 2.271789247266537e+17], [9.022526192645363e+16, 3.4630915159027885e+17, 2.961705983069718e+16], [6.27467413281637e+17, 5.352237002724103e+17, 4.0710512724874584e+16], [2.7121635936073864e+16, 9.559141431809126e+17, 2.3100638011299347e+17], [7.336723787255985e+17, 9.561807870837819e+17, 8.197523342700305e+17], [9.65549155249862e+17, 6.518160205786941e+17, 2.5875944300062646e+17], [8.724620805819523e+17, 5.120786318025812e+17, 4.208758439490533e+17], [8.0384992395618e+17, 7.53551710982665e+17, 4.616006561481688e+17], [3.207932542737637e+17, 9.220258449427598e+16, 7.350804641228801e+17], [8.221969952309605e+17, 5.537854204570599e+17, 1.9625492204797446e+17], [6.843846210895644e+17, 8.845153901616354e+16, 1.1522648479174946e+16], [6.9517317909120744e+16, 4.811131863877961e+17, 6.492761114061316e+17], [9.179365271588809e+17, 4.086880988874223e+17, 5.903437403860246e+17], [6.120249803822577e+17, 7.164927676878301e+17, 7.476383865296791e+17], [8.37262931067502e+17, 5.797645398282193e+17, 5.223514494093147e+17], [9.576121449633975e+17, 9.9029653149447e+16, 3.0877778645137946e+17], [9.895247628422701e+17, 4.415733341984085e+17, 5.127515795594461e+17], [9.487891792714646e+17, 3.543307953642266e+17, 3.436007971612495e+17], [3.4046368718016806e+17, 3.636124469354327e+17, 7.484693400039219e+17], [4.162910825905176e+17, 1.2344049446340477e+17, 1.6811041039467654e+17], [3.6451329027105894e+17, 1.7197463933221923e+17, 7.019218153045453e+17], [7.23406638654784e+17, 2.4794000045084918e+17, 1.4925215726578333e+17]], [[5.819467676861754e+17, 4.5168370707101504e+17, 9.588879385996589e+17], [7.732582228830305e+17, 9.678410315464714e+16, 9.087301991939856e+16], [2.5535695790077418e+17, 1.3991961298336342e+17, 8.558911284604481e+17], [1.876026108281328e+17, 9.508998288191807e+17, 7.554473839581263e+17], [4.974839169293477e+16, 7.646527258816398e+17, 5.130875490593687e+17], [5.516761547010927e+17, 9.660407652812664e+17, 5.211145013241639e+17], [5.351565578930978e+17, 6.560844583492262e+17, 8.380498541698459e+17], [3.97111880486834e+17, 5.297544340050278e+17, 3.008464793817195e+17], [1.7865526877660275e+17, 2.6392436548055408e+17, 2.2536746454158506e+17], [7.029592834641112e+17, 3.04664815430582e+17, 6.475940263366281e+17], [6.367137437972093e+16, 9.238649699179529e+17, 3.798828076628862e+17], [7.19969000166088e+17, 3.4275615244268544e+17, 8.077458395591619e+17], [9.42745673667665e+17, 3837795274074129.5, 9.53398503499941e+17], [2.2146176238456074e+17, 5.017136873724333e+17, 3.519194906099241e+17], [4.941571502572706e+17, 6.260209974837322e+17, 1.320003546271058e+17], [4.664163168890822e+16, 4.468510435010776e+17, 2.1883768750331955e+17], [1.1903156899662315e+17, 4.761293344798041e+17, 5.7363222844607155e+17], [2.9693402698805894e+17, 1.6397091770395866e+17, 8.736623878708398e+17], [2.0080341672631898e+17, 1.6690955935549046e+17, 6.795330557206172e+17], [3.953968337510156e+17, 7.410238854034403e+17, 6.204468778091871e+17], [2.6500973924546845e+17, 2.507209561237035e+17, 1.7150716112761888e+17], [9.338024988471109e+17, 5.394325747932238e+16, 6.720507705917272e+17], [7.637225070761664e+17, 5.841109395026944e+16, 2.3852900503179242e+17], [8.012429711204108e+17, 2.2492749932622013e+17, 5.461094160031056e+17], [6.022607282560372e+17, 4.683379892312501e+17, 9.963262603456653e+17], [3.900381069645543e+17, 6.753345486288292e+17, 9.941219230007293e+16], [1.8374951802953965e+17, 2.749555215449708e+17, 2.5098586230759456e+17], [2.906049323103257e+17, 9.0385786172404e+16, 5.24325959759782e+17], [7.70750141778349e+17, 4.536364425213202e+16, 1.9329495784869267e+17], [5.5180641281836384e+17, 6.198392830174996e+17, 6.527405703001354e+17], [1.7598453626013578e+17, 4.9641134303748704e+17, 9.621159747673672e+17], [7.373926937556689e+17, 1.5209516962157875e+17, 3.1679448530443066e+17]]]
        tmp = [[[str(int(tmp[i][j][k] % p)) for k in range(nChannels)] for j in range(h)] for i in range(h)]
        final_json["in"] = tmp
        
        
        # make elements of final_json["in"] to string
        [[[str(X_in[i][j][k] % p) for k in range(nChannels)] for j in range(h)] for i in range(h)]
        with open(input_path, "w") as f:
            json.dump(final_json, f)
        # print all keys circuit_json
        print("circuit_json keys:", circuit_json.keys())
        print("\n\n")
        print("input_json keys:", input_json.keys())
        print("============================")
    else:
        input_path = tmp_folder + "input.json"
    wit_file = tmp_folder + "witness.wtns"
    print(sag)

    mem_usage = []
    time_cost = []
    benchmark_start_time = time.time()

    for i in range(len(test_images)):
        print ("process for image ",i)
        cost = 0
        X = test_images[i]
        start_time = time.time()

        if not model_name == 'resnet20':
            _ = prepare_input_json_cnn(X, layers, input_path)

        # command = ['node', wit_json_file, wasm_file, input_path, wit_file]
        # subprocess.run(command)
        commands = [['node', wit_json_file, wasm_file, input_path, wit_file],
                    ['snarkjs', 'groth16', 'prove',zkey, wit_file, tmp_folder+'proof.json', tmp_folder+'public.json']]

        for command in commands:
            # compute the time of the prove command (second command)
            if command == commands[1]:
                start_time = time.time()

            stdout, _, usage = execute_and_monitor(command)
            
            if command == commands[1]:
                time_prove = time.time() - start_time
            # print ('command:', command)
            # print (stdout)
            if "ERROR" in stdout:
                print (stdout)
                return
            cost += usage

        if verify:
            command = ['snarkjs', 'groth16', 'verify',veri_key, tmp_folder+'public.json', tmp_folder+'proof.json']
            subprocess.run(command)
        # print ("stdout:", stdout)
            
        out = load_and_convert_json_to_int(tmp_folder+'public.json')
        out = [x if x < p//2 else 0 for x in out]
        pred = np.argmax(out)

        if pred != predictions[i]:
            loss += 1
            print ("Loss happens on index", i)

        mem_usage.append(cost)
        # time_cost.append(time.time() - start_time)
        time_cost.append(time_prove)

    print ("Total time:", time.time() - benchmark_start_time)
    layers = model_name.split("_")
    arch = arch_folders[model_name][:-1]
    arch = '-'.join(word.capitalize() for word in arch.split('-')) + '_Kernal'

    layers[0] = str(int(layers[0])**2)
    new_row = {
        'Framework': ['circomlib-ml (tensorflow)'],
        'Architecture': [f'{arch} ({"x".join(layers[:-1])}_{layers[-1]}x{layers[-1]})'],
        '# Layers': [len(layers)-1],
        '# Parameters': [params[model_name]],
        'Testing Size': [len(mem_usage)],
        'Accuracy Loss (%)': [loss/len(mem_usage) * 100],
        'Avg Memory Usage (MB)': [sum(mem_usage) / len(mem_usage)],
        'Std Memory Usage': [pd.Series(mem_usage).std()],
        'Avg Proving Time (s)': [sum(time_cost) / len(time_cost)],
        'Std Proving Time': [pd.Series(time_cost).std()]
    }

    new_row_df = pd.DataFrame(new_row)
    print (new_row_df)

    if save:
        df = load_csv()
        df = pd.concat([df, new_row_df], ignore_index=True)
        csv_path = '../../benchmarks/benchmark_results.csv'
        df.to_csv(csv_path, index=False)

    return

def show_models():
    for key in params:
        layers = key.split("_")
        if int(layers[0]) < 30:
            arch = arch_folders[key]
        else:
            arch = "input" + (len(layers)-1) * "-dense" 

        print (f'model_name: {key} | arch: {arch}')

def update_zkey(ceremony_folder, model_name, output_folder = './tmp/'):
    print ('Update zkey to avoid mismatch in witness and circuit')
    r1cs_path = output_folder + model_name + ".r1cs"
    zkey_1 = ceremony_folder + 'test_0000.zkey'
    ptau_3 = ceremony_folder + 'pot12_final.ptau'
    command = ['snarkjs', 'groth16', 'setup', r1cs_path, ptau_3, zkey_1]
    # print (command)
    res = subprocess.run(command, capture_output=True, text = True)
    # print ("afer update")
    if "ERROR" in res.stdout:
        print (res.stdout)

# Function to load JSON content and convert it to a list of integers
def load_and_convert_json_to_int(file_path):
    try:
        # Open and read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Convert each string in the list to an integer
        int_list = [int(item) for item in data]
        
        return int_list
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
    except ValueError:
        print("Error converting string to int.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


def find_digit(output):
    match = re.search(r'non-linear constraints: (\d+)', output)
    if match:
        constraints = int(match.group(1))
        # Calculate k such that 2**k > 2 * constraints
        k = 1
        while 2**k <= 2 * constraints:
            k += 1
        print(f"Constraints: {constraints}, k: {k}")
        return k
    else:
        print("Constraints not found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark result for a given model and testsize.")

    # Mutually exclusive for showing models only
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument('--list', action='store_true', help='Show list of supported models and exit')

    parser.add_argument('--save', action='store_true', help='Flag to indicate if save results')
    parser.add_argument('--debug', action='store_true', help='Flag to indicate if verify proofs')

    parser.add_argument('--size', type=int, help='Test Size')
    parser.add_argument('--model', type=str, help='Model file path')

    parser.add_argument('--output', type=str, default="tmp",help='Specify the output folder')

    args = parser.parse_args()

    if args.list:
        show_models()
        sys.exit()

    if args.model not in params and args.model != "resnet20":
        print ("Please check the model name by using '--list'")
        sys.exit()

    if not args.model or args.size is None:
        parser.error('--model and --size are required for benchmarking.')

    if args.model == "resnet20":
        layers = [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64] 
        target_circom = "./golden_circuits/resnet20.circom" # output of keras2circom
    else:
        layers = [int(x) for x in args.model.split("_")]
        target_circom = "_".join(str(x) for x in layers) + '.circom'

    model_path = "../../models/"
    output_folder = f'./{args.output}/'
    os.makedirs(output_folder, exist_ok=True)
    
    # check if the files exist
    r1cs_file = os.path.join(output_folder, target_circom.replace(".circom", ".r1cs"))
    if os.path.exists(r1cs_file):
        print("bencmark.py - r1cs file already exists")
    else:
        command = ['circom', "./golden_circuits/" + target_circom, "--r1cs", "--wasm", "--sym", "-o", output_folder]
        res = subprocess.run(command, capture_output=True, text = True)
        print (res.stdout)
    # digit = find_digit(res.stdout)

    zkey_1 = output_folder + f"ceremony-{args.model}/test_0000.zkey"
    veri_key = output_folder + f"ceremony-{args.model}/vk.json"

    # Check if zkey_1 and veri_key exist
    if not os.path.exists(zkey_1) or not os.path.exists(veri_key):
        print ('Start Trusted-setup before performing benchmark')
        # Call trusted_setup.py if either file does not exist
        trusted_setup_command = [
            'python', 'trusted_setup.py',
            '--model', args.model,
            '--output', args.output
        ]
        subprocess.run(trusted_setup_command)

    if layers[0] > 30 and not (args.model=='resnet20'):
        dnn = True
    else:
        dnn = False

    if args.model == "resnet20":
        # arch_folder = arch_folders[args.model]
        # model_path = "../../models/"
        # model_in_path = model_path+arch_folder+args.model + '.h5'
        # model_in_path = '../zkml/resnet20.h5'

        # model = gen_model_cnn(layers, model_in_path)
        model = ResNet20Cifar100()
        # TODO: load weights and biases
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        predicted_labels, tests = prepare_cnn(model, layers, model_name = args.model)

        benchmark_cnn(tests[:args.size], predicted_labels[:args.size], 
                layers, args.model, output_folder, output_folder+"input.json", zkey_1, veri_key, verify=args.debug, save=args.save)

    elif dnn:
        arch_folder = "input" + (len(layers)-1) * "-dense" + "/"
        model_path = "../../models/"
        model_in_path = model_path+arch_folder+args.model + '.h5'

        model = gen_model_dnn(layers, model_in_path)

        predicted_labels, tests = prepare(model, layers)
        weights, biases = transfer_weights(layers, model, 36)

        benchmark_dnn(tests[:args.size], predicted_labels[:args.size], weights, biases,
                  layers, args.model, output_folder, output_folder+"input.json", zkey_1, veri_key, verify=args.debug, save=args.save)
       
    else:
        arch_folder = arch_folders[args.model]
        model_path = "../../models/"
        model_in_path = model_path+arch_folder+args.model + '.h5'

        model = gen_model_cnn(layers, model_in_path)

        predicted_labels, tests = prepare_cnn(model, layers)

        benchmark_cnn(tests[:args.size], predicted_labels[:args.size], 
                layers, args.model, output_folder, output_folder+"input.json", zkey_1, veri_key, verify=args.debug, save=args.save)


