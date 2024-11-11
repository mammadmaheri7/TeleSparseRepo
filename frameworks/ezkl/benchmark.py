import re
import os, subprocess, sys
import onnxruntime
import torch, struct, os, psutil, subprocess, time, threading
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import json, ezkl
import pandas as pd

import subprocess, concurrent
import psutil
import time
import argparse

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

params = {"784_56_10": 44543,
          "196_25_10": 5185,
          "196_24_14_10": 5228,
            "28_6_16_10_5": 5142,
            "14_5_11_80_10_3": 4966, # @TODO: May doublecheck
            "28_6_16_120_84_10_5": 44530,
            "resnet20": (-1),
            "efficientnetb0": (-1),
            "mobilenetv1": (-1)}

accuracys = {"784_56_10": 0.9740,
            "196_25_10": 0.9541,
            "196_24_14_10": 0.9556,
            "28_6_16_10_5": 0.9877,
            "14_5_11_80_10_3": 0.9556, # @TODO: May doublecheck
            "28_6_16_120_84_10_5": 0.9877}

arch_folders = {"28_6_16_10_5": "input-conv2d-conv2d-dense/",
                "14_5_11_80_10_3": "input-conv2d-conv2d-dense-dense/",
                "28_6_16_120_84_10_5": "input-conv2d-conv2d-dense-dense-dense/",
                "resnet20": "resnet20/",
                "efficientnetb0": "efficientnetb0/",
                "mobilenetv1": "mobilenetv1/"}
import psutil
def get_cpu_load():
    # Get CPU usage for each core
    cpu_loads = psutil.cpu_percent(interval=1, percpu=True)
    return cpu_loads

def select_k_cpus_with_lowest_load(k):
    # Get CPU usage for each core
    cpu_loads = get_cpu_load()
    
    # Create a list of tuples (core_id, load)
    cpu_load_tuples = list(enumerate(cpu_loads))
    
    # Sort the list based on load
    sorted_cpu_loads = sorted(cpu_load_tuples, key=lambda x: x[1])
    
    # Get the IDs of the K cores with lowest load
    selected_cpus = [core[0] for core in sorted_cpu_loads[:k]]
    
    return selected_cpus


def set_cpu_affinity(pid, cpu_list):
    try:
        p = psutil.Process(pid)
        p.cpu_affinity(cpu_list)
        print(f"CPU affinity for process {pid} set to: {cpu_list}")
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} does not exist.")
    except psutil.AccessDenied:
        print("Permission denied. You may need sudo privileges to set CPU affinity.")

def dnn_datasets():
    (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # suffle the data
    np.random.seed(7)
    idx = np.random.permutation(len(test_images))
    test_images = test_images[idx]
    test_labels = test_labels[idx]

    # Convert to PyTorch tensors
    test_images_pt = torch.tensor(test_images).float()
    test_labels_pt = torch.tensor(test_labels)
    # Flatten and normalize the images
    test_images_pt = test_images_pt.view(-1, 28*28) / 255.0  # Flatten and normalize

    # Assuming test_images_pt is your PyTorch tensor with shape [num_samples, 784]
    test_images_pt_reshaped = test_images_pt.view(-1, 1, 28, 28)  # Reshape to [num_samples, channels, height, width]

    # Downsample images
    test_images_pt_downsampled = F.interpolate(test_images_pt_reshaped, size=(14, 14), mode='bilinear', align_corners=False)

    # Flatten the images back to [num_samples, 14*14]
    test_images_pt_downsampled = test_images_pt_downsampled.view(-1, 14*14)

    return test_images_pt, test_images_pt_downsampled, test_labels_pt

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

    # Convert to PyTorch format [batch_size, channels, height, width]
    train_images_pt = torch.tensor(train_images_tf).permute(0, 3, 1, 2).float()
    test_images_pt = torch.tensor(test_images_tf).permute(0, 3, 1, 2).float()


    train_images_pt_14 =  torch.tensor(test_images_tf_14).permute(0, 3, 1, 2).float()
    test_images_pt_14 =  torch.tensor(test_images_tf_14).permute(0, 3, 1, 2).float()

    return test_images_pt, test_images_pt_14


import os, glob
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    print("ImportError: Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True, testsize=-1, args=None):
    if testsize != -1:
        labels = []
        files = []
        # import pdb; pdb.set_trace()
        for i, l in enumerate(sorted(os.listdir(data_dir))):
            ps = glob.glob(os.path.join(data_dir, l, "*.JPEG"))
            files += ps
            labels += [i] * len(ps)
        labels = labels[::len(files) // testsize][:-1]
        files = files[::len(files) // testsize][:-1]
        print(is_training, len(files))
        images, labels = fn.readers.file(files=files,
                                        labels=labels,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        random_shuffle=True,
                                        pad_last_batch=True,
                                        name="Reader")
    else:
        images, labels = fn.readers.file(file_root=data_dir,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        random_shuffle=True,
                                        pad_last_batch=True,
                                        name="Reader")

    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in  the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
    images = fn.resize(images,
                        device=dali_device,
                        size=size,
                        mode="not_smaller",
                        interp_type=types.INTERP_CUBIC)
    # images = fn.jitter(images, 
    #                    interp_type=types.INTERP_CUBIC)

    images = fn.crop_mirror_normalize(images,
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[d * 255 for d in args.mean],
                                      std=[d * 255 for d in args.std],
                                      mirror=False)
    labels = labels.cpu()
    return images, labels

def get_val_imagenet_dali_loader(args, val_batchsize=32, crop_size=224, val_size=256):
    args.local_rank = 0
    args.dali_cpu = False
    args.world_size = 1
    args.workers = 1
    if not hasattr(args, 'imagenet_dir'):
        args.imagenet_dir = "/rds/general/user/mm6322/home/imagenet"
    valdir = os.path.join(args.imagenet_dir, 'val')
#     valdir = os.path.join(data_route["imagenet"], 'val')
    pipe = create_dali_pipeline(batch_size=val_batchsize,
                                num_threads=args.workers,
                                device_id=None,
                                seed=12 + args.local_rank,
                                data_dir=valdir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=True,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=False,
                                testsize=args.val_testsize,
                                args=args)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    return val_loader

def evaluate_pytorch_model(model, datasets, labels):
    # Create TensorDataset for test data
    test_dataset = TensorDataset(datasets, labels)
    # Create a DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def monitor_memory(pid, freq = 0.001):
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
    start_time = time.time()
    if command[0] == 'python':
        command[0] = 'python3'  # Update to python3 if needed
        command.insert(1, '-u')

    print(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(monitor_memory, process.pid)
        stdout, stderr = process.communicate()
        max_memory = future.result()
    if show:
        print(f"Maximum memory used: {max_memory} MB")
        print("Total time:", time.time() - start_time)
    return stdout, stderr, max_memory

def benchmark_dnn(test_images, predictions, model, model_name, mode = "resources", output_folder='./tmp/', save = False, notes = ""):
    data_path = os.path.join(output_folder, 'input.json')
    model_path = os.path.join(output_folder, 'network.onnx')

    sampled_data = test_images[0].unsqueeze(0).detach().cpu()
    # sampled_data = torch.rand(1, 784)
    
    with torch.no_grad():
        torch.onnx.export(model, 
                    sampled_data, 
                    model_path, 
                    export_params=True, 
                    # opset_version=10, 
                    opset_version=12,
                    do_constant_folding=True, 
                    input_names=['input'], 
                    output_names=['output'],
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    
    # torch.onnx.export(model, x, model_path, export_params=True, opset_version=12, do_constant_folding=True, input_names=['input_0'], output_names=['output'])
    loss = 0
    mem_usage = []
    time_cost = []
    benchmark_start_time = time.time()

    for i, img in enumerate(test_images):
        print ("Process for image", i)
        start_time = time.time()
        # Convert the tensor to numpy array and reshape it for JSON serialization
        x = (img.cpu().detach().numpy().reshape([-1])).tolist()
        data = dict(input_data = [x])

        # Serialize data into file:
        json.dump(data, open(data_path, 'w'))

        command = ["python", "gen_proof.py", "--model", model_path, "--data", data_path, "--output", output_folder, "--mode", mode]
        # subprocess.run(command)
        # stdout = "1234"
        # usage = 1
        stdout, _, usage = execute_and_monitor(command)

        # retrive the proof time from the stdout
        match = re.search(r'proof took (\d+\.\d+)', stdout)
        if match:
            proof_took = match.group(1)
            print(f"Proof took: {proof_took} seconds")
        else:
            print("Proof took value not found")

        # retrive the prediction from the stdout
        try:
            # pred = int(stdout[-2])
            pred = int(re.search(r'Prediction: (\d+)', stdout).group(1))
        except ValueError:
            print(f"Failed to convert {stdout[-2]} to int. Full output: {stdout}")
            pred  = -1

        if pred != predictions[i]:
            loss += 1
            print ("Loss happens on index", i, "predicted_class", pred)
        mem_usage.append(usage)
        # time_cost.append(time.time() - start_time)
        time_cost.append(float(proof_took))

    print ("Total time:", time.time() - benchmark_start_time)

    layers = model_name.split("_")
    arch = "Input" + (len(layers)-1) * "-Dense"
    new_row = {
        'Framework': ['ezkl (pytorch)'],
        'Architecture': [f'{arch} ({"x".join(layers)})'],
        '# Layers': [len(layers)],
        '# Parameters': [params[model_name]],
        'Testing Size': [len(mem_usage)],
        'Accuracy Loss (%)': [loss/len(mem_usage) * 100],
        'Avg Memory Usage (MB)': [sum(mem_usage) / len(mem_usage)],
        'Std Memory Usage': [pd.Series(mem_usage).std()],
        'Avg Proving Time (s)': [sum(time_cost) / len(time_cost)],
        'Std Proving Time': [pd.Series(time_cost).std()],
        'Notes': [f'mode={mode}']
    }

    if notes:
        new_row['Notes'] = [new_row['Notes'][0] + " | " + notes]
    new_row_df = pd.DataFrame(new_row)
    print (new_row_df)

    if save:
        df = load_csv()
        df = pd.concat([df, new_row_df], ignore_index=True)
        csv_path = '../../benchmarks/benchmark_results.csv'
        df.to_csv(csv_path, index=False)

    return

def benchmark_cnn(test_images, predictions, model, model_name, mode = "resources", output_folder='./tmp/', save=False, notes="", labels=None):
    print("Benchmarking CNN model called")
    # check model is instance of string and contain .onnx
    if isinstance(model, str) and ".onnx" in model:
        model_path = model
    else:
        model_path = os.path.join(output_folder, 'network.onnx')

    data_path = os.path.join(output_folder,model_name, 'input.json')
    loss = 0
    loss_with_true_label = 0
    mem_usage = []
    time_cost = []
    proof_size = []
    verification_times = []

    benchmark_start_time = time.time()

    # setting 50 percent of model weights to zero
    # print ("====== Setting 90 percent of model weights to zero ======")
    # # set 90 percent of trainable param model to zero
    # for param in model.parameters():
    #     param.data = param.data * (torch.rand(param.size()) > 0.1).float()

    sampled_data = test_images[0].unsqueeze(0).detach().cpu()
    # print("Sampled data shape:", sampled_data.shape)
    # print summary of the model

    if not isinstance(model, str) or ".onnx" not in model:
        model.eval()
        try:
            with torch.no_grad():
                    torch.onnx.export(model, 
                                sampled_data, 
                                model_path, 
                                export_params=True, 
                                # opset_version=10, 
                                opset_version=12,
                                do_constant_folding=True, 
                                input_names=['input'], 
                                output_names=['output'],
                                dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                                'output' : {0 : 'batch_size'}})
                    import onnx
                    on = onnx.load(model_path)
                    for tensor in on.graph.input:
                        for dim_proto in tensor.type.tensor_type.shape.dim:
                            if dim_proto.HasField("dim_param"): # and dim_proto.dim_param == 'batch_size':
                                dim_proto.Clear()
                                dim_proto.dim_value = 1   # fixed batch size
                    for tensor in on.graph.output:
                        for dim_proto in tensor.type.tensor_type.shape.dim:
                            if dim_proto.HasField("dim_param"):
                                dim_proto.Clear()
                                dim_proto.dim_value = 1   # fixed batch size
                    onnx.save(on, model_path)

                    on = onnx.load(model_path)
                    on = onnx.shape_inference.infer_shapes(on)
                    onnx.save(on, model_path)
        except Exception as e:
            print(f"Error: {e}")
            # throw error
            exit(1)
    else:
        print("Model is already in ONNX format and will be used for benchmarking - directory:", model_path)

    for i in range(len(test_images)):
        print ("Process for image", i)
        start_time = time.time()
        img = test_images[i:i+1].detach().cpu()

        if not isinstance(model, str) or ".onnx" not in model:
            with torch.no_grad():
                    torch.onnx.export(model, 
                                img, 
                                model_path, 
                                export_params=True, 
                                # opset_version=10, 
                                opset_version=12,
                                do_constant_folding=True, 
                                input_names=['input'], 
                                output_names=['output'],
                                dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                                'output' : {0 : 'batch_size'}})
                    
                    import onnx
                    on = onnx.load(model_path)
                    for tensor in on.graph.input:
                        for dim_proto in tensor.type.tensor_type.shape.dim:
                            if dim_proto.HasField("dim_param"): # and dim_proto.dim_param == 'batch_size':
                                dim_proto.Clear()
                                dim_proto.dim_value = 1   # fixed batch size
                    for tensor in on.graph.output:
                        for dim_proto in tensor.type.tensor_type.shape.dim:
                            if dim_proto.HasField("dim_param"):
                                dim_proto.Clear()
                                dim_proto.dim_value = 1   # fixed batch size
                    onnx.save(on, model_path)

                    on = onnx.load(model_path)
                    on = onnx.shape_inference.infer_shapes(on)
                    onnx.save(on, model_path)
        else:
            print("Model is already in ONNX format and will be used for benchmarking - directory:", model_path)

        # Serialize data into file:
        x = (img.cpu().detach().numpy().reshape([-1])).tolist()
        data = dict(input_data = [x])
        # if data_path is not exist, create it
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        json.dump(data, open(data_path, 'w'))

        command = ["python", "gen_proof.py", "--model", model_path, "--data", data_path, "--output", output_folder, "--mode", mode]
        # subprocess.run(command)
        # stdout = "1234"
        # usage = 1
        
        stdout, error, usage = execute_and_monitor(command)

        # log stdout to log file
        os.mkdir('./logs') if not os.path.exists('./logs') else None
        with open(f'./logs/{model_name}_log.txt', 'a') as f:
            f.write(f"Image {i}:\n")
            f.write(stdout)
            f.write("\n\n\n\n")
        with open(f'./logs/{model_name}_error.txt', 'a') as f:
            f.write(f"Image {i}:\n")
            f.write(error)
            f.write("\n\n\n\n")
        # print("===== Error:", error)
        # print("===== Stdout:", stdout)

        # retrive the proof time from the stdout
        match = re.search(r'proof took (\d+\.\d+)', stdout)
        if match:
            proof_took = match.group(1)
            print(f"Proof took: {proof_took} seconds")
        else:
            print("Proof took value not found")
            proof_took = 0

        # Extract the verification time from the stdout
        match = re.search(r'verify took (\d+\.\d+)', stdout)
        if match:
            m = match.group(1)
            # convert to int and convert to ms
            m = float(m) * 10
            verification_times.append(str(m)) # convert to ms
            print(f"Verification took: {match.group(1)} seconds")
        else:
            print("Verification time value not found")
            # return

        # Extract the predicted class from the stdout
        try:
            pred = int(stdout[-2])
        except ValueError:
            print(f"Failed to convert {stdout[-2]} to int. Full output: {stdout}")
            pred  = -1

        # Calculate the loss
        if pred != predictions[i]:
            loss += 1
            print ("Loss happens on index", i, "predicted_class", pred, "\t onnx_prediction", predictions[i])

        if pred != labels[i]:
            loss_with_true_label += 1
            print ("Loss happens on index", i, "predicted_class", pred, "\t true_label", labels[i])
        
        mem_usage.append(usage)
        time_cost.append(float(proof_took))
        # compute proof size
        proof_path = os.path.join(output_folder, 'proof.json')
        proof_size.append(os.path.getsize(proof_path) / 1024)  # in KB

    print ("Total time:", time.time() - benchmark_start_time)

    print("List mem usage:", mem_usage)
    print("List time cost:", time_cost)
    print("List proof size:", proof_size)
    print("List verification times:", verification_times)


    if model_name=="resnet20":
        layers = [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64] 
    elif model_name=="efficientnetb0":
        layers = [11, 11, 11]
    elif model_name == "mobilenetv1":
        layers = [32, 64, 128, 128, 256, 256]
    else:
        layers = model_name.split("_")

    arch = arch_folders[model_name][:-1]
    arch = '-'.join(word.capitalize() for word in arch.split('-')) + '_Kernal'

    layers[0] = str(int(layers[0])**2)

    new_row = {
        'Framework': ['ezkl (pytorch)'],
        # 'Architecture': [f'{arch} ({"x".join(layers[:-1])}_{layers[-1]}x{layers[-1]})'],
        'Architecture': [f'{arch}'],
        '# Layers': [len(layers)-1],
        '# Parameters': [params[model_name]],
        'Testing Size': [len(mem_usage)],
        'Accuracy Loss (%)': [loss/len(mem_usage) * 100],
        'Acc@1 (%)' : [loss_with_true_label/len(test_images) * 100],
        'Avg Memory Usage (MB)': [sum(mem_usage) / len(mem_usage)],
        'Std Memory Usage': [pd.Series(mem_usage).std()],
        'Avg Proving Time (s)': [sum(time_cost) / len(time_cost)],
        'Std Proving Time': [pd.Series(time_cost).std()],
        'Proof Size (KB)': [sum(proof_size) / len(proof_size)],
        'Std Proof Size (KB)': [pd.Series(proof_size).std()],
        'Verification Time (ms)': [sum([float(x) for x in verification_times]) / len(verification_times)],
        'Std Verification Time (ms)': [pd.Series([float(x) for x in verification_times]).std()],
        'Notes': notes
    }

    new_row_df = pd.DataFrame(new_row)
    print (new_row_df)

    if save:
        df = load_csv()
        df = pd.concat([df, new_row_df], ignore_index=True)
        csv_path = '../../benchmarks/benchmark_results.csv'
        df.to_csv(csv_path, index=False)

    return

def load_csv():
    csv_path = '../../benchmarks/benchmark_results.csv'

    columns = ['Framework', 'Architecture', '# Layers', '# Parameters', 'Testing Size', 'Accuracy Loss (%)', 'Acc@1 (%)', 
            'Avg Memory Usage (MB)', 'Std Memory Usage', 
            'Avg Proving Time (s)', 'Std Proving Time' ,
            'Proof Size (KB)', 'Std Proof Size (KB)', 'Verification Time (ms)', 'Std Verification Time (ms)',
            'Notes']

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

def gen_model_dnn(layers, state_dict):
    if len(layers) == 3:
        class Net(nn.Module):
            def __init__(self, num_classes=10):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(layers[0], layers[1])  # Flatten 
                self.fc2 = nn.Linear(layers[1], layers[2])  

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

    elif len(layers) == 4:
        class Net(nn.Module):
            def __init__(self, num_classes=10):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(layers[0], layers[1])  # Flatten 
                self.fc2 = nn.Linear(layers[1], layers[2])
                self.fc3 = nn.Linear(layers[2], num_classes)  

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
    else:
        print ("Layers not Support")
        return None
    
    model = Net()
    model.load_state_dict(state_dict)
    model.eval()
    return model

# @ TODO: Hardcoded
def gen_model_cnn(layers, state_dict):
    if len(layers) == 6:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # Convolutional encoder
                self.conv1 = nn.Conv2d(1, layers[1], layers[-1]) 
                self.conv2 = nn.Conv2d(layers[1], layers[2], layers[-1]) 

                # Fully connected layers / Dense block
                self.fc1 = nn.Linear(11 * 2 * 2, layers[3]) # 256 * 120
                self.fc2 = nn.Linear(layers[3], layers[4])

            def forward(self, x):
                # Convolutional block
                x = F.avg_pool2d(F.relu(self.conv1(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool
                x = F.avg_pool2d(F.relu(self.conv2(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool

                # TODO: figure out the resize, currently work on batch_size = 1
                batch_size = x.size(0)
                x = x.reshape(x.size(0),layers[2],-1)  # 16 output channels
                x = np.transpose(x, (0,2,1)).reshape(batch_size,-1)
                #x = x.reshape(batch_size,-1)

                # Fully connected layers
                x = F.relu(self.fc1(x))
                x = self.fc2(x)  # No activation function here, will use CrossEntropyLoss later
                return x
            
    elif len(layers) == 7:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # Convolutional encoder
                self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
                self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel

                # Fully connected layers / Dense block
                self.fc1 = nn.Linear(16 *4 * 4,120) # 256 * 120
                self.fc2 = nn.Linear(120, 84)         # 120 inputs, 84 outputs
                self.fc3 = nn.Linear(84, 10)          # 84 inputs, 10 outputs (number of classes)

            def forward(self, x):
                # Convolutional block
                x = F.avg_pool2d(F.relu(self.conv1(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool
                x = F.avg_pool2d(F.relu(self.conv2(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool

                # TODO: figure out the resize, currently work on batch_size = 1
                batch_size = x.size(0)
                x = x.reshape(x.size(0),16,-1)  # 16 output channels
                x = np.transpose(x, (0,2,1)).reshape(batch_size,-1)
                #x = x.reshape(batch_size,-1)

                # Fully connected layers
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)  # No activation function here, will use CrossEntropyLoss later
                return x
    elif len(layers) == 5:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # Convolutional encoder
                self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
                self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel

                # Fully connected layers / Dense block
                self.fc1 = nn.Linear(16 *4 * 4,10) # 256 * 120

            def forward(self, x):
                # Convolutional block
                x = F.avg_pool2d(F.relu(self.conv1(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool
                x = F.avg_pool2d(F.relu(self.conv2(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool

                # TODO: figure out the resize, currently work on batch_size = 1
                batch_size = x.size(0)
                x = x.reshape(x.size(0),16,-1)  # 16 output channels
                x = np.transpose(x, (0,2,1)).reshape(batch_size,-1)
                #x = x.reshape(batch_size,-1)

                # Fully connected layers
                x = self.fc1(x)  # No activation function here, will use CrossEntropyLoss later
                return x
    else:
        print ("Layers not Support")
        return None
    
    model = Net()
    model.load_state_dict(state_dict)
    model.eval()
    return model

def prepare(model, layers):
    if layers[0] == 196:
        _, test_images, _ = dnn_datasets()
    elif layers[0] == 784:
        test_images, _, _ = dnn_datasets()

    with torch.no_grad():  # Ensure gradients are not computed
        predictions = model(test_images)
        predicted_labels = predictions.argmax(dim=1)

    predicted_labels = predicted_labels.tolist()
    return predicted_labels, test_images

def prepare_by_onnx(model_name=None,onnx_path=None,num_samples=1,args=None):
    if model_name=='resnet20':
        # Define the transformations for the test dataset (Normalize CIFAR-100 according to dataset statistics)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # Mean and std for CIFAR-100
        ])
        # Load the CIFAR-100 test dataset
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        # Create the DataLoader for the test dataset
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        # test_images = next(iter(test_images))[0]
        predictions = []
        test_images = []
        test_labels = []

        for i, (images, labels) in enumerate(test_loader):
            if i>=num_samples:
                break
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            test_images.append(images)
            test_labels.append(labels)

            # do inference on the onnx model
            ort_session = onnxruntime.InferenceSession(onnx_path)
            ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            predicted_labels = np.argmax(ort_outs[0], axis=1)
            predictions.append(predicted_labels)

        
        # convert list to tensor
        test_images = torch.cat(test_images)
        test_labels = torch.cat(test_labels)
        predictions = torch.tensor(predictions).squeeze()

        return predictions, test_images, test_labels

    elif model_name=='efficientnetb0':
        # Transformation for imagenet dataset
        if not hasattr(args, 'imagenet_dir'):
            args.imagenet_dir = "/rds/general/user/mm6322/home/imagenet"

        args.val_testsize = num_samples + 1
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        args.mean = IMAGENET_DEFAULT_MEAN
        args.std = IMAGENET_DEFAULT_STD

        val_loader = get_val_imagenet_dali_loader(args, val_batchsize=1, crop_size=224, val_size=256)

        predictions = []
        test_images = []
        test_labels = []

        for i, data in enumerate(val_loader):
            if i>=num_samples:
                break
            images = data[0]['data']
            labels = data[0]['label']
            test_images.append(images)
            test_labels.append(labels)

            # do inference on the onnx model
            ort_session = onnxruntime.InferenceSession(onnx_path)
            ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            predicted_labels = np.argmax(ort_outs[0], axis=1)
            predictions.append(predicted_labels)

        # convert list to tensor
        test_images = torch.cat(test_images)
        test_labels = torch.cat(test_labels)
        predictions = torch.tensor(predictions).squeeze()

        return predictions, test_images, test_labels

    elif model_name=='mobilenetv1':
        # cifar10 dataset
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        predictions = []
        test_images = []
        test_labels = []

        for i, (images, labels) in enumerate(test_loader):
            if i>=num_samples:
                break
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            test_images.append(images)
            test_labels.append(labels)

            # do inference on the onnx model
            ort_session = onnxruntime.InferenceSession(onnx_path)
            ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            predicted_labels = np.argmax(ort_outs[0], axis=1)
            predictions.append(predicted_labels)

        # convert list to tensor
        test_images = torch.cat(test_images)
        test_labels = torch.cat(test_labels)
        predictions = torch.tensor(predictions).squeeze()

        return predictions, test_images, test_labels

    else:
        print("Model not supported")
        # error ValueError: Model not supported
        exit(1)



def prepare_cnn(model, layers):
    if layers[0] == 14:
        _, test_images = cnn_datasets()
    elif layers[0] == 28:
        test_images, _= cnn_datasets()

    with torch.no_grad():  # Ensure gradients are not computed
        predictions = model(test_images)
        predicted_labels = predictions.argmax(dim=1)

    predicted_labels = predicted_labels.tolist()
    return predicted_labels, test_images

def gen_model_dnn(layers, state_dict):
    if len(layers) == 3:
        class Net(nn.Module):
            def __init__(self, num_classes=10):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(layers[0], layers[1])  # Flatten 
                self.fc2 = nn.Linear(layers[1], layers[2])  

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

    elif len(layers) == 4:
        class Net(nn.Module):
            def __init__(self, num_classes=10):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(layers[0], layers[1])  # Flatten 
                self.fc2 = nn.Linear(layers[1], layers[2])
                self.fc3 = nn.Linear(layers[2], layers[3])  

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
    else:
        print ("Layers not Support")
        return None
    
    model = Net()
    model.load_state_dict(state_dict)
    model.eval()
    return model

def show_models():
    for key in params:
        layers = key.split("_")
        if int(layers[0]) < 30:
            arch = arch_folders[key]
        else:
            arch = "input" + (len(layers)-1) * "-dense" 

        print (f'model_name: {key} | arch: {arch}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate benchmark result for a given model and testsize.",
        epilog="Example usage: python benchmark.py --size 100 --model model_name"
    )
    #parser = argparse.ArgumentParser(description="Generate benchmark result for a given model and testsize.")
    parser.add_argument('--size', type=int, help='Test Size')
    parser.add_argument('--model', type=str, help='Model file path')

    # Mutually exclusive for showing models only
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument('--list', action='store_true', help='Show list of supported models and exit')

    parser.add_argument('--save', action='store_true', help='Flag to indicate if save results')
    parser.add_argument('--agg', type=int, help='Set the start for aggregating benchmark results')

    parser.add_argument('--accuracy', action='store_true', help='Flag to indicate if use accuracy mode which may sacrifice efficiency')

    # add argument sparsity with default value 0.0
    parser.add_argument('--sparsity', type=float, default=0.0, help='Sparsity of the model')
    parser.add_argument('--teleported', action='store_true', help='Flag to indicate if use teleported mode')

    # number of cpu cores
    parser.add_argument('--cores', type=int, help='Number of CPU cores to use', default=32)

    # args.imagenet_dir
    parser.add_argument('--imagenet_dir', type=str, help='Directory of the ImageNet dataset', default="/rds/general/user/mm6322/home/imagenet")

    args = parser.parse_args()

    if args.list:
        show_models()
        sys.exit()

    if args.model not in params:
        print ("Please check the model name by using '--list'")
        sys.exit()

    if not args.model or args.size is None:
        parser.error('--model and --size are required for benchmarking.')

    # limit the number of cores
    selected_cpus = select_k_cpus_with_lowest_load(args.cores)
    print(f"Selected {args.cores} CPUs with lowest load:", selected_cpus)
    # Get the process ID of the current process
    pid = os.getpid()
    set_cpu_affinity(pid, selected_cpus)
    
    # Define the layers for the model
    if args.model == "resnet20":
        layers = [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64]
        layers = [str(x) for x in layers]
    elif args.model == "efficientnetb0":
        layers = [11, 11, 11]
        layers = [str(x) for x in layers]
    elif args.model == "mobilenetv1":
        layers = [32, 64, 128, 128, 256, 256]
        layers = [str(x) for x in layers]
    else:
        layers = [int(x) for x in args.model.split("_")]
    model_path = "../../models/"

    start = 0

    if not args.model=="resnet20" and not args.model=="efficientnetb0" and not args.model=="mobilenetv1" and layers[0] > 30 :
        dnn = True
    else:
        dnn = False

    # if args.accuracy and dnn:
    if args.accuracy:
        mode = "accuracy"
    else:
        mode = "resources"

    # Update notes
    notes = f'mode={mode}'
    if args.agg:
        start = args.agg
        notes += f' | start from {start}'
    if args.sparsity > 0.0:
        notes += f' | sparsity={args.sparsity}'
    if args.teleported:
        notes += " | teleported"

    # Benchmarking on specific architecture
    if args.model == "resnet20" or args.model == "efficientnetb0" or args.model == "mobilenetv1":
        arch_folder = arch_folders[args.model].rstrip("/")
        
        # define the onnx path 
        # onnx_path = f"../../models/resnet20/{args.model}"
        onnx_path = f"../../models/{arch_folder}/{args.model}"
        dataset_name = "cifar100" if args.model == "resnet20" else "imagenet" if args.model == "efficientnetb0" else "cifar10"
        onnx_path += f"_{dataset_name}"
        if args.sparsity > 0.0:
            onnx_path += f"_sparse{str(int(args.sparsity))}"
        if args.teleported:
            onnx_path += "_teleported"
        onnx_path += ".onnx"
        
        # do inference on the onnx model + dataset loading
        # TODO: uncomment the following line to do inference on the onnx model, after connecting the teleportaion on each image
        # predicted_labels, test_images, test_labels = prepare_by_onnx(args.model,onnx_path,num_samples=args.size,args=args)
        if args.model == "resnet20":
            test_image = np.load(f"../../models/{arch_folder}/8.npy")
            test_images = torch.tensor(test_image).unsqueeze(0)
            predicted_labels = torch.tensor([75])
            test_labels = torch.tensor([75])
            # repeat the test_images, predicted_labels, test_labels for args.size times
            test_images = test_images.repeat(args.size, 1, 1, 1)
            predicted_labels = predicted_labels.repeat(args.size)
            test_labels = test_labels.repeat(args.size)
        else:
            print("EXPILITCI DATASET DID NOT IMPLEMENTED YET")
            exit(1)

        # calculate the accuracy of original onnx model        
        accuracy_orignal_onnx = (predicted_labels == test_labels).sum().item() / len(test_labels)
        print(f"Accuracy of original ONNX model: {accuracy_orignal_onnx}")

        # do benchmarking
        benchmark_cnn(test_images, predicted_labels, onnx_path, args.model, 
                    mode=mode, save=args.save, notes=notes, labels=test_labels)
    elif dnn:
        arch_folder = "input" + (len(layers)-1) * "-dense" + "/"

        model_path = "../../models/"

        state_dict = torch.load(model_path + arch_folder+ args.model + ".pth")

        output_folder = './tmp/' + "_".join([str(x) for x in layers]) + "/"
        os.makedirs(output_folder, exist_ok=True)

        model = gen_model_dnn(layers, state_dict)

        if args.sparsity > 0.0:
            # set args.sparsity percent of trainable param model to zero
            for param in model.parameters():
                param.data = param.data * (torch.rand(param.size()) > args.sparsity).float()

        predicted_labels, tests = prepare(model, layers)
        benchmark_dnn(tests[start:start+args.size], predicted_labels[start:start+args.size], model, args.model, 
                    mode=mode, save=args.save, notes=notes)
    else:
        arch_folder = arch_folders[args.model]
        
        state_dict = torch.load(model_path + arch_folder+ args.model + ".pth")

        model = gen_model_cnn(layers, state_dict)

        if args.sparsity > 0.0:
            # set args.sparsity percent of trainable param model to zero
            for param in model.parameters():
                param.data = param.data * (torch.rand(param.size()) > args.sparsity).float()
            notes += f' | sparsity={args.sparsity}'
        
        predicted_labels, tests = prepare_cnn(model, layers)

        benchmark_cnn(tests[:args.size], predicted_labels[:args.size], model, args.model, 
                mode=mode,save=args.save, notes=notes)
