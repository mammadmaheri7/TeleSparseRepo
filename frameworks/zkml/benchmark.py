import numpy as np
import re, os, argparse, sys
import tensorflow as tf
import concurrent.futures, subprocess, threading, psutil, time
import pandas as pd

params = {"784_56_10": 44543,
          "196_25_10": 5185,
          "196_24_14_10": 5228,
            "28_6_16_10_5": 5142,
            "14_5_11_80_10_3": 4966, # @TODO: May doublecheck
            "28_6_16_120_84_10_5": 44530,
            "resnet20": (-1),
            "effnetb0": (-1)

            }
         

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
                "effnetb0": "effnetb0/",
                }

def get_predictions(interpreter, test_images):
    predictions = []

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    for i, img in enumerate(test_images):
        test_image = np.expand_dims(img, axis=0).astype(np.float32)
        
        # Set the value for the input tensor
        interpreter.set_tensor(input_details[0]['index'], test_image)
        
        # Run the inference
        interpreter.invoke()

        # Retrieve the output and dequantize
        output = interpreter.get_tensor(output_details[0]['index'])
        output = np.argmax(output, axis=1)
        predicted_class = output[0]

        predictions.append(predicted_class)


    return predictions

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
    start_time = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(monitor_memory, process.pid)
        stdout, stderr = process.communicate()
        max_memory = future.result()
    if show:
        print(f"Maximum memory used: {max_memory} MB")
        print("Total time:", time.time() - start_time)
    return stdout, stderr, max_memory

def dnn_datasets():
    # Load TensorFlow MNIST data
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

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

def cnn_datasets(dataset_name=None,args=None):
    if dataset_name is not None and dataset_name == "cifar100":
        cifar100_nm = [[0.5071,0.4867,0.4408],[0.2675,0.2565,0.2761]]
        # Load TensorFlow CIFAR100 data
        cifar100 = tf.keras.datasets.cifar100
        (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
        train_images_tf = train_images / 255.0
        test_images_tf = test_images / 255.0

        # Normalize the images
        train_images_tf = (train_images_tf - cifar100_nm[1]) / cifar100_nm[0]
        test_images_tf = (test_images_tf - cifar100_nm[1]) / cifar100_nm[0]

        train_images_tf = train_images_tf.reshape(train_images.shape[0], 32, 32, 3)
        test_images_tf = test_images_tf.reshape(test_images.shape[0], 32, 32, 3)
        return test_images_tf, test_labels
    elif dataset_name is not None and dataset_name == "imagenet":
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        # if any(st in model for st in ["vit", "deit"]):
        #     args.mean = [0.5,] * 3
        #     args.std = [0.5,] * 3
        # else:
        args.mean = IMAGENET_DEFAULT_MEAN
        args.std = IMAGENET_DEFAULT_STD
        # if "384" in model:
        #     train_loader,test_loader = get_trainval_imagenet_dali_loader(args, batch_size, 384, 384)
        #     calib_loader = get_calib_imagenet_dali_loader(args, batch_size, 384, 384, calib_size=args.calib_size)
        # else:
        train_loader,test_loader = get_trainval_imagenet_dali_loader(args, 32)
        # select N images as test set
        N = 2048
        test_images = []
        test_labels = []
        for i, (images, labels) in enumerate(test_loader):
            test_images.append(images.numpy())
            test_labels.append(labels.numpy())
            if i >= N:
                break
        test_images = np.concatenate(test_images, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        test_images = test_images[:N]
        test_labels = test_labels[:N]
        return test_images, test_labels
        
        # calib_loader = get_calib_imagenet_dali_loader(args, batch_size, calib_size=args.calib_size)

    else:
        # Load TensorFlow MNIST data
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        train_images_tf = train_images / 255.0
        test_images_tf = test_images / 255.0
        train_images_tf = train_images_tf.reshape(train_images.shape[0], 28, 28, 1)
        test_images_tf = test_images_tf.reshape(test_images.shape[0], 28, 28, 1)

        train_images_tf_14 = tf.image.resize(train_images_tf, [14, 14]).numpy()
        test_images_tf_14 = tf.image.resize(test_images_tf, [14, 14]).numpy()

        return test_images_tf, test_labels

def get_trainval_imagenet_dali_loader(args, batchsize=32, crop_size=224, val_size=256):
    args.local_rank = 0
    args.dali_cpu = False
    args.world_size = 1
    args.workers = 1
    args.testsize = -1
    args.val_testsize = -1 #args.calib_size
    # if args.imagenet_dir is None:
    # check if the args object has the attribute
    if not hasattr(args, 'imagenet_dir'):
        args.imagenet_dir = "/rds/general/user/mm6322/home/imagenet"
    traindir = os.path.join(args.imagenet_dir, 'train')
    
    pipe = create_dali_pipeline(batch_size=batchsize,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=traindir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=True,
                                testsize=args.testsize,
                                args=args)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    val_loader = get_val_imagenet_dali_loader(args, batchsize, crop_size, val_size)
    return train_loader, val_loader


def benchmark(test_images, predictions, model_name, model_in_path, circuit_folder, test = False, save = False, notes = "",labels=None):
    # Convert the model
    tmp_folder = "./tmp/"
    msgpack_folder = tmp_folder + "msgpack/"
    os.makedirs(msgpack_folder, exist_ok=True)

    model_convert_path = "./tools/converter.py"
    model_out_path = tmp_folder + "msgpack/converted_model.msgpack"
    config_path = tmp_folder + "msgpack/config.msgpack"

    if model_name == "resnet20":
        scale_factor = (2**12)
        k = 21
        num_cols = 32
        num_randoms = 1024 * 32
    elif model_name == "effnetb0":
        scale_factor = (2**12)
        k = 21
        num_cols = 32
        num_randoms = 1024 * 32
    else:
        scale_factor = (2**12)
        k = 18
        num_cols = 10 
        num_randoms = 1024 

    print("=== PROVING PARAMETERS ===" , "\t scale_factor:", scale_factor, "\t k:", k, "\t num_cols:", num_cols, "\t num_randoms:", num_randoms)
    time.sleep(2)

    command = ["python", model_convert_path, "--model", f"{model_in_path}",
            "--model_output", f"{model_out_path}", "--config_output",
            config_path, "--scale_factor", str(scale_factor),
            "--k", str(k), "--num_cols", str(num_cols), "--num_randoms",
            str(num_randoms)]
    execute_and_monitor(command)


    loss = 0
    loss_with_true_label = 0
    img_inputs_path = tmp_folder + "inputs/"
    os.makedirs(img_inputs_path, exist_ok=True)
    
    input_convert_path = "./tools/input_converter.py"
    config_out_path = msgpack_folder+"config.msgpack"

    time_circuit = circuit_folder + "time_circuit"
    test_circuit = circuit_folder + "test_circuit"

    call_path = time_circuit
    if test:
        call_path = test_circuit

    model_out_path = tmp_folder + "msgpack/converted_model.msgpack"

    mem_usage = []
    time_cost = []
    proof_size = []
    verification_times = []
    benchmark_start_time = time.time()

    # for i, img in enumerate(test_images):
    for i in range(len(test_images)):
        img = test_images[i]
        label_img = labels[i][0]

        cost = 0
        print ("Process for image", i)
        start_time = time.time()

        np.save(f"{img_inputs_path}{str(i)}.npy", img)
        
        # Convert the input to the model
        img_in_path = img_inputs_path + str(i)+ ".npy"
        img_out_path = msgpack_folder + "img_" + str(i) + ".msgpack"


        command_1 = ["python", f"{input_convert_path}", "--model_config", f"{config_out_path}",
                "--inputs", img_in_path, "--output", img_out_path]
        # print (command_1)
        command_2 = [call_path, model_out_path, img_out_path, "kzg"]
        # print (command_2)
        _, _, usage = execute_and_monitor(command_1)
        cost += usage
        stdout, _, usage = execute_and_monitor(command_2)
        cost += usage

        # Extract the proving time from the output
        proving_time_pattern = r"Proving time: ([\d\.]+)s"
        # Search for the pattern in the text
        match = re.search(proving_time_pattern, stdout)
        if match:
            proving_time = match.group(1)
            print(f"Proving time: {proving_time} seconds")
            verification_times.append(proving_time)
        else:
            print("Proving time not found.") 
            return

        # Extract the verification time from the output
        verification_time_pattern = r"Verifying time: ([\d\.]+)ms"
        # Search for the pattern in the text
        match = re.search(verification_time_pattern, stdout)
        if match:
            verification_time = match.group(1)
            print(f"Verification time: {verification_time} ms")

        else:
            print("Verification time not found.")
            # return


        # Extract x values using regex
        x_values = [int(x) for x in re.findall(r'final out\[\d+\] x: (-?\d+) \(', stdout)][-10:]
        #x_values = [int(x) for x in re.findall(r'final out\[\d+\] x: (\d+)', stdout)][-10:]
        #print (x_values)

        # Find max value and its index
        max_value = max(x_values)
        max_index = x_values.index(max_value)
        # print (max_index)
        
        if max_index != predictions[i]:
            loss += 1
            print ("Loss happens on index", i, "predicted_class", max_index, "predicted_tflite", predictions[i])

        if label_img is not None and max_index != label_img:
            loss_with_true_label += 1
            print ("Loss happens on index", i, "predicted_class", max_index, "true_label", label_img)
        
        mem_usage.append(cost)
        # time_cost.append(time.time() - start_time)
        time_cost.append(float(proving_time))
        
        # compute proof size
        proof_path = "./proof"
        proof_size.append(os.path.getsize(proof_path) / 1024)

        # print("start stdout")
        # print(stdout)
        # print("=====================================")

        # log stdout to log file
        # mkdir logs if not exist
        os.mkdir('./logs') if not os.path.exists('./logs') else None
        with open(f'./logs/{model_name}_log.txt', 'a') as f:
            f.write(f"Image {i}:\n")
            f.write(stdout)
            f.write("\n\n\n\n")



    print ("Total time:", time.time() - benchmark_start_time)

    if model_name=="resnet20":
        layers = [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64] 
        layers = [str(x) for x in layers]
    elif model_name=="effnetb0":
        layers = ["11","11","11"]
    else:
        layers = model_name.split("_")

    if int(layers[0]) < 30:
        arch = arch_folders[model_name][:-1]
        arch = '-'.join(word.capitalize() for word in arch.split('-')) + '_Kernal'
        layers[0] = str(int(layers[0])**2)

        new_row = {
            'Framework': ['zkml (tensorflow)'],
            'Architecture': [f'{arch} ({"x".join(layers[:-1])}_{layers[-1]}x{layers[-1]})'],
            '# Layers': [len(layers)-1],
            '# Parameters': [params[model_name]],
            'Testing Size': [len(mem_usage)],
            'Accuracy Loss (%)': [loss/len(test_images) * 100],
            'Acc@1' : [loss_with_true_label/len(test_images) * 100],
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
        # arch = f'{arch_folder} ({"x".join(layers)})'
    else:
        layers = model_name.split("_")
        arch = "Input" + (len(layers)-1) * "-Dense"

        new_row = {
            'Framework': ['zkml (tensorflow)'],
            'Architecture': [f'{arch} ({"x".join(layers)})'],
            '# Layers': [len(layers)],
            '# Parameters': [params[model_name]],
            'Testing Size': [len(mem_usage)],
            'Accuracy Loss (%)': [loss/len(test_images) * 100],
            'Acc@1' : [loss_with_true_label/len(test_images) * 100],
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
            'Avg Memory Usage (MB)', 'Std Memory Usage', 'Avg Proving Time (s)', 'Std Proving Time' , 
            'Proof Size (KB)', 'Std Proof Size (KB)', 'Verification Time (s)', 'Std Verification Time (s)',
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

def show_models():
    for key in params:
        layers = key.split("_")
        if int(layers[0]) < 30:
            arch = arch_folders[key]
        else:
            arch = "input" + (len(layers)-1) * "-dense" 

        print (f'model_name: {key} | arch: {arch}')


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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate benchmark result for a given model and testsize.",
        epilog="Example usage: python benchmark.py --size 100 --model model_name"
    )    
    # Mutually exclusive for showing models only
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument('--list', action='store_true', help='Show list of supported models and exit')

    parser.add_argument('--size', type=int, help='Test Size')
    parser.add_argument('--model', type=str, help='Model file path')
    parser.add_argument('--agg', type=int, help='Set the start for aggregating benchmark results')

    parser.add_argument('--save', action='store_true', help='Flag to indicate if save results')
    parser.add_argument('--arm', action='store_true', help='Flag to indicate if use Arm64 Arch')

    # add argument for sparsity
    parser.add_argument('--sparsity', type=float, help='Sparsity of the model', default=0.0)

    # number of cpu cores
    parser.add_argument('--cores', type=int, help='Number of CPU cores to use', default=32)


    args = parser.parse_args()

    if args.list:
        show_models()
        sys.exit()

    if not args.model or args.size is None:
        parser.error('--model and --size are required for benchmarking.')

    if args.model not in params:
        print ("Please check the model name by using '--list'")
        sys.exit()

    # limit the number of cores
    selected_cpus = select_k_cpus_with_lowest_load(args.cores)
    print(f"Selected {args.cores} CPUs with lowest load:", selected_cpus)
    # Get the process ID of the current process
    pid = os.getpid()
    set_cpu_affinity(pid, selected_cpus)

    if args.model == "resnet20":
        layers = [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64]
    elif args.model == "effnetb0":
        layers = [11, 11, 11]
    else:
        layers = [int(x) for x in args.model.split("_")]
    model_path = "../../models/"

    if not args.save:
        args.save = False

    start = 0
    notes = ""
    if args.agg:
        start = args.agg
        notes = f'start from {start}'

    if args.model == "resnet20" or args.model == "effnetb0":
        cnn = True
    elif layers[0] > 30:
        dnn = True
    else:
        dnn = False

    if args.arm:
        # circuit_folder = "./bin/m1_mac/arm_64/"
        circuit_folder = "./bin/m1_mac/"
    else:
        circuit_folder = "./bin/"

    if args.model == "resnet20":
        print(" MAKE SURE TO RUN convert_onnx_to_tflite FIRST TO SAVE THE MODEL IN tf_lite FORMAT")
        arch_folder = "resnet20/"
        os.makedirs(model_path + arch_folder, exist_ok=True)
        model_in_path = model_path + arch_folder + args.model + '.tflite'
        interpreter = tf.lite.Interpreter(model_path=model_in_path)
        interpreter.allocate_tensors()
        tests, true_labels = cnn_datasets(dataset_name="cifar100",args=args)
        predicted_labels = get_predictions(interpreter, tests)
    elif args.model == "effnetb0":
        print(" MAKE SURE TO RUN convert_onnx_to_tflite FIRST TO SAVE THE MODEL IN tf_lite FORMAT")
        arch_folder = "effnetb0/"
        os.makedirs(model_path + arch_folder, exist_ok=True)
        model_in_path = model_path + arch_folder + args.model + '.tflite'
        interpreter = tf.lite.Interpreter(model_path=model_in_path)
        interpreter.allocate_tensors()
        tests, true_labels = cnn_datasets(dataset_name="imagenet",args=args)
        predicted_labels = get_predictions(interpreter, tests)
    elif dnn:
        arch_folder = "input" + (len(layers)-1) * "-dense" + "/"
        model_path = "../../models/"
        model_in_path = model_path+arch_folder+args.model + '.tflite'
        interpreter = tf.lite.Interpreter(model_path=model_in_path)
        interpreter.allocate_tensors()
        if layers[0] == 784:
            tests, _ = dnn_datasets()
        else:
            _, tests = dnn_datasets()
        predicted_labels = get_predictions(interpreter, tests)
    else:
        arch_folder = arch_folders[args.model]
        model_in_path = model_path + arch_folder + args.model +'.tflite'

        if args.sparsity>0:
            # save a model with sparsity
            model_in_path = model_path+arch_folder+args.model + '_sparsity_'+str(args.sparsity) + '.tflite'
            print(f" === Using sparsity model: {model_in_path}")
            # append sparsity to notes
            notes += f' with sparsity {args.sparsity}'

        interpreter = tf.lite.Interpreter(model_path=model_in_path)
        interpreter.allocate_tensors()

        if layers[0] == 28:
            tests, _ = cnn_datasets(args=args)
        else:
            _, tests = cnn_datasets(args=args)
        predicted_labels = get_predictions(interpreter, tests)


    benchmark(tests[start:start+args.size], predicted_labels[start:start+args.size], args.model, model_in_path, circuit_folder, save=args.save, notes = notes, labels=true_labels[start:start+args.size] if true_labels is not None else None)