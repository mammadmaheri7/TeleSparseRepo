import itertools
import torch
import random
import math
import numpy as np
import os
from torch import nn
import torch.nn.functional as Fhtop
import argparse
import ezkl
import torch.nn as nn
from functools import partial
from timm.models.registry import register_model

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
model_zoo_path = os.path.join(current_dir, "../models/model_zoo")
sys.path.append(os.path.abspath(model_zoo_path))
from mobilenetv1cob import MobileNetV1COB,Struct, adjust_state_dict_keys, Block, MobileNetV1COB

from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.neuron import BatchNorm2dCOB
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

# define global variable to store the best loss found
global best_loss
global cor_best_pred_error
global cor_best_range
# initialize best_loss
best_loss = 1e9


import copy
import torch
import torch.nn as nn
import numpy as np
import random
import functools
import torch.multiprocessing as mp
import time
import json
import onnx
import csv


# Activation hook for storing activation statistics
def activation_hook(name, activation_stats, activations_output=None, layer_idx=None):
    def hook(module, input, output):
        input_tensor = input[0]
        activation_stats[name] = {'min': input_tensor.min().item(), 'max': input_tensor.max().item()}
        if activations_output is not None:
            # key of the dict is the layer number (extracted from the name)
            activations_output[layer_idx] = output
    return hook

# The function that calculates loss based on COB and runs inference
def f_ack(cob, input_data=None, original_pred=None, layer_idx=None, original_loss=None, tm=None, activation_orig=None, grad_orig=None, hessian_sensitivity=False, args=None):    
    # Set up model with the new COB
    teleported_model = tm
    # Apply the COB
    teleported_model = teleported_model.teleport(cob, reset_teleportation=False)

    # Reset activation stats and run a forward pass
    activation_stats = {}
    activations_quant = {}

    hook_handles = []
    for i, layer in enumerate(teleported_model.network.children()):
        if isinstance(layer, (nn.ReLU, ReLUCOB)):
            handle = layer.register_forward_hook(activation_hook(f'relu_{i}', activation_stats=activation_stats, activations_output=activations_quant,layer_idx=layer_idx))
            hook_handles.append(handle)

    teleported_model.eval()
    with torch.no_grad():
        pred = teleported_model.network(input_data)

    for handle in hook_handles:
        handle.remove()

    # Calculate the range loss
    # loss = sum([stats['max'] - stats['min'] for stats in activation_stats.values()])
    all_min = [stats['min'] for stats in activation_stats.values()]
    all_max = [stats['max'] for stats in activation_stats.values()]
    loss = max(all_max) - min(all_min)
    loss /= original_loss
    # Calculate the prediction error
    # pred_error = np.abs(original_pred - pred.detach().cpu().numpy()).mean()

    if hessian_sensitivity:
        pred_error = 0.0
        activation_quant = activations_quant[layer_idx]
        # assert that the activation_quant has only one key
        assert len(activations_quant.keys()) == 1
        # print("debug - key: ", activation_quant.shape)
        # Compute the difference between activations
        delta = activation_quant - activation_orig
        # Compute the squared gradients
        grad_squared = grad_orig.pow(2)
        # Compute the element-wise product
        elementwise_product = delta.pow(2) * grad_squared
        # Sum over all elements to get the loss for this layer
        # print("debug - elementwise_product: ", elementwise_product.shape)
        layer_loss = elementwise_product.sum()
        # Accumulate the total prediction error
        pred_error += layer_loss.item()
        if random.random() < 0.0005:
            print(f"debug - grad_squared_norm: {grad_squared.norm()} \t delta2_norm: {delta.pow(2).norm()} \t layer_loss: {layer_loss}")
    else:
        pred_error = np.abs(original_pred - pred).mean()
        pred_error /= np.abs(original_pred).mean()
        pred_error = pred_error.item()
    # TODO: change the 10 with args.pred_mul (adding that to the function signature)
    total_loss = loss + args.pred_mul * pred_error

    if random.random() < 0.001:
        print(f"pred_error: {pred_error} \t range_loss: {loss}")

    # Undo the teleportation
    teleported_model.undo_teleportation()
    return total_loss, loss, pred_error

def worker_func_batch(args):
    idx_batch, key, base, params_dict, step_size, func = args
    perturbed_params_dict = copy.deepcopy(params_dict)
    # perturbed_params_dict = params_dict
    p_flat = perturbed_params_dict[key].flatten()
    grads = []
    
    # Compute gradients for each perturbation in the batch
    for idx in idx_batch:
        p_flat[idx] += step_size
        out,_,_ = func(perturbed_params_dict["cob"])
        directional_derivative = (out - base) / step_size
        grads.append((idx, directional_derivative))
        p_flat[idx] -= step_size  # Reset the perturbation

    return grads

# Batched CGE using multiprocessing
@torch.no_grad()
def cge_batched(func, params_dict, mask_dict, step_size, pool, base=None, num_process=4, ignoring_indices=None):
    if base is None:
        base,_,_ = func(params_dict["cob"])

    grads_dict = {}
    for key, param in params_dict.items():
        if 'orig' in key:
            mask_key = key.replace('orig', 'mask')
            mask_flat = mask_dict[mask_key].flatten()
        else:
            mask_flat = torch.ones_like(param).flatten()

        directional_derivative = torch.zeros_like(param)
        directional_derivative_flat = directional_derivative.flatten()

        # check if ignoring_indices is not None
        if ignoring_indices is not None:
            mask_flat[ignoring_indices] = 0

        # set 50 percent of the non-zero mask to zero
        non_zero_mask = (mask_flat != 0).float()
        dropout_mask = torch.bernoulli(non_zero_mask * 0.9)
        mask_flat = mask_flat * dropout_mask
        # mask_flat = mask_flat * torch.bernoulli(torch.full_like(mask_flat, 0.5))


        # Prepare batches of indices
        idx_list = mask_flat.nonzero().flatten().tolist()
        # batch_size = len(idx_list) // num_process
        # check whether the batch size is dividable by the number of processes to make sure that each process gets the same number of indices
        if len(idx_list) % num_process != 0:
            batch_size = len(idx_list) // num_process + 1
        else:
            batch_size = len(idx_list) // num_process
        batches = [idx_list[i:i + batch_size] for i in range(0, len(idx_list), batch_size)]
        
        # Create task arguments for each batch
        tasks = [(batch, key, base, params_dict, step_size, func) for batch in batches]

        # Use the already initialized pool to run the worker_func_batch in parallel
        results = pool.map(worker_func_batch, tasks)

        # Collect results from all workers and update the directional_derivative tensor
        for result_batch in results:
            for idx, grad in result_batch:
                directional_derivative_flat[idx] = grad

        grads_dict[key] = directional_derivative.to(param.device)

    return grads_dict

# Training loop using the persistent pool
def train_cob(input_teleported_model,input_orig_model, original_pred, layer_idx, original_loss_idx, LN, args, activation_orig = None, grad_orig = None):
    # initial_cob_idx = torch.ones(960)  # Initial guess for COB
    cob_size, index_conv2d = LN.get_cob_size(return_index_conv2d=True)
    # flatten the index_conv2d
    index_conv2d = [item for sublist in index_conv2d for item in sublist]
    initial_cob_idx = torch.ones(cob_size)  # Initial guess for COB

    # Prepare the function for constrained gradient estimation
    ackley = functools.partial(
        f_ack,
        input_data=input_orig_model,
        original_pred=original_pred,
        layer_idx=layer_idx,
        original_loss=original_loss_idx,
        tm=LN,
        activation_orig = activation_orig,
        grad_orig = grad_orig,
        hessian_sensitivity = args.hessian_sensitivity,
        args=args
    )

    eval_ackley = functools.partial(
        f_ack,
        input_data=input_teleported_model,
        original_pred=original_pred,
        layer_idx=layer_idx,
        original_loss=original_loss_idx,
        tm=LN,
        hessian_sensitivity = args.hessian_sensitivity,
        args=args
    )

    best_cob = None
    # best_loss = float('inf')
    num_process = 2

    with mp.Manager() as manager:
        best_loss = manager.Value('d', float('inf'))  # Shared float variable for the best loss
        cor_best_pred_error = manager.Value('d', 0.0)  # Shared float variable for prediction error
        cor_best_range = manager.Value('d', 0.0)  # Shared float variable fo

        # compute the loss before the optimization
        loss,r_error,p_error = ackley(initial_cob_idx)
        print(f"Initial Loss: {loss} \t P_E: {p_error} \t R_E: {r_error}")

        # Initialize the process pool once and reuse it for all iterations
        with mp.Pool(num_process) as pool:
            # Training loop to optimize COB
            for step in range(args.steps):
                # Get the gradient of the COB using the batched CGE with persistent pool
                t0 = time.time()
                grad_cob = cge_batched(ackley, 
                                       {"cob": initial_cob_idx}, 
                                       None, args.zoo_step_size, pool, num_process=num_process,
                                       ignoring_indices=index_conv2d)
                t1 = time.time()
                # Update the COB using gradient descent
                if not args.hessian_sensitivity:
                    initial_cob_idx -= args.cob_lr * grad_cob["cob"]
                else:
                    # normalize the gradient
                    update = grad_cob["cob"] / grad_cob["cob"].norm()
                    initial_cob_idx -= args.cob_lr * update
                t2 = time.time()

                # check that all cob should be positive
                initial_cob_idx = torch.clamp(initial_cob_idx, min=1e-5)

                print("stats updated cob: min: ",initial_cob_idx.min().item()," max: ",initial_cob_idx.max().item())

                # Calculate the loss with the updated COB
                loss,r_error,p_error = eval_ackley(initial_cob_idx)
                t3 = time.time()

                # Update the best loss and COB if the current loss is better
                if loss < best_loss.value:
                    best_loss.value = loss
                    best_cob = initial_cob_idx.clone()  # Save the best COB
                    cor_best_pred_error.value = p_error
                    cor_best_range.value = r_error

                    # print(f"Step: {step} \t Loss: {loss}")

                print(f"Step: {step} \t Loss: {loss} \t \t P_E: {p_error} \t R_E: {r_error} \t  \t Time: {t1-t0}")
            

        return best_cob, best_loss.value, cor_best_range.value, cor_best_pred_error.value


def set_onnx_size(args, BATCHS, onnx_path):
    on = onnx.load(onnx_path)
    for tensor in on.graph.input:
        for dim_proto in tensor.type.tensor_type.shape.dim:
            print("dim_proto:",dim_proto)
            if dim_proto.HasField("dim_param"): # and dim_proto.dim_param == 'batch_size':
                dim_proto.Clear()
                dim_proto.dim_value = BATCHS   # fixed batch size
    for tensor in on.graph.output:
        for dim_proto in tensor.type.tensor_type.shape.dim:
            if dim_proto.HasField("dim_param"):
                dim_proto.Clear()
                dim_proto.dim_value = BATCHS   # fixed batch size

    onnx.save(on, onnx_path)
    on = onnx.load(onnx_path)
    on = onnx.shape_inference.infer_shapes(on)
    onnx.save(on, onnx_path)

if __name__ == '__main__':
    # set spawn start method
    mp.set_start_method('spawn', force=True)
    # args = get_default_args()
    # print(args.__dict__)

    # default_model = "vit_tiny"
    # default_data_path = "/rds/general/user/mm6322/home/imagenet"
    # default_resume = "./sparse-cap-acc-tmp/deit_tiny_patch16_224_sparsity=0.50_best.pth"
    # default_resume = "/rds/general/user/mm6322/home/verifiable_NN_ezkl/examples/notebooks/CAP_pruned_models/Checkpoints/deit_tiny_patch16_224_sparsity=0.50_best.pth"
    # default_sparsity = 0.5
    # default_batch_size = 1
    # default_pruning_method = "CAP"
    # default_prefix_dir = "sparse-cap-acc-tmp/"
    default_input_param_scale = 7
    default_log_rows = 20
    default_num_cols = 2
    default_scale_rebase_multiplier = 1
    defualt_hessian_sensitivity = False

    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--model', default=default_model, type=str, help='Model type')
    # parser.add_argument('--data_path', default=default_data_path, type=str, help='Data path')
    # parser.add_argument('--resume', default=default_resume, type=str, help='Resume path')
    # parser.add_argument('--sparsity', default=default_sparsity, type=float, help='Sparsity value')
    # parser.add_argument('--batch_size', default=default_batch_size, type=int, help='Batch size')
    # parser.add_argument('--pruning_method', default=default_pruning_method, type=str, help='Pruning method')
    # parser.add_argument('--prefix_dir', default=default_prefix_dir, type=str, help='Prefix directory')
    # parser.add_argument('--input_param_scale', default=default_input_param_scale, type=int, help='Input parameter scale')
    # parser.add_argument('--log_rows', default=default_log_rows, type=int, help='Log rows')
    # parser.add_argument('--num_cols', default=default_num_cols, type=int, help='Number of columns')
    # parser.add_argument('--scale_rebase_multiplier', default=default_scale_rebase_multiplier, type=int, help='Scale rebase multiplier')
    # args = parser.parse_args(namespace=args)
    # args.model = default_model
    # args.data_path = default_data_path
    # args.resume = default_resume
    # args.sparsity = default_sparsity
    # args.batch_size = default_batch_size
    # args.pruning_method = default_pruning_method
    # args.prefix_dir = default_prefix_dir
    # args.input_param_scale = default_input_param_scale
    # args.log_rows = default_log_rows
    # args.num_cols = default_num_cols
    # args.scale_rebase_multiplier = default_scale_rebase_multiplier
    # args.hessian_sensitivity = defualt_hessian_sensitivity
    parser = argparse.ArgumentParser(description='Teleportation of a ResNet20 model trained on CIFAR-100')
    parser.add_argument('--teleport_dense_model', type=bool, default=False, help='Teleport the dense model')
    args = parser.parse_args()

    # args = argparse.Namespace()
    args.batch_size = 1

    if not args.teleport_dense_model:
        args.pretrained_model_path = '../models/model_zoo/mobilenetv1_sparse_best.pth'
    else:
        args.pretrained_model_path = '../models/model_zoo/mobilenetv1_dense_best.pt'
    args.prefix_dir = "mobilenetv1_teleport_ZO_temp/"
    args.hessian_sensitivity = False

    # Create the prefix directory if it does not exist
    os.makedirs(args.prefix_dir, exist_ok=True)
    BATCHS = args.batch_size

    # model = build_model(args, pretrained=False)
    pretrained_sparse_model = torch.load(args.pretrained_model_path, map_location='cpu')['model'] if not args.teleport_dense_model \
          else torch.load(args.pretrained_model_path, map_location='cpu') # Named sparse but could be dense (depends on the args.teleport_dense_model)
    # retrieve the cfg of the pretrained model
    retrieved_cfg = []
    for module in pretrained_sparse_model.children(): 
        if isinstance(module, nn.Conv2d):
            retrieved_cfg.append((module.out_channels, module.stride[0]))
        elif isinstance(module, nn.Sequential):
            for sub_module in module.children():
                if isinstance(sub_module, nn.Conv2d):
                    retrieved_cfg.append((sub_module.out_channels, sub_module.stride[0]))
                elif isinstance(sub_module, Block):
                    retrieved_cfg.append((sub_module.conv2.out_channels, sub_module.conv1.stride[0]))
    print("retrieved_cfg: ",retrieved_cfg)
    mobilenet_cob = MobileNetV1COB(num_classes=10, cfg=retrieved_cfg)
    print("mobilenet_cob: ",mobilenet_cob)
    # loading pretrained weights
    sparse_state_dict = pretrained_sparse_model.state_dict()
    new_sparse_state_dict = adjust_state_dict_keys(sparse_state_dict)
    mobilenet_cob.load_state_dict(new_sparse_state_dict, strict=True)
    model = mobilenet_cob
    model.eval()
    print("model - after eval: ",model)
    print("pretrained_sparse_model - before eval: ",pretrained_sparse_model)
    pretrained_sparse_model.eval()
    print("pretrained_sparse_model - after eval: ",pretrained_sparse_model)


    # set eps to 1e-5
    model.bn1.eps=1e-5
    # do the set for all bn in model layers
    for l_index in range(13):
        cob_bn = getattr(model, f"layer{l_index}_bn1")
        cob_bn.eps = 1e-5
        setattr(cob_bn, "eps", 1e-5)
        cob_bn = getattr(model, f"layer{l_index}_bn2")
        cob_bn.eps = 1e-5
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # test model works like the pretrained model
    input = torch.randn(1, 3, 32, 32)
    output_model = model(input)
    output_pretrained = pretrained_sparse_model(input)
    print("DIFF output_model - output_pretrained: ",(output_model - output_pretrained).abs().max())
    assert (output_model - output_pretrained).abs().max() < 1e-5
    del pretrained_sparse_model

    # compute the number of zeros in the model / total number of parameters
    total_params = 0
    zeros = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        zeros += (param.data == 0).sum().item()
    print(f"Total number of parameters: {total_params}")
    print(f"Number of zeros: {zeros}")
    print(f"Percentage of zeros: {zeros / total_params * 100:.2f}%")

    model = model.cpu()
    model.eval()

    if not args.hessian_sensitivity:
        for param in model.parameters():
            param.requires_grad_(False)

    input = torch.randn(1, 3, 32, 32)
    x = input.detach().clone()
    print("x.shape:",x.shape)

    onnx_path = args.prefix_dir + "network_complete.onnx" if not args.teleport_dense_model else args.prefix_dir + "network_complete_dense.onnx"

    # Export the model
    torch.onnx.export(    
        model,               # model being run
        x,                   # model input (or a tuple for multiple inputs)
        onnx_path,            # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=15,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                        'output': {0:'batch_size'},
        },

    )

    # makin the network_complete.onnx fixed batch size
    on = onnx.load(onnx_path)
    for tensor in on.graph.input:
        for dim_proto in tensor.type.tensor_type.shape.dim:
            print("dim_proto:",dim_proto)
            if dim_proto.HasField("dim_param"): # and dim_proto.dim_param == 'batch_size':
                dim_proto.Clear()
                dim_proto.dim_value = BATCHS   # fixed batch size
    for tensor in on.graph.output:
        for dim_proto in tensor.type.tensor_type.shape.dim:
            if dim_proto.HasField("dim_param"):
                dim_proto.Clear()
                dim_proto.dim_value = BATCHS   # fixed batch size
    onnx.save(on, onnx_path)
    on = onnx.load(onnx_path)
    on = onnx.shape_inference.infer_shapes(on)
    onnx.save(on, onnx_path)

    # generate data for all layers
    if not args.teleport_dense_model:
        data_path = os.path.join(os.getcwd(),args.prefix_dir, "input_convs.json")
        data = dict(input_data = [((x).detach().numpy()).reshape([-1]).tolist()])
        json.dump( data, open(data_path, 'w' ))
    else: 
        data = json.load(open(os.path.join(os.getcwd(),args.prefix_dir, "input_convs.json")))

    # Define the CSV file and write the header if it doesn't exist
    csv_file_path = args.prefix_dir + 'ZO-accuracy.csv'
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'EXPERIMENT SETTINGS',
                'Layer_Index', 'Sample_Name',
                'Activation_Loss_Org', 'Range_Error' ,'Prediction_Error',
        ])
            
    array_param_visibility = ["fixed"]
    array_input_param_scale = [16]
    array_num_cols = [64]
    array_max_log_rows = [16]
    array_scale_rebase = [1]
    array_lookup_margin = [2]

    # iterate over all the possible combinations
    combinations = list(itertools.product(array_param_visibility, array_input_param_scale, array_num_cols, array_max_log_rows, array_scale_rebase, array_lookup_margin))
    if not args.hessian_sensitivity:
        list_of_no_teleportation = None
    else:
        list_of_no_teleportation = []

    # save 10 random images of CIFAR10 in the prefix_dir/images
    import torchvision.transforms as transforms
    from PIL import Image
    import os
    import random
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision
    from torch.utils.data import DataLoader


    os.makedirs(args.prefix_dir + "images", exist_ok=True)

    if not args.teleport_dense_model:
        # remove previous images
        for file in os.listdir(args.prefix_dir + "images"):
            os.remove(os.path.join(args.prefix_dir + "images", file))
    # download the CIFAR10 dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=1, shuffle=True)

    if not args.teleport_dense_model:
        for i, data in enumerate(testloader):
            if i == 10:
                break
            img, target = data
            img = img.squeeze(0)
            # img = img.permute(1, 2, 0)
            assert img.shape == (3, 32, 32)
            img = img.numpy()
            # plt.imsave(args.prefix_dir + f"images/{i}.JPEG", img) 
            np.save(args.prefix_dir + f"images/{i}.npy", img)


    list_jpeg = os.listdir(args.prefix_dir + "images")
    list_jpeg = [x for x in list_jpeg if x.endswith(".npy")]

    # computing accuracy of the teleportation
    teleport_correct = 0
    teleport_total = 0

    print("========= START =========")
    print(model)
    print("========= END =========")

    # with no gradient pytorch
    # with torch.no_grad():
    if True:
        # iterate over all the possible combinations
        for p in combinations:
            param_visibility, input_param_scale, num_cols, max_log_rows, scale_rebase, lookup_margin = p
            # string experiment_settings as comma separated values
            experiment_settings = f"{param_visibility}/{input_param_scale}/{num_cols}/{max_log_rows}/{scale_rebase}/{lookup_margin}"
            print("========= START =========")
            print(f"input_param_scale: {input_param_scale}, num_cols: {num_cols}, max_log_rows: {max_log_rows}, param_visibility: {param_visibility}, lookup_margin: {lookup_margin}")

            # copy the model
            model.eval()
            
            # generate compression-model and setting for all images among all layers
            list_jpeg = list(reversed(list_jpeg))

            for jpeg_path in list_jpeg:
                print("=== jpeg_path:",jpeg_path)
                # img = Image.open(args.prefix_dir + f"images/{jpeg_path}")
                img = np.load(args.prefix_dir + f"images/{jpeg_path}")
                img_name = os.path.splitext(jpeg_path)[0]
                # img = img.resize((32,32))
                assert img.shape == (3, 32, 32)

                # data = transforms.ToTensor()(img).unsqueeze(0)
                data = torch.tensor(img).unsqueeze(0).float()
                print("=== data.shape:",data.shape)

                # compute the output_orig and grad_orig dictionaries
                activations_orig = None
                gradients_orig = None

                if args.hessian_sensitivity:
                    # Define a function to create a forward hook that captures the layer index
                    model.train()
                    activations_orig = {}
                    gradients_orig = {}

                    def get_forward_hook(layer_idx):
                        def forward_hook_orig(module, input, output):
                            # Ensure the output requires gradients
                            output.requires_grad_(True)
                            activations_orig[layer_idx] = output
                        return forward_hook_orig

                    # Register forward hooks on the original model's activation layers
                    hook_handles = []
                    for layer_idx, block_orig in enumerate(model.blocks):
                        mlp_orig = block_orig.mlp
                        for layer_orig in mlp_orig.children():
                            if isinstance(layer_orig, (nn.ReLU, nn.Sigmoid, nn.GELU, nn.LeakyReLU)):
                                # Register a forward hook with the layer index
                                handle_orig = layer_orig.register_forward_hook(get_forward_hook(layer_idx))
                                hook_handles.append(handle_orig)
                    # Perform a forward pass through the original model
                    result = model(data)
                    # Define the loss function (assuming you have the true labels)
                    criterion = nn.CrossEntropyLoss()
                    # TODO: # Replace 'labels' with your actual target tensor
                    targets = result.argmax(dim=1)
                    # Compute the loss using the original model's output
                    loss_cls = criterion(result, targets)

                    # Compute the gradients w.r.t. the original model's activations
                    for layer_idx in activations_orig:
                        grad = torch.autograd.grad(loss_cls, activations_orig[layer_idx], retain_graph=True)[0]
                        gradients_orig[layer_idx] = grad.detach().cpu()
                    # detach all output stored in the dictionary
                    activations_orig = {key: value.detach().cpu() for key, value in activations_orig.items()}

                    # Print the shapes of the gradients and activations for verification
                    for layer_idx in gradients_orig:
                        print(f"Layer {layer_idx}: Gradient shape {gradients_orig[layer_idx].shape}, Activation shape {activations_orig[layer_idx].shape}")
                    # Disable gradients for all parameters of the original model
                    for param in model.parameters():
                        param.requires_grad_(False)
                    # Remove the hooks to prevent side effects
                    for handle in hook_handles:
                        handle.remove()
                    # print shape of the grad and output in the dictionaries
                    for key in gradients_orig:
                        print("key:",key,"\t grad.shape:",gradients_orig[key].shape,"\t output.shape:",activations_orig[key].shape)
                    model.eval()
                else:
                    with torch.no_grad():
                        # compute the range of the activations of the model (for all layers)
                        # 1. set the hooks
                        hook_handles = []
                        activation_stats_all = {}
                        for layer_idx, layer in enumerate(model.children()):
                            if isinstance(layer, (nn.ReLU, ReLUCOB)):
                                handle = layer.register_forward_hook(activation_hook(f'relu_{layer_idx}', activation_stats=activation_stats_all, layer_idx=layer_idx))
                                hook_handles.append(handle)

                        # inference on original model
                        result = model(data)

                        print("activation_stats_all:",activation_stats_all)
                        # 2. finding the range based on the hooks
                        range_list_all = {key : activation_stats_all[key]['max'] - activation_stats_all[key]['min'] for key in activation_stats_all.keys()}
                        print("range_list_all:",range_list_all)
                        print("min of all min:",min([stats['min'] for stats in activation_stats_all.values()]))
                        print("max of all max:",max([stats['max'] for stats in activation_stats_all.values()]))

                        # 3. remove the hooks
                        for handle in hook_handles:
                            handle.remove()
                            
                        # 4. define list of no teleportation
                        # topk = 3
                        # list_of_no_teleportation = [k for k, v in sorted(range_list_all.items(), key=lambda item: item[1])[:topk]]
                        # print("list_of_no_teleportation:",list_of_no_teleportation)
                       
                # inference on original model
                network_input_data = copy.deepcopy(data)

                # generate data for all layers
                data_path = os.path.join(os.getcwd(),args.prefix_dir, f"input_convs.json")
                data_dict = dict(input_data = [((data).detach().numpy()).reshape([-1]).tolist()])
                json.dump( data_dict, open(data_path, 'w' ))

                with torch.no_grad():
                    args.pred_mul = 0
                    args.steps = 50
                    args.cob_lr = 0.05
                    args.zoo_step_size = 0.0005

                    # max of all max - min of all min
                    all_min = [stats['min'] for stats in activation_stats_all.values()]
                    all_max = [stats['max'] for stats in activation_stats_all.values()]
                    original_loss_all_layers = max(all_max) - min(all_min)

                    # track best loss
                    best_loss = 1e9
                    cor_best_pred_error = 1e9
                    cor_best_range = 1e9

                    model.eval()
                    original_pred = model(network_input_data)

                    # Apply best COB and save model weights
                    LN = NeuralTeleportationModel(network = model, input_shape=(1, 3, 32, 32))
                    cob_size = LN.get_cob_size()
                    print("========= COB SIZE: ==========",cob_size)
                    LN.eval()

                    act_idx = activations_orig[layer_idx] if activations_orig is not None else None
                    grad_idx = gradients_orig[layer_idx] if gradients_orig is not None else None

                    best_cob,best_loss,cor_best_range,cor_best_pred_error = train_cob(data, data, original_pred, 0, original_loss_all_layers, LN, args, activation_orig=act_idx, grad_orig=grad_idx)
                    print("BEST LOSS:",best_loss)
                    LN = LN.teleport(best_cob, reset_teleportation=True)
                    # save the .pth of the teleported model
                    pth_path = args.prefix_dir + f"mobilenetv1_cob_activation_norm_teleported.pth" if not args.teleport_dense_model else args.prefix_dir + f"mobilenetv1_cob_activation_norm_teleported_dense.pth"
                    torch.save(LN.network.state_dict(), pth_path)

                    onnx_path = args.prefix_dir + f"mobilenetv1_cob_activation_norm_teleported.onnx" if not args.teleport_dense_model else args.prefix_dir + f"mobilenetv1_cob_activation_norm_teleported_dense.onnx"
                    # Export the optimized model to ONNX
                    export_model = LN.network.eval().cpu()
                    dummy_input = torch.randn(1, 3, 32, 32)

                    torch.onnx.export(export_model, dummy_input, onnx_path, verbose=False, 
                                      export_params=True, opset_version=15, do_constant_folding=True, 
                                      input_names=['input'], output_names=['output'],
                                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                                    'output': {0:'batch_size'}},
                                      )
                    # # making the mobilenetv1_cob_activation_norm_teleported.onnx fixed batch size
                    on = onnx.load(onnx_path)
                    for tensor in on.graph.input:
                        for dim_proto in tensor.type.tensor_type.shape.dim:
                            if dim_proto.HasField("dim_param"):
                                dim_proto.Clear()
                                dim_proto.dim_value = BATCHS
                    for tensor in on.graph.output:
                        for dim_proto in tensor.type.tensor_type.shape.dim:
                            if dim_proto.HasField("dim_param"):
                                dim_proto.Clear()
                                dim_proto.dim_value = BATCHS
                    onnx.save(on, onnx_path)
                    on = onnx.load(onnx_path)
                    on = onnx.shape_inference.infer_shapes(on)
                    onnx.save(on, onnx_path)

                    # check the validation of the teleportation
                    # 1.extract onnx corrosponding to the teleported model (in original onnx)
                    # input_path = args.prefix_dir + f"network_split_{layer_idx}_False.onnx"
                    # output_path = args.prefix_dir + f"block{layer_idx}_cob_activation_norm.onnx"
                    # input_names = [f"/blocks.{layer_idx}/Add_2_output_0"]
                    # output_names = [f"/blocks.{layer_idx}/mlp/fc2/Add_output_0"]
                    # onnx.utils.extract_model(input_path, output_path, input_names, output_names, check_model=True)
                
                    # write the results to the csv file
                    with open(csv_file_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            experiment_settings,
                            0, img_name,
                            original_loss_all_layers, cor_best_range, cor_best_pred_error,
                        ])
                    
                    print("\n\n")

                    # compute the new prediction of model (after all layers are teleported)
                    # new_pred = new_model(network_input_data)
                    new_pred = LN.network(data)
                    # check whether the teleportation is successful
                    print("MAX ARGUMENT NEW PREDICTION:", new_pred.argmax())
                    print("ORIGINAL PREDICTION:", result.argmax())
                    if new_pred.argmax() == result.argmax():
                        teleport_correct += 1
                    teleport_total += 1
                    norm1 = torch.norm(new_pred - result)
                    print("ACCURACY OF TELEPORTATION:", teleport_correct/teleport_total)
                    # log on the file txt (append the accuracy of teleportation + number of corrot and total)
                    with open(args.prefix_dir + "accuracy_teleportation.txt", "a") as f:
                        f.write(f"ACCURACY OF TELEPORTATION: {teleport_correct/teleport_total} \t CORRECT: {teleport_correct} \t TOTAL: {teleport_total} \t NORM1: {norm1}\n")
                    print("==========================")
                    time.sleep(2)
                    break