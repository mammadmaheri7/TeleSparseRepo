# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision

def activation_hook(layer_name):
    def hook(module, input, output):
        input_tensor = input[0]
        activation_stats[layer_name] = {
            # l1 norm
            'norm': input_tensor.norm(),
            'max': input_tensor.max().item(),
            'min': input_tensor.min().item()
        }
    return hook

def load_model_weights(model, path, revere=False):
    # load model weights
    name_dict = {
        '1': 'fc1',
        '3': 'fc2',
        '5': 'fc3'
    }

    state_dict = torch.load(path)
    # print all state_dict keys
    print(state_dict)
    new_state_dict = {}
    for key, value in state_dict.items():
        # remove the 'network.' prefix if it exists
        if key.startswith('network.'):
            key = key.replace('network.', '')
        # get the first part of the key (before the dot)
        key_first = key.split('.')[0]
        if revere:
            # get the key corrosponding to the key_first (value is the key_first)
            new_key = [k for k, v in name_dict.items() if v == key_first]             # len should be 1
            assert len(new_key) == 1
            new_key = new_key[0]
            new_key = new_key + '.' + '.'.join(key.split('.')[1:])
        else:
            new_key = name_dict[key_first] + '.' + '.'.join(key.split('.')[1:])
        new_state_dict[new_key] = value
    print(new_state_dict)
    model.load_state_dict(new_state_dict)
    return model

input_data = np.load('input_data.npy')
input_data = torch.tensor(input_data)
print(input_data.shape)
model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        # nn.ReLU(),
        nn.Sigmoid(),
        nn.Linear(128, 64),
        # nn.ReLU(),
        nn.Sigmoid(),
        nn.Linear(64, 10)
    )
# model.load_state_dict(torch.load('model_weights_cob_activation_norm_teleported.pth'))
load_model_weights(model, 'model_weights_cob_activation_norm_teleported.pth',revere=True)
# Register hooks to the layers before all activation functions
activation_stats = {}
for i, layer in enumerate(model):
    if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Sigmoid):
        model[i].register_forward_hook(activation_hook(f'relu_{i}'))
model.eval()
pred = model(input_data)
for layer, stats in activation_stats.items():
    print(f'Layer {layer}: {stats}')