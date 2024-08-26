import argparse
import torch
import nevergrad as ng

from matplotlib import pyplot as plt
from torch import nn
from neuralteleportation.layers.layer_utils import swap_model_modules_for_COB_modules
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

from neuralteleportation.models.model_zoo.mlpcob import MLPCOB
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.layers.neuralteleportation import COBForwardMixin, FlattenCOB
from neuralteleportation.layers.neuron import LinearCOB
from neuralteleportation.layers.activation import ReLUCOB, SigmoidCOB, GELUCOB, LeakyReLUCOB
from neuralteleportation.layers.neuron import LayerNormCOB

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.flatten = FlattenCOB()
        self.norm = LayerNormCOB(784)
        self.fc1 = LinearCOB(784, 128)
        self.relu0 = GELUCOB()
        self.fc2 = LinearCOB(128, 10)

    def forward(self, x):
        x1 = self.flatten(x)
        x1 = self.norm(x1)
        x2 = self.fc1(x1)
        x3 = self.relu0(x2)
        x4 = self.fc2(x3)
        return x4


def load_model_weights(model, path, revere=False):
    # load model weights
    name_dict = {
        '1': 'norm',
        '2': 'fc1',
        '4': 'fc2',
    }

    state_dict = torch.load(path)
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
    model.load_state_dict(new_state_dict)
    return model

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000, help='Number of optimization steps')
    parser.add_argument("--seed", type=int, default=1234, help='Seed for torch random')
    parser.add_argument("--lr", type=float, default=1e-3, help='Learning rate for cob optimizer')
    parser.add_argument("--cob_range", type=float, default=1,
                        help='Range for the teleportation to create target weights')
    parser.add_argument("--center", type=float, default=1,
                        help='Center for the teleportation to create target weights')
    parser.add_argument("--sample_type", type=str, default='intra_landscape', help='Sampling type for cob')

    return parser.parse_args()

def train_model(model, train_loader, optimizer, criterion, device):
        model.train()
        for epoch in range(1):
            for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
        return model

# Hook function
def activation_hook(layer_name, activation_stats):
    def hook(module, input, output):
        input_tensor = input[0]
        activation_stats[layer_name] = {
            # l1 norm
            'norm': input_tensor.norm(),
            # 'weighted_norm': (input_tensor / module.cob).max() if hasattr(module, 'cob') else None,
            'max': input_tensor.max(),
            'min': input_tensor.min(),
            # 'cob': module.cob if hasattr(module, 'cob') else None,
            # 'index_of_max': input_tensor.argmax(),
            # 'index_of_min': input_tensor.argmin()
        }
        # print("Debug - activation_hook: ", input_tensor.max(), input_tensor.min())

    return hook

def compute_loss(model, cob, input_data, original_pred, activation_stats):
    # Set up model with the new COB
    model = LinearNet()
    model = NeuralTeleportationModel(model, input_shape=(1, 1, 28, 28))
    load_model_weights(model.network, 'model_weights_cob_activation_norm.pth')
    model.set_weights(initial_weights)
    
    model = model.teleport(cob, reset_teleportation=True)

    # Reset activation stats and run a forward pass
    activation_stats = {}
    for i, layer in enumerate(model.network.children()):
            if isinstance(layer, nn.ReLU) or isinstance(layer, ReLUCOB) or isinstance(layer, SigmoidCOB) or isinstance(layer, nn.Sigmoid) or isinstance(layer, GELUCOB) or isinstance(layer, nn.GELU) or isinstance(layer, LeakyReLUCOB) or isinstance(layer, nn.LeakyReLU):
                layer.register_forward_hook(activation_hook(f'relu_{i}',activation_stats=activation_stats))
    model.eval()
    pred = model.network(input_data)
    model.train()

    # Compute loss based on activation stats
    loss = sum([stats['max'] - stats['min'] for stats in activation_stats.values()])
    # second term of loss - difference between the cob and the ones tensor
    # loss += 10 * (cob - torch.ones_like(cob)).abs().mean()
    pred_error = np.absolute(original_pred - pred.detach().cpu().numpy()).mean() 
    loss += 10 * pred_error

    return loss, pred_error

if __name__ == '__main__':
    args = argument_parser()
    torch.manual_seed(args.seed)

    # Load or train your model
    model = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm(784),
        nn.Linear(784, 128),
        nn.GELU(),
        nn.Linear(128, 10)
    )

    if os.path.exists('model_weights_cob_activation_norm.pth'):
        model.load_state_dict(torch.load('model_weights_cob_activation_norm.pth'))
    else:
        # Train model by MNIST dataset
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 64
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        model = train_model(model, train_loader, optimizer, criterion, device)
        torch.save(model.state_dict(), 'model_weights_cob_activation_norm.pth')

    # Register hooks to the layers before all activation functions
    activation_stats = {}
    for i, layer in enumerate(model.children()):
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Sigmoid) or isinstance(layer, nn.GELU) or isinstance(layer, nn.LeakyReLU):
            layer.register_forward_hook(activation_hook(f'relu_{i}', activation_stats=activation_stats))

    # Prepare input data and initial predictions
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    random_index = np.random.randint(0, len(test_dataset))
    input_data = test_dataset[random_index][0]
    original_pred = model(input_data).detach().cpu().numpy()
    original_loss = sum([stats['max'] - stats['min'] for stats in activation_stats.values()])
    print(f"Original loss: {original_loss}, Original prediction error: {0}")

    torch.onnx.export(model, input_data, 'model_weights_cob_activation_norm.onnx', verbose=False, export_params=True, opset_version=12, do_constant_folding=True, input_names=['input_0'], output_names=['output'])
    np.save("input_data.npy", input_data.numpy())

    import time
    time.sleep(2)

    # Setup the NeuralTeleportationModel
    model = LinearNet()
    model = NeuralTeleportationModel(model, input_shape=(1, 1, 28, 28))
    load_model_weights(model.network, 'model_weights_cob_activation_norm.pth')
    # model.network.load_state_dict(torch.load('model_weights_cob_activation_norm.pth'))

    # Get initial weights
    initial_weights = model.get_weights().detach()
    initial_cob = model.generate_random_cob(cob_range=args.cob_range, requires_grad=True,center=args.center,sampling_type=args.sample_type)

    # define global variable to store the best loss found
    global best_loss
    # initialize best_loss
    best_loss = 1e9

    # Define the parameter space for the COB
    parametrization = ng.p.Instrumentation(
        ng.p.Array(shape=(initial_cob.size()),lower=0,upper=2)
    )
    # Define Nevergrad optimizer
    optimizer = ng.optimizers.CMA(parametrization=parametrization, budget=args.steps)
    optimizer.suggest(np.ones(initial_cob.size()))

    # Define the function to minimize using Nevergrad
    def ng_loss_function(cob_flat):
        # cob = cob_flat.reshape(initial_weights.shape)
        # print("Debug - cob_flat: ", cob_flat)
        cob = torch.tensor(cob_flat)
        loss, pred_error = compute_loss(model, cob, input_data, original_pred, activation_stats)
        print(f"Loss: {loss}, Prediction Error: {pred_error}")

        global best_loss
        if best_loss is None or loss < best_loss:
            best_loss = loss

        return loss.item()

    # Perform optimization using Nevergrad
    recommendation = optimizer.minimize(ng_loss_function)

    # Extract the best COB found
    best_cob = recommendation.value[0][0]
    best_cob = torch.tensor(best_cob)
    print(f"Best COB found: {best_cob}")

    # loss associated with the best COB
    # best_loss = recommendation.loss
    print(f"Loss associated with the best COB: {best_loss}")

    # Apply best COB and save model weights
    model = model.teleport(best_cob, reset_teleportation=True)
    torch.save(model.network.state_dict(), 'model_weights_cob_activation_norm_teleported.pth')

    # Export the optimized model to ONNX
    torch.onnx.export(model.network, input_data, 'model_weights_cob_activation_norm_teleported.onnx', verbose=False, export_params=True, opset_version=12, do_constant_folding=True, input_names=['input_0'], output_names=['output'])

    # Print final results
    print("Best COB found using Nevergrad.")
    
    
    # # Plotting history (if needed)
    # plt.figure()
    # history = np.array(loss_values)  # Assuming _all_values records loss history
    # plt.plot(history, color='blue', label='History')
    # plt.xlabel("Steps")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()
