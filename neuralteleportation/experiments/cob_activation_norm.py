import argparse
import torch

from matplotlib import pyplot as plt
from torch import optim
from torch import nn
from neuralteleportation.layers.layer_utils import swap_model_modules_for_COB_modules
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

from neuralteleportation.models.model_zoo.mlpcob import MLPCOB
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import LinearCOB
from neuralteleportation.layers.activation import ReLUCOB

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.flatten = FlattenCOB()
        self.fc1 = LinearCOB(784, 128)
        self.relu1 = ReLUCOB()
        self.fc2 = LinearCOB(128, 64)
        self.relu2 = ReLUCOB()
        self.fc3 = LinearCOB(64, 10)

    def forward(self, x):
        x1 = self.flatten(x)
        x2 = self.fc1(x1)
        x3 = self.relu1(x2)
        x4 = self.fc2(x3)
        x5 = self.relu2(x4)
        x6 = self.fc3(x5)
        return x6

def load_model_weights(model, path, revere=False):
    # load model weights
    name_dict = {
        '1': 'fc1',
        '3': 'fc2',
        '5': 'fc3'
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
    # add argument for center
    parser.add_argument("--center", type=float, default=1,
                        help='Center for the teleportation to create target weights')

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

if __name__ == '__main__':
    args = argument_parser()

    torch.manual_seed(args.seed)

    # model = nn.Sequential(
    # nn.Conv2d(1, 32, 3, 1),
    # nn.ReLU(),
    # nn.Conv2d(32, 64, 3, stride=2),
    # nn.ReLU(),
    # nn.Flatten(),
    # nn.Linear(9216, 128),  # Use a larger layer before the final output
    # nn.ReLU(),
    # nn.Linear(128, 10)  # Directly output logits for 10 classes
    # )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    
    # check model already trained
    if os.path.exists('model_weights_cob_activation_norm.pth'):
        model.load_state_dict(torch.load('model_weights_cob_activation_norm.pth'))
    else:
        # trian model by mnist dataset
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
    for i, layer in enumerate(model):
        if isinstance(layer, nn.ReLU):
            model[i].register_forward_hook(activation_hook(f'relu_{i}'))

    # Test the model
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    # input_data = next(iter(test_loader))[0]
    # get random data from the test loader
    random_index = np.random.randint(0, len(test_dataset))
    input_data = test_dataset[random_index][0].unsqueeze(0)
    original_pred = model(input_data)

    # save the input data
    np.save("input_data.npy", input_data.numpy())
    
    for layer, stats in activation_stats.items():
        print(f'Layer {layer}: {stats}')
    
    # copy activations
    original_activations = {layer: stats for layer, stats in activation_stats.items()}
    orginial_loss = sum([stats['norm'] for stats in activation_stats.values()])
    original_pred = original_pred.detach().cpu().numpy()

    # model = swap_model_modules_for_COB_modules(model)
    model = LinearNet()
    model = NeuralTeleportationModel(model, input_shape=(1, 1, 28, 28))
    model.training = True
    
    # model.network.load_state_dict(torch.load('model_weights_cob_activation_norm.pth'))
    load_model_weights(model.network, 'model_weights_cob_activation_norm.pth')

    # Get the initial set of weights and teleport.
    initial_weights = model.get_weights().detach()

    # Generate a new random cob
    cob = model.generate_random_cob(cob_range=args.cob_range, requires_grad=True,center=args.center)
    # new_cob = cob.clone().detach().requires_grad_(True)

    history = []
    cob_error_history = []
    min_loss = sum([stats['norm'] for stats in activation_stats.values()])
    history.append(min_loss.item())
    best_cob = cob.clone().detach()

    for e in range(args.steps):
        print("\n === Step: ", e , " === ")
        model.set_weights(initial_weights)
        # cob = new_cob
        
        # Teleport with this cob
        model = model.teleport(cob,reset_teleportation=True)

        activation_stats = {}
        for i, layer in enumerate(model.network.children()):
            if isinstance(layer, nn.ReLU) or isinstance(layer, ReLUCOB):
                # model.network[i].register_forward_hook(activation_hook(f'relu_{i}'))
                layer.register_forward_hook(activation_hook(f'relu_{i}'))
        pred = model(input_data)

        # define loss as the value of the l1 norm of the activations
        loss = sum([stats['norm'] for stats in activation_stats.values()])
        print("Loss: ", loss.item(), "\t l1 norm of activations: ", [stats['norm'].item() for stats in activation_stats.values()])
        # print max and min of the activations
        print("Max of activations: ", [stats['max'] for stats in activation_stats.values()])
        print("Min of activations: ", [stats['min'] for stats in activation_stats.values()])

        # Backwards pass
        grad = torch.autograd.grad(loss, cob, create_graph=True)
        
        # compute the distance between original pred and pred (numpy has no abs method)
        print("Pred diff mean", np.absolute(original_pred - pred.detach().cpu().numpy()).mean())

        # grad[0] = grad[0] + 0.1 * grad[0].mean() * torch.randn_like(grad[0])
        # new_cob = cob.clone().detach() - args.lr * grad[0] / grad[0].norm()
        cob = cob - args.lr * grad[0] / (grad[0].norm() + 1e-8)
        # add a small noise to the cob
        cob = cob + 0.05 * cob.mean() * torch.randn_like(cob)
        # cob = cob - args.lr * grad[0]
        # set all new_cob element to be positive
        # new_cob = torch.abs(new_cob)
        cob = torch.abs(cob)
        cob.grad = None
        # reset the teleportation
        if e % 100 == 0:
            cob = model.generate_random_cob(cob_range=args.cob_range, requires_grad=True,center=args.center)
        
        history.append(loss.item())

        if loss < min_loss:
            print("----------- New best loss: ", loss.item())
            min_loss = loss
            best_cob = cob.clone().detach()
            # save the model weights
            torch.save(model.state_dict(), f'model_weights_cob_activation_norm_teleported.pth')


    print(" ====== Best COB ====== ")
    print(min_loss)
    # load the best model to check the activations
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    # model.load_state_dict(torch.load('model_weights_cob_activation_norm_teleported.pth'))
    load_model_weights(model, 'model_weights_cob_activation_norm_teleported.pth',revere=True)
    # Register hooks to the layers before all activation functions
    activation_stats = {}
    for i, layer in enumerate(model):
        if isinstance(layer, nn.ReLU):
            model[i].register_forward_hook(activation_hook(f'relu_{i}'))
    model.eval()
    pred = model(input_data)
    for layer, stats in activation_stats.items():
        print(f'Layer {layer}: {stats}')
    print("diff pred mean: ", np.absolute(original_pred - pred.detach().cpu().numpy()).mean())

    # original model
    print(" ====== Original COB ====== ")
    print(orginial_loss)
    for layer, stats in original_activations.items():
        print(f'Layer {layer}: {stats}')


    plt.figure()
    history = np.array(history)
    history = history[~np.isnan(history)]
    # set values bigger than twice the mean to be the twice of the mean
    mean_history = history.mean()
    history[history>2*mean_history] = 2*mean_history
    # plot the history with red color for values equal to twice the mean and blue color for other values
    plt.plot(history, color='blue', label='History')
    for i, value in enumerate(history):
        if value == 2 * mean_history:
            # set size point to be 10
            plt.plot(i, value, 'ro')  # Red color for points meeting the condition
        else:
            plt.plot(i, value, 'bo')  # Blue color for other points


    plt.title("Loss")
    plt.show()
