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


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000, help='Number of optimization steps')
    parser.add_argument("--seed", type=int, default=1234, help='Seed for torch random')
    parser.add_argument("--lr", type=float, default=1e-3, help='Learning rate for cob optimizer')
    parser.add_argument("--cob_range", type=float, default=1,
                        help='Range for the teleportation to create target weights')

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
        nn.Linear(128, 10),
        nn.ReLU()
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
    input_data = next(iter(test_loader))[0]
    original_pred = model(input_data)
    
    for layer, stats in activation_stats.items():
        print(f'Layer {layer}: {stats}')
    
    original_pred = original_pred.detach().cpu().numpy()

    model = swap_model_modules_for_COB_modules(model)
    model = NeuralTeleportationModel(model, input_shape=(1, 1, 28, 28))
    model.training = True
    
    # load model weights
    model.network.load_state_dict(torch.load('model_weights_cob_activation_norm.pth'))

    # Get the initial set of weights and teleport.
    initial_weights = model.get_weights().detach()

    # Generate a new random cob
    cob = model.generate_random_cob(cob_range=args.cob_range, requires_grad=True)
    # new_cob = cob.clone().detach().requires_grad_(True)

    history = []
    cob_error_history = []

    args.steps = 100

    for e in range(args.steps):
        print("\n === Step: ", e , " === ")
        model.set_weights(initial_weights)
        # cob = new_cob
        
        # Teleport with this cob
        model = model.teleport(cob,reset_teleportation=True)

        activation_stats = {}
        for i, layer in enumerate(model.network):
            if isinstance(layer, nn.ReLU):
                model.network[i].register_forward_hook(activation_hook(f'relu_{i}'))
        pred = model(input_data)

        # define loss as the value of the l1 norm of the activations
        loss = sum([stats['norm'] for stats in activation_stats.values()])
        print("Loss: ", loss.item(), "\t l1 norm of activations: ", [stats['norm'].item() for stats in activation_stats.values()])

        # Backwards pass
        grad = torch.autograd.grad(loss, cob, create_graph=True)
        
        # compute the distance between original pred and pred (numpy has no abs method)
        print("Pred diff mean", np.absolute(original_pred - pred.detach().cpu().numpy()).mean())

        # grad[0] = grad[0] + 0.1 * grad[0].mean() * torch.randn_like(grad[0])
        # new_cob = cob.clone().detach() - args.lr * grad[0] / grad[0].norm()
        cob = cob - args.lr * grad[0] / grad[0].norm()
        # set all new_cob element to be positive
        # new_cob = torch.abs(new_cob)
        cob = torch.abs(cob)
        cob.grad = None
        
        history.append(loss.item())
        
        # cob_error_history.append((cob - target_cob).square().mean().item())
        # if e % 100 == 0:
            # print("Step: {}, loss: {}, cob error: {}".format(e, loss.item(), (cob - target_cob).abs().mean().item()))

    plt.figure()
    # plt.plot(history)
    # concat history[0] and history[10:]
    # plt.plot([history[0]] + history[10:])
    plt.plot(history)
    plt.title("Loss")
    plt.show()
