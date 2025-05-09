# TeleSparseRepo

## Overview
TeleSparseRepo is a research project focused on Telesparse architectures for efficient deep learning models. This repository contains the implementation and evaluation tools for sparse neural networks.

## Overal Steps
1. Sparsify the model
2. Teleport the mode to optimize the range of activate function
3. Generate the proof for the outputed model and benchmark

## Project Structure
```
TeleSparseRepo/
├── data/                  # Dataset handling and processing
├── models/                # Neural network model implementations
├── training/             # Training scripts and configurations
├── evaluation/           # Evaluation metrics and testing
├── utils/                # Utility functions and helper tools
└── experiments/          # Experimental configurations and results
```

## Key Features
- Implementation of Telesparse architecture
- Sparse neural network training framework
- Performance evaluation tools
- Dataset preprocessing utilities
- Model visualization components

## Requirements
```bash
python >= 3.8
pytorch >= 1.8.0
numpy
scipy
matplotlib
tensorboard
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/TeleSparseRepo.git
cd TeleSparseRepo
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Prune the models
This steps is done by [RD Prune](https://github.com/Akimoto-Cris/RD_PRUNE) for ResNet and MobileNetv1 Model and [CAP](https://github.com/IST-DASLab/CAP) for vision transformer model. The resulted pruned model have saved in the Results directory for ease of use.

Sample command to run on RD_Prune to find the pruned ResNet20 model
```bash
python iterate.py --dataset cifar100 --model resnet20_cifar100 --pruner rd --worst_case_curve --calib_size 1024 --iter_end 1
```


### Optimize the Activatation by Teleportation

python resnet20_teleport_ZO.py --prefix_dir <path to the directory containing the onnx file (output of the previous step (pruning))>

After running the command three corropsonding file will be generated in the 'prefix_dir' directory:

I.	resnet20_cob_activation_norm_teleported.onnx
II.	network_complete.onnx (Teleportable version of the model)
III. input_convs.json


### Benchmarking
To reproduce the Table1 results run the following bash files
```bash
./run_table1_ezkl_resnet20.sh
./run_table1_zklml_resnet20.sh
```



## Project Components

### Data Module
- Handles dataset loading and preprocessing
- Supports various data formats and transformations
- Implements data augmentation strategies

### Models
- Contains base model architectures
- Implements sparse layer configurations
- Supports model compression techniques

### Training
- Provides training loops and optimization
- Implements learning rate scheduling
- Supports distributed training

### Evaluation
- Contains metrics calculation
- Provides visualization tools
- Generates performance reports



## Citation
If you use this code in your research, please cite:
```
@article{maheri2025telesparse,
  title={TeleSparse: Practical Privacy-Preserving Verification of Deep Neural Networks},
  author={Maheri, Mohammad M and Haddadi, Hamed and Davidson, Alex},
  journal={arXiv preprint arXiv:2504.19274},
  year={2025}
}
```

## Contact
For questions or issues, please open an issue in the repository or contact [maintainer email].
