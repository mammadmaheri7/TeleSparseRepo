# TeleSparseRepo

## Overview
**TeleSparseRepo** is a research codebase for exploring *verifiable unlearning* and *privacy-preserving inference* via sparse neural networks. It integrates pruning, teleportation, and zkML benchmarking, with tools and interfaces for reproducible experiments targeting proof generation and activation optimization.

This project supports the paper accepted at **PETS 2025**.

---

## Workflow Summary
1. **Prune** model weights using structured methods.
2. **Teleport** activations to optimize proof-efficiency.
3. **Benchmark** proof systems (e.g., zkML, ezkl).

---

## Project Structure
```bash
TeleSparseRepo/
├── Benchmark/                     # Benchmarking and evaluation scripts
│   ├── benchmarks/               # Charts, plots, tables
│   ├── models/                   # Model structures and formats
│   └── Results/                  # Output metrics and zk proof files
├── frameworks/                   # zkML engines: ezkl, zkml, opML, circomlib-ml
├── NeuralTeleportation/          # Model teleportation logic
│   ├── neuralteleportation/     # Activation teleportation core logic
│   ├── requirements/            # Component dependencies
│   └── tests/                   # Unit tests for teleportation logic
├── env/                          # Virtualenv configurations
├── requirements.txt              # Project dependencies
├── run_table1_ezkl_resnet20.sh   # Benchmark script for ezkl
├── run_table1_zkml_resnet20.sh   # Benchmark script for zkML
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/TeleSparseRepo.git
cd TeleSparseRepo
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
cd NeuralTeleportation
pip install -e .
```

---

## Step-by-Step Usage

### 🔧 1. Prune Models
Use RD-Prune for ResNet/MobileNet:
```bash
python iterate.py --dataset cifar100 --model resnet20_cifar100 \
                  --pruner rd --worst_case_curve \
                  --calib_size 1024 --iter_end 1
```
Output will be a pruned ONNX model.

---

### 🌀 2. Optimize Activations via Teleportation
```bash
python resnet20_teleport_ZO.py --prefix_dir path/to/pruned_model_dir
```
Generates:
- `resnet20_cob_activation_norm_teleported.onnx`
- `network_complete.onnx`
- `input_convs.json`

---

### 📊 3. Benchmark zk Proof Systems
Run the two benchmarking scripts:
```bash
./run_table1_ezkl_resnet20.sh
./run_table1_zkml_resnet20.sh
```
Results are saved under `Benchmark/Results/`

---

## Requirements
- Python >= 3.8
- PyTorch >= 1.8
- numpy, scipy, matplotlib
- [ezkl](https://github.com/zkonduit/ezkl)
- [zkml](https://github.com/zkml/zkml)

---

## Notes
- Teleportation logic is based on the [neuralteleportation](https://github.com/vitalab/neuralteleportation) repository.
- Benchmarking pipeline and scripts are adapted from [ZKML-Benchmark](https://github.com/Guy1m0/ZKML-Benchmark/tree/main).

---

## Citation
If you use this code in your work, please cite:
```bibtex
@article{maheri2025telesparse,
  title={TeleSparse: Practical Privacy-Preserving Verification of Deep Neural Networks},
  author={Maheri, Mohammad M and Haddadi, Hamed and Davidson, Alex},
  journal={Proceedings on Privacy Enhancing Technologies (PETS)},
  year={2025}
}
```

---

## Contact
Please open a GitHub issue or contact `m.maheri23 [at] imperial.ac.uk` for questions or contributions.
