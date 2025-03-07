# Graph Low-Rank Adapters of High Regularity for Graph Neural Networks and Graph Transformers

This repository contains the official implementation of **Graph Low-Rank Adapters of High Regularity for Graph Neural Networks and Graph Transformers**, a novel parameter-efficient fine-tuning (PEFT) method specifically designed for Graph Neural Networks (GNNs) and Graph Transformers (GTs). Our novel adapter, GConv-Adapter, leverages a two-fold normalized graph convolution and trainable low-rank weight matrices to achieve state-of-the-art performance while maintaining a low parameter count.

## Introduction

As Graph Neural Networks continue to grow in size with the development of graph foundation models, efficient fine-tuning methods become increasingly important. GConv-Adapter addresses this challenge by providing a parameter-efficient approach that:

1. **Leverages graph structure**: Unlike some existing graph adapters, GConv-Adapter is sensitive to the input graph's structure and can capture second-hop interactions.
2. **Maintains stability**: Our adapter has a bounded Lipschitz constant for graphs with sufficient connectivity, ensuring stable learning.
3. **Achieves strong performance**: GConv-Adapter achieves state-of-the-art or competitive results on both inductive and transductive learning tasks.

## Setup

### Requirements

To set up the environment, you can use either the inductive or transductive environment YAML files:

For inductive learning tasks:
```bash
conda env create -f requirements/mpnn_inductive_environment.yml
conda activate mpnn_inductive
```

For transductive learning tasks:
```bash
conda env create -f requirements/gtn_transductive_environment.yaml
conda activate gtn_transductive
```

### Dataset Preparation

The repository is structured to work with standard graph datasets:

- **Inductive datasets** (molecular property prediction): ESOL, FreeSolv, Lipophilicity, Tox21, SIDER, ClinTox, BACE, MUV, and HIV
- **Transductive datasets** (node classification): Cora, Citeseer, and PubMed

Datasets will be automatically downloaded when running the experiments for the first time.

## Usage

### Fine-tuning with GConv-Adapter

To fine-tune a pre-trained GNN or GT model using GConv-Adapter:

#### Inductive Learning (Molecular Property Prediction)

```bash
python scripts/finetune_inductive_learning.py \
    ++pretrained_model=pretrained/chem_gin_supervised_contextpred.pth \
    ++model.gnn_type=gin \
    ++dataset.name=esol \
    ++seed=0 \
    +finetune=gconv_adapter
```

#### Transductive Learning (Node Classification)

```bash
python scripts/finetune_transductive_learning.py \
    ++dataset.name=cora \
    ++seed=0 \
    +finetune=gconv_adapter
```

### Running Ablation Studies

We provide several scripts to run ablation studies on different components of GConv-Adapter:

#### Insertion Form Ablation

```bash
./ablations/1_run_insertion_form_ablation.sh -d esol,lipo -s scripts/finetune_inductive_learning.py
```

#### Normalization Ablation

```bash
./ablations/2_run_normalization_ablation.sh -d cora,citeseer,pubmed -s scripts/finetune_transductive_learning.py -t sequential -p pre,post
```

#### Learnable Scalar Ablation

```bash
./ablations/3_run_learnable_scalar_ablation.sh -d esol,lipo,clintox,hiv -s scripts/finetune_inductive_learning.py -t sequential -p pre,post -n none
```

#### Skip Connection Ablation

```bash
./ablations/4_run_skip_connection_ablation.sh -d cora,citeseer,pubmed -s scripts/finetune_transductive_learning.py -t sequential -p pre,post -n none -l True
```

#### Normalized Adjacency Ablation

```bash
./ablations/5_run_normalized_adj_ablation.sh -d esol,lipo,tox21,hiv -s ./scripts/finetune_inductive.py -t sequential -p pre,post -n none -l True -c True
```

### Batch Experiments

For running multiple experiments in sequence, you can use the batch experiment scripts:

```bash
python experiments/run_inductive_learning.py --gnn_models gin --datasets esol,lipo --seeds 0,1
```

```bash
python experiments/run_transductive_learning.py --datasets cora,citeseer --seeds 0,1
```

### Analyzing Results

To analyze and visualize the results from multiple experiments:

```bash
python experiments/wandb_inductive_learning_experiments_scraper.py --project "Your-WandB-Project-Name" --entity "Your-WandB-Entity"
```

```bash
python experiments/wandb_transductive_learning_experiments_scraper.py --project "Your-WandB-Project-Name" --entity "Your-WandB-Entity"
```

To analyze ablation studies results:

```bash
python experiments/wandb_ablations_scraper.py --project_name "Your-WandB-Project-Name" --wandb_tag "insertion_form"
```

## Repository Structure

```
.
├── ablations/                 # Scripts for running ablation studies
├── configs/                   # Configuration files for models and experiments
├── experiments/               # Scripts for running and analyzing experiments
├── requirements/              # Environment YAML files
├── scripts/                   # Main training and evaluation scripts
├── src/                       # Source code
│   ├── dataset/               # Dataset loaders and utilities
│   ├── finetune/              # Fine-tuning methods implementation
│   ├── graphs/                # Graph utilities and analysis
│   ├── layers/                # Layer implementations for GNNs
│   ├── metrics/               # Evaluation metrics
│   ├── models/                # Model architectures
│   └── utils/                 # Utility functions
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Citing GConv-Adapter

If you use GConv-Adapter in your research, please cite our paper:

```bibtex
@inproceedings{
  papageorgiou2025graph,
  title={Graph Low-Rank Adapters of High Regularity for Graph Neural Networks and Graph Transformers},
  author={Pantelis Papageorgiou and Haitz S{\'a}ez de Oc{\'a}riz Borde and Anastasis Kratsios and Michael M. Bronstein},
  booktitle={First Workshop on Scalable Optimization for Efficient and Adaptive Foundation Models},
  year={2025},
  url={https://openreview.net/forum?id=gxhZj6uvFC}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
