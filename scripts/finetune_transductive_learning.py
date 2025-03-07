import os
import sys
import wandb

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from sklearn.neighbors import kneighbors_graph

from src.dataset.transductive.dataset import load_dataset
from src.dataset.transductive.data_utils import load_fixed_splits, adj_mul, get_gpu_memory_map
from src.utils.metrics import evaluate, eval_acc, eval_rocauc, eval_f1
from src.utils.parse import parse_method

from src.utils.models import get_model_size, summarize_model
from src.utils.utils import serialize_finetuning_args
from src.utils.logging import setup_logging

import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Main function with Hydra config
@hydra.main(config_path="../configs/", config_name="finetune_transductive", version_base="1.2")
def main(cfg: DictConfig):
    # Initialize wandb
    if cfg.wandb.project:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name if cfg.wandb.name else serialize_finetuning_args(cfg.finetune) if cfg.finetune else "FT",
            config=dict(OmegaConf.to_container(cfg, resolve=True)),
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes
        )
    # Set up logging
    output_dir = HydraConfig.get().runtime.output_dir
    logger = setup_logging(output_dir)
    logger.info(f"Hydra run directory: {output_dir}\n")

    # Fix seed for reproducibility
    fix_seed(cfg.seed)

    # Setup device
    device = torch.device("cpu") if cfg.device.cpu else torch.device(f"cuda:{cfg.device.device_id}") if torch.cuda.is_available() else torch.device("cpu")

    ### Load and preprocess data ###
    logger.info("Loading dataset...")
    dataset = load_dataset(cfg.dataset.data_dir, cfg.dataset.name, cfg.dataset.sub_dataset)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    # Get the splits for all runs
    if cfg.training.rand_split:
        split_idx_lst = dataset.get_idx_split(train_prop=cfg.training.train_prop, valid_prop=cfg.training.valid_prop)
    elif cfg.training.rand_split_class:
        split_idx_lst = dataset.get_idx_split(split_type='class', label_num_per_class=cfg.training.label_num_per_class)
    elif cfg.dataset.name in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products', 'amazon2m']:
        split_idx_lst = dataset.load_fixed_splits()
    else:
        split_idx_lst = load_fixed_splits(cfg.dataset.data_dir, dataset, name=cfg.dataset.name, protocol=cfg.training.protocol)

    # Move dataset to the device
    dataset.label = dataset.label.to(device)
    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

    ### Basic information of datasets ###
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    logger.info(f"Dataset {cfg.dataset.name}: num nodes {n}, num edges {e}, num node feats {d}, num classes {c}")

    # Symmetrize edge index if needed
    if not cfg.gnn_baseline.directed and cfg.dataset.name != 'ogbn-proteins':
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

    ### Loss function ###
    criterion = nn.BCEWithLogitsLoss() if cfg.dataset.name in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins') else nn.NLLLoss()

    ### Performance metric ###
    eval_func = eval_rocauc if cfg.metric.name == 'rocauc' else eval_f1 if cfg.metric.name == 'f1' else eval_acc

    ### Relational bias adjacency matrices ###
    adjs = []
    adj, _ = remove_self_loops(dataset.graph['edge_index'])
    adj, _ = add_self_loops(adj, num_nodes=n)
    adjs.append(adj)
    if cfg.model.method == 'nodeformer':
        for i in range(cfg.model.rb_order - 1):
            adj = adj_mul(adj, adj, n)
            adjs.append(adj)
    dataset.graph['adjs'] = adjs

    ### Training loop ###
    split_idx = split_idx_lst
    train_idx = split_idx['train'].to(device)

    ### Load model ###
    logger.info(f"Loading model {cfg.model.method}...")
    model = parse_method(cfg, dataset, n, c, d, device)

    model.reset_parameters()

    base_model_summary_df = summarize_model(model)

    # Load pretrained model if provided
    if cfg.training.pretrained_model and os.path.exists(cfg.training.pretrained_model):
        logger.info(f"Loading pretrained model from {cfg.training.pretrained_model}")
        checkpoint = torch.load(cfg.training.pretrained_model, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info("Pretrained model loaded successfully.")
    else:
        logger.info("No pretrained model provided or file does not exist.")

    # Specify fine-tuning method
    finetune_method = None
    finetune_args = None
    model_save_path = None 

    if not cfg.finetune:
        finetune_method = 'Full Fine-Tuning'
        model_save_path = os.path.join(cfg.training.model_dir, f"{cfg.dataset.name}-{cfg.model.method}_{cfg.training.pretrained_model.split('/')[-1][:-4]}_FT_{cfg.seed}.pth")
    else:
        if 'adapter' in cfg.finetune:
            finetune_method = cfg.finetune.adapter._target_.split(".")[-1]
            model.add_adapter(adapter_class=cfg.finetune.adapter, positions=cfg.finetune.positions, type=cfg.finetune.type)
        else:
            finetune_method = cfg.finetune._target_.split(".")[-1]
            finetuning = instantiate(cfg.finetune, model=model)
            model = finetuning.apply()
        finetune_args = {k: v for k, v in cfg.finetune.items()}
        serialized_args = serialize_finetuning_args(finetune_args)
        model_save_path = os.path.join(cfg.training.model_dir, f"{cfg.dataset.name}-{cfg.model.method}_{cfg.training.pretrained_model.split('/')[-1][:-4]}_{serialized_args}_{cfg.seed}.pth")

        # Fine-tune input/output fc
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
    
    logger.info(f"Fine-Tuninng method: {finetune_method}")
    logger.info(f"Fine-Tuning arguments: {finetune_args}\n")

    # Move model to device 
    model.to(device)

    # Summary information of the model
    summary_df = summarize_model(model)
    summary_df['Added Parameters'] = summary_df.apply(
        lambda row: 'Yes' if row['Module'] not in base_model_summary_df['Module'].values and row['Trainable'] == 'Yes' else 'No', 
        axis=1
    )
    total_params = summary_df.loc[summary_df['Module'] == 'Total', 'Num Parameters'].values[0]
    fc_params = sum(param.numel() for name, param in model.named_parameters() if 'fc' in name and param.requires_grad)
    if finetune_method == "Full Fine-Tuning":
        trainable_params = summary_df.loc[summary_df['Module'] == 'Total', 'Trainable'].values[0]
    else:
        trainable_params = summary_df.loc[summary_df['Module'] == 'Total', 'Trainable'].values[0] - fc_params
    percentage_of_trainable_params = trainable_params / total_params
    added_params = summary_df[summary_df['Added Parameters'] == 'Yes']['Num Parameters'].sum()
    model_size = get_model_size(model)
    logger.info("\n" + summary_df.to_string() + "\n")
    logger.info("Model size: {:.3f} MB".format(model_size))
    logger.info("Percentage of trainable parameters in the model: {:.8f}%".format(100*percentage_of_trainable_params))
    logger.info("Number of parameters added: {}\n".format(added_params))

    # Log model parameters once to wandb
    if cfg.wandb.project:
        wandb.log({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "percentage_trainable_parameters": percentage_of_trainable_params,
            "model_size_MB": model_size
        })

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=cfg.model.weight_decay, lr=cfg.model.lr)
    best_val = float('-inf')

    for epoch in range(cfg.training.epochs):
        model.train()
        optimizer.zero_grad()

        if cfg.model.method == 'nodeformer':
            out, link_loss_ = model(dataset.graph['node_feat'], dataset.graph['adjs'], cfg.model.tau)
        elif cfg.model.method == 'difformer':
            out = model(dataset.graph['node_feat'], dataset.graph['adjs'])
        else:
            out = model(dataset)

        if cfg.dataset.name in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
            true_label = dataset.label if dataset.label.shape[1] > 1 else F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])

        if cfg.model.method == 'nodeformer':
            loss -= cfg.model.lamda * sum(link_loss_) / len(link_loss_)

        loss.backward()
        optimizer.step()

        if epoch % cfg.training.eval_step == 0:
            result = evaluate(model, dataset, split_idx, eval_func, criterion, cfg)

            if result[1] > best_val:
                best_val = result[1]
                if cfg.training.save_model:
                    torch.save(model.state_dict(), model_save_path)

            logger.info(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test: {100 * result[2]:.2f}%")
            if cfg.wandb.project:
                wandb.log({"Train_Accuracy": result[0], "Validation_Accuracy": result[1], "Test_Accuracy": result[2], "Train_Loss": loss})

    if cfg.wandb.project:
        artifact = wandb.Artifact("finetuned_model", type="model")
        artifact.add_file(model_save_path)
        wandb.log_artifact(artifact)
        logger.info("Model logged as a Weights & Biases artifact.")

    if cfg.wandb.project:
        wandb.finish()

    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()
