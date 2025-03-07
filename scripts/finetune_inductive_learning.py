import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

from src.utils.models import get_model_size, summarize_model
from src.utils.logging import setup_logging
from src.utils.utils import serialize_finetuning_args


def train(model, device, data_loader, optimizer, criterion, logger):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        device (torch.device): The device to use for training (CPU or GPU).
        data_loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn.Module): Loss function to compute the training loss.
        logger (logging.Logger): Logger for logging eval information.

    Returns:
        float: The average loss of the epoch
    """
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Training", leave=True, position=0):
        try:
            batch = batch.to(device)
            optimizer.zero_grad()
            # Forward pass
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # Prepare target
            target = batch.y.view(output.shape).to(torch.float64).to(device)
        except RuntimeError as e:
            logger.info(e)
            continue

        # Filter out NaN values
        valid_mask = ~torch.isnan(target)
        valid_output = output[valid_mask]
        valid_target = target[valid_mask]
        
        # Compute the loss only for valid targets
        if len(valid_target) > 0:  # Ensure there's something to compute
            loss_matrix = criterion(valid_output.double(), valid_target)
            loss = torch.sum(loss_matrix) / len(valid_target)
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
    # Compute average loss
    avg_loss = total_loss / len(data_loader)
    return avg_loss



def test(model, device, data_loader, logger, metric='rmse'):
    """
    Evaluates the model and computes the average specified metric.

    Args:
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): The device to use for evaluation (CPU or GPU).
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        logger (logging.Logger): Logger for logging eval information.
        metric (str): The metric to compute ('roc_auc' or 'rmse').

    Returns:
        float: The average specified metric across valid targets.
    """
    # Validate metric argument
    assert metric.lower() in ['roc-auc', 'rmse'], "Metric must be '(ROC-AUC) roc-auc' or '(RMSE) rmse'."

    # Set the model to evaluation mode
    model.eval()
    true_labels = []
    predicted_scores = []
    # Iterate over batches in the data loader
    for batch in tqdm(data_loader, desc="Evaluating"):
        # Move the batch to the specified device
        batch = batch.to(device)
        try:
            # Perform forward pass without computing gradients
            with torch.no_grad():
                predictions = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # Prepare the target
            target = batch.y.view(predictions.shape).to(torch.float64).to(device)
        except RuntimeError as e:
            logger.info(e)
            continue
        # Filter out NaN values
        valid_mask = ~torch.isnan(target)
        valid_predictions = predictions[valid_mask]
        valid_target = target[valid_mask]
        
        # Append valid true labels and predictions to the lists
        true_labels.append(valid_target)
        predicted_scores.append(valid_predictions)

    # Concatenate all valid batches into single arrays
    true_labels = torch.cat(true_labels, dim=0).cpu().numpy()
    predicted_scores = torch.cat(predicted_scores, dim=0).cpu().numpy()

    # Ensure true_labels and predicted_scores are 2D
    if true_labels.ndim == 1:
        true_labels = true_labels.reshape(-1, 1)
    if predicted_scores.ndim == 1:
        predicted_scores = predicted_scores.reshape(-1, 1)

    # Calculate the specified metric for each target
    scores = []
    for i in range(true_labels.shape[1]):
        # Calculate ROC AUC
        if metric.lower() == 'roc-auc':
            # AUC is only defined when there is at least one positive and one negative data point.
            if np.sum(true_labels[:, i] == 1) > 0 and np.sum(true_labels[:, i] == 0) > 0:
                # Compute ROC AUC score for the valid indices
                scores.append(roc_auc_score(true_labels[:, i], predicted_scores[:, i]))
        
        # Calculate RMSE
        elif metric.lower() == 'rmse':
            if np.any(~np.isnan(true_labels[:, i])):
                scores.append(np.sqrt(mean_squared_error(true_labels[:, i], predicted_scores[:, i])))

    # Check if any target labels are missing and logger.info the missing ratio
    if len(scores) < true_labels.shape[1]:
        logger.info("Some target labels are missing!")
        logger.info("Missing ratio: %f" % (1 - float(len(scores)) / true_labels.shape[1]))

    # Return the average specified metric across valid targets, or 0.0 if no valid scores
    return sum(scores) / len(scores) if scores else 0.0


@hydra.main(config_path="../configs/", config_name="finetune_inductive", version_base="1.2")
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

    # Create dir to export finetuned model
    model_save_dir = 'finetuned'
    os.makedirs(model_save_dir, exist_ok=True)

    # Set random seeds for reproducibility
    logger.info(f"Setting random seed to {cfg.seed} for reproducibility.")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    logger.info(f"Using device: {device}\n")

    # Determine the number of tasks based on the dataset
    dataset_info = {
        "esol": {
            "num_tasks": 1,
            "task_type": "Regression",
            "metric": "RMSE",
            "split": "Random"
        },
        "freesolv": {
            "num_tasks": 1,
            "task_type": "Regression",
            "metric": "RMSE",
            "split": "Random"
        },
        "lipo": {
            "num_tasks": 1,
            "task_type": "Regression",
            "metric": "RMSE",
            "split": "Random"
        },
        "tox21": {
            "num_tasks": 12,
            "task_type": "Classification",
            "metric": "ROC-AUC",
            "split": "Random"
        },
        "sider": {
            "num_tasks": 27,
            "task_type": "Classification",
            "metric": "ROC-AUC",
            "split": "Random"
        },
        "clintox": {
            "num_tasks": 2,
            "task_type": "Classification",
            "metric": "ROC-AUC",
            "split": "Random"
        },
        "bace": {
            "num_tasks": 1,
            "task_type": "Classification",
            "metric": "ROC-AUC",
            "split": "Scaffold"
        },
        "hiv": {
            "num_tasks": 1,
            "task_type": "Classification",
            "metric": "ROC-AUC",
            "split": "Scaffold"
        },
        "muv": {
            "num_tasks": 17,
            "task_type": "Classification",
            "metric": "ROC-AUC",
            "split": "Random"
        }
    }

    if cfg.dataset.name not in dataset_info:
        raise ValueError("Invalid dataset name.")
    num_tasks = dataset_info[cfg.dataset.name]["num_tasks"]
    logger.info(f"Dataset: {cfg.dataset.name}, Number of tasks: {num_tasks}\n")

    # Paths to the dataset files
    train_dataset_path = os.path.join(cfg.dataset.base_dir, cfg.dataset.name, 'train_dataset.pt')
    valid_dataset_path = os.path.join(cfg.dataset.base_dir, cfg.dataset.name, 'valid_dataset.pt')
    test_dataset_path = os.path.join(cfg.dataset.base_dir, cfg.dataset.name, 'test_dataset.pt')

    # Check if dataset files exist
    if os.path.exists(train_dataset_path) and os.path.exists(valid_dataset_path) and os.path.exists(test_dataset_path):
        # Load the datasets
        logger.info("Loading datasets from saved files...")
        train_dataset = torch.load(train_dataset_path)
        valid_dataset = torch.load(valid_dataset_path)
        test_dataset = torch.load(test_dataset_path)
        logger.info("Loading datasets from saved files completed.\n")
    else:
        # Set up the dataset
        logger.info("Setting up the dataset...")
        dataset = MoleculeNet(cfg.dataset.base_dir, name=cfg.dataset.name)
        dataset.data.x = dataset.data['x'][:, [0, 1]]
        dataset.data.edge_attr = dataset.data['edge_attr'][:, [1, 2]]
        logger.info(f"Loaded dataset with {len(dataset)} molecules\n")
        
        # Split dataset according to specified split (scaffold by default)
        dataset_name = cfg.dataset.name.lower()
        split_method = dataset_info[dataset_name]["split"].lower()
        split_methods = {
            "scaffold": "src.dataset.inductive.splitters.scaffold_split",
            "random": "src.dataset.inductive.splitters.random_split" 
        }
        cfg.split._target_ = split_methods[split_method]
        logger.info(f"Splitting dataset using {cfg.split._target_.split('.')[-1]} method...")
        smiles_list = dataset.data.smiles if split_method == "scaffold" else None
        train_dataset, valid_dataset, test_dataset = instantiate(cfg.split, dataset=dataset, smiles_list=smiles_list)
        logger.info(f"Dataset split into {len(train_dataset)} training, {len(valid_dataset)} validation, and {len(test_dataset)} test samples\n")

        # Save the datasets
        logger.info(f"Training dataset saved to {train_dataset_path}")
        torch.save(train_dataset, train_dataset_path)
        logger.info(f"Validation dataset saved to {valid_dataset_path}")
        torch.save(valid_dataset, valid_dataset_path)
        logger.info(f"Test dataset saved to {test_dataset_path}\n")
        torch.save(test_dataset, test_dataset_path)

    # Create the dataloaders for train, val, and test sets
    logger.info("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=cfg.dataloader.batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=cfg.dataloader.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataloader.batch_size, shuffle=False)
    logger.info("Data loaders created\n")

    # Set up model
    logger.info("Setting up the model...")
    model = instantiate(cfg.model, num_tasks=num_tasks, device=device)
    base_model_summary_df = summarize_model(model)
    logger.info(f"Model architecture:\n{model}\n")
    
    # Load pretrained model   
    try:
        logger.info(f"Loading pretrained model from {cfg.pretrained_model}...")
        model.load_state_dict(torch.load(cfg.pretrained_model, map_location=device, weights_only=True), strict=True)
        logger.info("Pretrained model loaded successfully.\n")
    except RuntimeError as e:
        error_msg = str(e)
        if "Missing key(s) in state_dict: \"graph_pred_linear.weight\", \"graph_pred_linear.bias\"." in error_msg:
            model.load_state_dict(torch.load(cfg.pretrained_model, map_location=device, weights_only=True), strict=False)
            logger.info("Pretrained model loaded successfully.\n")
        else:
            logger.error(f"Error loading the pretrained model: {error_msg}")
            raise e

    # Specify fine-tuning method
    finetune_method = None
    finetune_args = None
    model_save_path = None 

    if not cfg.finetune:
        finetune_method = 'Full Fine-Tuning'
        model_save_path = os.path.join(model_save_dir, f"{cfg.dataset.name}_{cfg.pretrained_model.split('/')[-1][:-4]}_FT_{cfg.seed}.pth")
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
        model_save_path = os.path.join(model_save_dir, f"{cfg.dataset.name}_{cfg.pretrained_model.split('/')[-1][:-4]}_{serialized_args}_{cfg.seed}.pth")
    logger.info(f"Fine-Tuninng method: {finetune_method}")
    logger.info(f"Fine-Tuning arguments: {finetune_args}\n")

    # Move model to device 
    model.to(device)
    
    # Summary information of the model
    summary_df = summarize_model(model)
    # Add a column 'Added Parameters' to indicate whether the parameter is added.
    # It will be 'Yes' if the module is not present in the base model and is trainable in the current model, otherwise 'No'.
    summary_df['Added Parameters'] = summary_df.apply(
        lambda row: 'Yes' if row['Module'] not in base_model_summary_df['Module'].values and row['Trainable'] == 'Yes' else 'No', 
        axis=1
    )
    # Extract the total number of parameters from the summary DataFrame
    total_params = summary_df.loc[summary_df['Module'] == 'Total', 'Num Parameters'].values[0]
    # Extract the total number of trainable parameters from the summary DataFrame
    trainable_params = summary_df.loc[summary_df['Module'] == 'Total', 'Trainable'].values[0]
    # Calculate the percentage of trainable parameters
    percentage_of_trainable_params = trainable_params / total_params
    # Sum the number of parameters marked as 'Yes' in the 'Added Parameters' column
    added_params = summary_df[summary_df['Added Parameters'] == 'Yes']['Num Parameters'].sum()
    # Calculate the model size in MB
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

    # Set up optimizer with different learning rates for different parts of the model
    logger.info("Setting up optimizer...")
    model_param_group = [{"params": model.gnn.parameters()}]
    if cfg.model.pooling_method == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr": cfg.training.lr * cfg.training.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": cfg.training.lr * cfg.training.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    logger.info(f"Optimizer:\n{optimizer}\n")

    # Set up criterion
    if dataset_info[cfg.dataset.name]["task_type"] == "Classification":
        criterion = nn.BCEWithLogitsLoss(reduction = "none")
    elif dataset_info[cfg.dataset.name]["task_type"] == "Regression":
        criterion = nn.MSELoss(reduction='none')

    # Train/Evaluation 
    # Initialize best validation score and early stopping counter
    metric = dataset_info[cfg.dataset.name]["metric"]
    best_val_score = float('-inf') if metric.lower() == 'roc-auc' else float('inf')
    epochs_no_improve = 0

    for epoch in range(1, cfg.training.epochs + 1):
        logger.info(f"Epoch {str(epoch).zfill(3)}: Training model...")
        epoch_loss = train(model, device, train_loader, optimizer, criterion, logger)
        logger.info(f"Epoch {str(epoch).zfill(3)}: Training model completed.")
        logger.info(f'Epoch {str(epoch).zfill(3)}: Training Loss: {epoch_loss:.6f}')

        logger.info(f"Epoch {str(epoch).zfill(3)}: Evaluating model on train set...")
        train_score = test(model, device, train_loader, logger, dataset_info[cfg.dataset.name]["metric"])
        logger.info(f"Epoch {str(epoch).zfill(3)}: Evaluating model on train set completed.")

        logger.info(f"Epoch {str(epoch).zfill(3)}: Evaluating model on validation set...")
        val_score = test(model, device, val_loader, logger, dataset_info[cfg.dataset.name]["metric"])
        logger.info(f"Epoch {str(epoch).zfill(3)}: Evaluating model on validation set completed.")

        logger.info(f"Epoch {str(epoch).zfill(3)}: Evaluating model on test set...")
        test_score = test(model, device, test_loader, logger, dataset_info[cfg.dataset.name]["metric"])
        logger.info(f"Epoch {str(epoch).zfill(3)}: Evaluating model on test set completed.")

        logger.info(f"Epoch {str(epoch).zfill(3)}: Train {metric}: {train_score:.6f} | Validation {metric}: {val_score:.6f} | Test {metric}: {test_score:.6f}\n")

        # Log epoch-specific metrics to wandb
        if cfg.wandb.project:
            wandb.log({
                "Train_Loss": epoch_loss,
                f"Train_{metric}": train_score,
                f"Validation_{metric}": val_score,
                f"Test_{metric}": test_score
            })

        # Early stopping
        if metric.lower() == 'roc-auc':
            # For ROC-AUC, higher is better
            is_improvement = val_score > best_val_score
        else:
            # For RMSE, lower is better
            is_improvement = val_score < best_val_score

        if is_improvement:
            best_val_score = val_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Model saved to {model_save_path}\n")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == cfg.training.early_stopping_patience:
            logger.info("Early stopping")
            break
    
    # If fine-tuning method is adapter and the adapter has a learnable scalar
    model.print_all_learnable_scalars()

    if cfg.wandb.project:
        artifact = wandb.Artifact("finetuned_model", type="model")
        artifact.add_file(model_save_path)
        wandb.log_artifact(artifact)
        logger.info("Model logged as a Weights & Biases artifact.")

    if cfg.wandb.project:
        wandb.finish()

if __name__ == "__main__":
    main()
