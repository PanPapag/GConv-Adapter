import torch
import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from a SMILES string.

    Args:
        smiles (str): SMILES string of a molecule.
        include_chirality (bool, optional): Whether to include chirality in the scaffold. Default is False.

    Returns:
        str: SMILES string of the scaffold.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold

def scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=False):
    """
    Split dataset by Bemis-Murcko scaffolds. This function can also ignore examples
    containing null values for a selected task when splitting. The split is deterministic.

    Args:
        dataset (torch_geometric.data.Dataset): PyTorch geometric dataset object.
        smiles_list (list): List of SMILES strings corresponding to the dataset.
        task_idx (int, optional): Column index of the data.y tensor. Filters out examples with null values
                                  in the specified task column before splitting. If None, no filtering is applied.
        null_value (float, optional): Value representing null in data.y if task_idx is provided. Default is 0.
        frac_train (float, optional): Fraction of data for training. Default is 0.8.
        frac_valid (float, optional): Fraction of data for validation. Default is 0.1.
        frac_test (float, optional): Fraction of data for testing. Default is 0.1.
        return_smiles (bool, optional): Whether to return the SMILES strings for each split. Default is False.

    Returns:
        tuple: Train, validation, and test slices of the input dataset. If return_smiles is True,
               also returns lists of SMILES strings for each split.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset), dtype=bool)
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    all_scaffolds = defaultdict(list)
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        all_scaffolds[scaffold].append(i)

    all_scaffold_sets = [scaffold_set for _, scaffold_set in sorted(all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]

    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []

    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(valid_idx)) == 0
    assert len(set(test_idx).intersection(valid_idx)) == 0

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if return_smiles:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles)
    else:
        return train_dataset, valid_dataset, test_dataset

def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                          frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    """
    Split dataset by Bemis-Murcko scaffolds randomly. This function can also ignore examples
    containing null values for a selected task when splitting.

    Args:
        dataset (torch_geometric.data.Dataset): PyTorch geometric dataset object.
        smiles_list (list): List of SMILES strings corresponding to the dataset.
        task_idx (int, optional): Column index of the data.y tensor. Filters out examples with null values
                                  in the specified task column before splitting. If None, no filtering is applied.
        null_value (float, optional): Value representing null in data.y if task_idx is provided. Default is 0.
        frac_train (float, optional): Fraction of data for training. Default is 0.8.
        frac_valid (float, optional): Fraction of data for validation. Default is 0.1.
        frac_test (float, optional): Fraction of data for testing. Default is 0.1.
        seed (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
        tuple: Train, validation, and test slices of the input dataset.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset), dtype=bool)
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)
    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(list(scaffolds.values()))
    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset

def random_split(dataset, task_idx=None, null_value=0,
                 frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0,
                 smiles_list=None):
    """
    Randomly split dataset into training, validation, and test sets. This function can also ignore examples
    containing null values for a selected task when splitting.

    Args:
        dataset (torch_geometric.data.Dataset): PyTorch geometric dataset object.
        task_idx (int, optional): Column index of the data.y tensor. Filters out examples with null values
                                  in the specified task column before splitting. If None, no filtering is applied.
        null_value (float, optional): Value representing null in data.y if task_idx is provided. Default is 0.
        frac_train (float, optional): Fraction of data for training. Default is 0.8.
        frac_valid (float, optional): Fraction of data for validation. Default is 0.1.
        frac_test (float, optional): Fraction of data for testing. Default is 0.1.
        seed (int, optional): Random seed for reproducibility. Default is 0.
        smiles_list (list, optional): List of SMILES strings corresponding to the dataset. If provided, the function
                                      also returns lists of SMILES strings for each split. Default is None.

    Returns:
        tuple: Train, validation, and test slices of the input dataset. If smiles_list is provided,
               also returns lists of SMILES strings for each split.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        non_null = y_task != null_value
        idx_array = np.where(non_null)[0]
        dataset = dataset[torch.tensor(idx_array)]
    else:
        pass

    num_mols = len(dataset)
    random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(frac_train * num_mols)]
    valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols) + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

    assert len(set(train_idx).intersection(valid_idx)) == 0
    assert len(set(valid_idx).intersection(test_idx)) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if smiles_list is None:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i] for i in train_idx]
        valid_smiles = [smiles_list[i] for i in valid_idx]
        test_smiles = [smiles_list[i] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles)

def cv_random_split(dataset, fold_idx=0, frac_train=0.9, frac_valid=0.1, seed=0, smiles_list=None):
    """
    Perform stratified k-fold cross-validation split on the dataset.

    Args:
        dataset (torch_geometric.data.Dataset): PyTorch geometric dataset object.
        fold_idx (int, optional): Index of the fold to use for validation. Default is 0.
        frac_train (float, optional): Fraction of data for training. Default is 0.9.
        frac_valid (float, optional): Fraction of data for validation. Default is 0.1.
        seed (int, optional): Random seed for reproducibility. Default is 0.
        smiles_list (list, optional): List of SMILES strings corresponding to the dataset. If provided, the function
                                      also returns lists of SMILES strings for each split. Default is None.

    Returns:
        tuple: Train and validation slices of the input dataset. If smiles_list is provided,
               also returns lists of SMILES strings for each split.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid, 1.0)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    labels = [data.y.item() for data in dataset]

    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, val_idx = idx_list[fold_idx]

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(val_idx)]

    if smiles_list is None:
        return train_dataset, valid_dataset
    else:
        train_smiles = [smiles_list[i] for i in train_idx]
        valid_smiles = [smiles_list[i] for i in val_idx]
        return train_dataset, valid_dataset, (train_smiles, valid_smiles)
