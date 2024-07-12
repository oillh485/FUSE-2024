from pathlib import path
from ttt.utils import listfolders, walk_keys
from io_utils import(
    get_datasets, 
    get_particle_id_set, 
    repare_data, 
    conditional_get_datasets,
    PreloadedSegmentationDataset,
    prepare_data,
    get_dihedral_augmentations)
import training_script 
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam




    


def normalize_data(X):
    # Center data img wise around 0, std deviation set to 1
    means = np.mean(X, axis=(1, 2, 3))
    X = X - means[:, None, None, None]
    stds = np.std(X, axis=(1, 2, 3))
    return X / stds[:, None, None, None]


def main():

    base_seed = 123
    base_rng = np.random.default_rng(seed=base_seed)
    base_path = Path(
    '/global/cfs/cdirs/m3980/simulated_HRTEM_datasets/cdse_defocus_series_2024_jun_21/images_300'
    )
    

    df = [int(x) for x in np.arange(-250, 300, 50)]
    target_key = df[7]  # defocus of 10 nm

    
    batch_folders = listfolders(base_path)
    pids = get_particle_id_set(batch_folders)

    train_ids = set(base_rng.choice(list(pids.keys()), size=1536, replace=False))
    val_ids = set(list(pids.keys())).difference(train_ids)


    device = torch.device(0)
    print(device)
    print("Loading data")

    dataset = conditional_get_datasets(
        batch_folders, train_ids, val_ids, target_key=target_key
        )

    prepare_data(dataset)

    N_training_points = 256

    # Prepare the training/validation sets
    X_train = dataset[target_key]["X_train"]
    Y_train = dataset[target_key]["Y_train"]

    X_validation = dataset[target_key]["X_validation"]
    Y_validation = dataset[target_key]["Y_validation"]
    
    print("Normalizing inputs")
    X_train = normalize_data(X_train)
    X_validation = normalize_data(X_validation)

    print("Moving data to device and creating datasets")
    X, Y = torch.from_numpy(X_train), torch.from_numpy(Y_train)

    X = X.to(device)
    Y = Y.to(device)


    train_split = slice(0, 7 * X.shaoe[0] // 8)
    val_split = slice(7 * X.shape[0] // 8, X.shape[0])
    train_dataset = PreloadedSegmentationDataset(X[train_split], Y[train_split])
    val_dataset = PreloadedSegmentationDataset(X[val_split], Y[val_split])


    batch_size = 16
    N_epochs = 5
    lr = 1e-3
    criterion = CrossEntropyLoss()
    unet = get_model()
    unet.to(device)
    optimizer = Adam(unet.parameteres())

    history = training_loop(train_dataset, val_dataset, unet, optimizer, criterion, N_epochs)

    
    

    


