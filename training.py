from itertools import product
from itertools import repeat
import numpy as np
import pathlib as Path

from ttt.utils import listfolders
from io_utils import(
    conditional_get_datasets,
    get_particle_id_set,
    prepare_data,
    PreloadedSegmentationDataset,
    get_dihedral_augmentations
)
from basic_transfer_learning_script import freeze_model
from mlr.database.utils import (
    write_metadata,
    get_new_folder_parallel
)
from mlr.projects.ood_gen.unet_dynamics_training import (
    normalize_data,
    get_model,
    training_loop,
    prepare_callbacks
)

import torch
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch import from_numpy
from copy import deepcopy



# Gets all unique combinations of pretraining and transfer learning defoci cominations
# Making sure unique defoci are used for the pretraining and transfer learning steps (eg no 0 df pretrain followed by more 0 df training)
def get_possible_combos(pretrain_options, transfer_learn_options):
     all_options = []
     for p in pretrain_options:
         for t in transfer_learn_options:
             if p != t:
                 all_options.append((p, t))
 
     return list(set(all_options))



def get_target_key(work):
    res = set()
    for task in work:
        pretrain_df, transfer_df = task[0]
        res.update([pretrain_df], [transfer_df])
    return res



def get_work_partitions(pretrain_defoci):
    pretrain_map = {df:i for i, df in enumerate(pretrain_defoci)}

    tl_defoci = np.arange(-250, 300, 50)
    N_tl_examples = [128, 256, 512]
    freeze_options = ('encoder', 'decoder', None)
    pretrain_lr = [1e-3, 1e-4]
    tl_lr = [1e-4, 1e-5]

    all_work = list(product(get_possible_combos(pretrain_defoci, tl_defoci), 
                        N_tl_examples,
                        freeze_options,
                        pretrain_lr,
                        tl_lr))
    
    work_partitions = {}
    for work_tuple in all_work:
        pretrain_dataset = pretrain_map[work_tuple[0][0]]

        if pretrain_dataset in work_partitions:
            work_partitions[pretrain_dataset].append(work_tuple)
        else:
            work_partitions[pretrain_dataset] = [work_tuple]

    return work_partitions



def get_work_portion(rank, N_processes):
    pretrain_defoci = (-100, -50, 0, 50, 100)
    N_pretrain_defoci = len(pretrain_defoci)
    work_subset = get_work_partitions(pretrain_defoci)[rank % N_pretrain_defoci]
    
    sub_rank = rank // N_pretrain_defoci
    N_sub_procesess = N_processes // N_pretrain_defoci
    return work_subset[sub_rank::N_sub_procesess]



def work_dispatch_test():
    from functools import reduce

    work, all_work = get_work_partitions()
    print(work.keys())

    N_processes = 600
    all_work_test = reduce(lambda x, y: x.union(y),
                            [set(get_work_portion(work, rank, N_processes))
                              for rank in range(N_processes) ])
    all_work_test_2 = reduce(lambda x, y: x + y,
                            [get_work_portion(work, rank, N_processes)
                              for rank in range(N_processes) ])
    

    print(f"Got everything? {all_work_test == all_work}")
    print(f"Only everything? {len(all_work_test_2) == len(all_work)}")



def run_training(device,
                 dataset,
                 defoci,
                 N_tl_training_points,
                 freeze_option,
                 pretrain_lr,
                 tl_lr
                 ):

    unet = get_model().to(device)
    criterion = CrossEntropyLoss()

    training_target_key = defoci
    batch_size = 16
    N_epochs = [1,1]
    N_pretrain_points = 512
    N_training_points = [N_pretrain_points, N_tl_training_points]
    learning_rate = [pretrain_lr, tl_lr]

    print(f'pretrain_defocus: {defoci[0]}')
    print(f'tl_defocus: {defoci[1]}')
    print(f'N_tl_training_points: {N_tl_training_points}')
    print(f'freeze_option: {freeze_option}')
    print(f'pretrain_lr: {pretrain_lr}')
    print(f'tl_lr: {tl_lr}')

    tl_metadata = {}
    weights = {}
    for idx, v in enumerate(training_target_key):
        X_train = dataset[v]["X_train"]
        Y_train = dataset[v]["Y_train"]
        train_dataset = PreloadedSegmentationDataset(torch.from_numpy(get_dihedral_augmentations(X_train[:N_training_points[idx], ...]))[:, None, :, :].to(device),
                                                torch.from_numpy(get_dihedral_augmentations(Y_train[:N_training_points[idx],...])).to(device))
        dataset["target"] = {"defocus": v, "dataset": train_dataset}
        
        optimizer = Adam(model.parameters(), lr=learning_rate[idx])
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
        
        if idx == 1:
            model = freeze_model(model, freeze_option)
        
        history = training_loop(dataset, model, optimizer, scheduler, criterion, N_epochs[idx], shuffle=True, batch_size=batch_size)
        weights[f"training_session_{idx}"] = deepcopy(model.state_dict())
        
        if idx == 1:
            tl_metadata[f"training_session_{idx}"] = {"history": history, "learning_rate": learning_rate[idx],
                                                    "N_epochs": N_epochs[idx], "batch_size": batch_size,
                                                    "frozen_section": freeze_option
                                                    }
        else:
            tl_metadata[f"training_session_{idx}"] = {"history": history, "learning_rate": learning_rate[idx],
                                                    "N_epochs": N_epochs[idx], "batch_size": batch_size
                                                    }





def main(rank, N_processes):
    ## we know my_work has all the same pretrain defocus
    # so load that out front

    base_seed = 123
    base_rng = np.random.default_rng(seed=base_seed)
    base_path = Path('/pscratch/sd/h/hoill/defocus_testing_v1/images_300')

    batch_folders = listfolders(base_path)
    pids = get_particle_id_set(batch_folders)
    train_ids = set(base_rng.choice(list(pids.keys()), size=1536, replace=False))
    val_ids = set(list(pids.keys())).difference(train_ids)

    device = torch.device(0)

    my_work = get_work_portion(rank, N_processes)
    data_target_key = get_target_key(my_work)

    dataset = conditional_get_datasets(batch_folders, train_ids, val_ids, target_key=data_target_key)
    prepare_data(dataset)
    prepare_callbacks(dataset, device)

    for work in my_work:
        run_training(device, dataset, *work) 
    


if __name__ == '__main__':
    import os
    # rank = int(os.environ["SLURM_PROCID"])
    # N_processes = int(os.environ["SLURM_NTASKS"])
    main(0,600)





