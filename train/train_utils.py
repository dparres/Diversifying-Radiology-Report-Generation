import torch
import numpy as np

####################
# HNM
####################

def multiassign(d, keys, values):
    d.update(zip(keys, values))

def Hard_Negative_Mining(dict_loss, dataset, BATCH_SIZE, value_crop=None, num_workers=23):

    # Use only losses higher than value_crop
    if value_crop is not None:
        dict_loss = {k:v for k,v in dict_loss.items() if v>value_crop}

    # Calculate Mean and Dev Std
    l_losses = np.array(list(dict_loss.values()))
    mean_ = np.mean(l_losses)
    std_ = np.std(l_losses)/2

    # New Dict with all values between: mean-std and mean+std
    dict_loss_HNM = {k:v for k,v in dict_loss.items() if v>(mean_-std_) and v<(mean_+std_)}
    del l_losses

    # Create new dataset with hard examples
    indices_HNM = np.array(list(dict_loss_HNM.keys()))

    if len(indices_HNM) < 1:
        return False
    subset_HNM_dataset = torch.utils.data.Subset(dataset, indices_HNM)

    HNM_trainloader = torch.utils.data.DataLoader(
        subset_HNM_dataset, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=num_workers, collate_fn=dataset.get_collate_fn()
        )

    return HNM_trainloader