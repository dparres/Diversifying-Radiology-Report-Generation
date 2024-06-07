import torch
import numpy as np


####################
# HNM
####################

def multiassign(d, keys, values):
    # TODO: en todas las losses: reduction='none'
    '''print("k: ", keys.shape)
    print("v ", values.shape)'''
    d.update(zip(keys, values))

def Hard_Negative_Mining(dict_loss, dataset, BATCH_SIZE, value_crop=None, num_workers=23):

    # Use only losses higher than value_crop
    if value_crop is not None:
        dict_loss = {k:v for k,v in dict_loss.items() if v>value_crop}

    # Calculate Mean and Dev Std
    l_losses = np.array(list(dict_loss.values()))
    mean_ = np.mean(l_losses)
    std_ = np.std(l_losses)/2
    #print("Num samples to analize: ", len(l_losses))
    #print("Mean: ", mean_)
    #print("Std: ", std_)
    

    # New Dict with all values between:
    # mean-std and mean+std
    dict_loss_HNM = {k:v for k,v in dict_loss.items() if v>(mean_-std_) and v<(mean_+std_)}
    #print("Num HNM: ", len(list(dict_loss_HNM.values())))
    del l_losses

    # Create new dataset with hard examples
    indices_HNM = np.array(list(dict_loss_HNM.keys()))

    if len(indices_HNM) < 1:
        return False
    #print("indices_HNM: ", len(indices_HNM))
    subset_HNM_dataset = torch.utils.data.Subset(dataset, indices_HNM)
    #print("Tam subset: ", len(subset_HNM_dataset))

    HNM_trainloader = torch.utils.data.DataLoader(subset_HNM_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=24, collate_fn=dataset.get_collate_fn())

    return HNM_trainloader




def HNM_radgraph(calculated_rgs, dataset, BATCH_SIZE, N_TO_SELECT, num_workers=23):

    
    sorted_indices = np.argsort(calculated_rgs)

    # Index of the lowest value
    min_indexs = sorted_indices[0:N_TO_SELECT]
    #print("min_indexs: ", min_indexs)

    #print("indices_HNM: ", len(indices_HNM))
    subset_HNM_dataset = torch.utils.data.Subset(dataset, min_indexs)
    #print("Tam subset: ", len(subset_HNM_dataset))

    HNM_trainloader = torch.utils.data.DataLoader(
        subset_HNM_dataset, 
        batch_size=2, #BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=dataset.get_collate_fn())

    return HNM_trainloader, min_indexs

