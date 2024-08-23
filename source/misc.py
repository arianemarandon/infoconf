import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def compute_emp_pvalue(test_statistic, null_statistics):
    return (1 + np.sum(null_statistics >= test_statistic)) / (len(null_statistics)+1)

def BH(pvalues, level): 
    """
    Benjamini-Hochberg procedure. 
    """
    n = len(pvalues)
    pvalues_sort_ind = np.argsort(pvalues) 
    pvalues_sort = np.sort(pvalues) #p(1) < p(2) < .... < p(n)

    comp = pvalues_sort <= (level* np.arange(1,n+1)/n) 
    #get first location i0 at which p(k) <= level * k / n
    comp = comp[::-1] 
    comp_true_ind = np.nonzero(comp)[0] 
    i0 = comp_true_ind[0] if comp_true_ind.size > 0 else n 
    nb_rej = n - i0

    return pvalues_sort_ind[:nb_rej]


def train_test_split_wrapper(X,Y, test_size, random_state):
    if isinstance(X, np.ndarray):
        return train_test_split(X,Y, test_size=test_size, random_state=random_state)
    elif isinstance(X, torch.utils.data.Dataset):
        if test_size < 1:
            test_size = int(len(X)*test_size)
        train_size=len(X)-test_size
        if random_state is not None:
            train_dataset, test_dataset = torch.utils.data.random_split(X, [train_size, test_size], generator=torch.Generator().manual_seed(random_state))
        else: train_dataset, test_dataset = torch.utils.data.random_split(X, [train_size, test_size])
        y_train, y_test = np.array([y for _,y in train_dataset]), np.array([y for _,y in test_dataset])
        return train_dataset, test_dataset, y_train, y_test

    else: raise NotImplementedError




class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = torch.utils.data.Subset(dataset, indices)
    def __getitem__(self, idx):
        return self.dataset[idx]
    def __len__(self):
        return len(self.dataset)
    

class custom_subset_and_targets(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels
    def __getitem__(self, idx):
        image=self.dataset[idx][0]
        target=self.targets[idx]
        return (image,target)
    def __len__(self):
        return len(self.dataset)
    

