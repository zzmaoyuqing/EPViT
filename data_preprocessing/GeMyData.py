import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import random_split


seed = 42
torch.manual_seed(seed)

class GeMyDataset(Dataset):
    """ my dataset: ensure the data and target connected
    ---------------------PPI5093----------------------------
    input file : dot_emb_sub   ndarray:(5093,224,224)
                 protein_target.txt
                            x_data     ndarray:(5093,224,224)
                            y_data     ndarray:(5093,)
                            protein    ndarray:(5093,)

    ---------------------PPI2708----------------------------
    input file : dot_emb_sub.npy    ndarray:(2708,224,224)
                 protein_targe.txt
                            x_data     ndarray:(2708,224,224)
                            y_data     ndarray:(2708,)
                            protein    ndarray:(2708,)

    ---------------------PPI3672----------------------------
    input file : dot_emb_sub.npy    ndarray:(3672,224,224)
                 protein_target.txt
                            x_data     ndarray:(3672,224,224)
                            y_data     ndarray:(3672,)
                            protein    ndarray:(3672,)
    """

    def __init__(self, dot_emb_sub, target_file):
        X = np.array(dot_emb_sub)
        y = pd.read_csv(target_file)

        self.len = X.shape[0]
        self.protein_target = y
        self.protein = self.protein_target.iloc[:, 1].values
        self.x_data = X
        self.y_data = self.protein_target.iloc[:, 2].values

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index], self.protein[index]

    def __len__(self):

        return self.len

# Split data as train_data, val data, and test data (0.6:0.2:0.2)
def split_data(dataset):
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset= random_split(
        dataset=dataset,
        lengths=[0.6, 0.2, 0.2])
    return train_dataset, val_dataset, test_dataset

# Split data as train_data and test data (0.8:0.2)
def split_data2(dataset):
    torch.manual_seed(seed)
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[0.8, 0.2])
    return train_dataset, test_dataset

def read_split_data(dataset, splited_dataset):
    indices = splited_dataset.indices
    splited_data = []
    for i in range(0, len(indices)):
        a = dataset.x_data[indices[i]]
        splited_data.append(a)
    return splited_data


def read_split_target(dataset, splited_dataset):
    indices = splited_dataset.indices
    splited_target = np.zeros((1, len(indices)))
    for i in range(0, len(indices)):
        splited_target[0, i] = dataset.y_data[indices[i]]
    return splited_target


