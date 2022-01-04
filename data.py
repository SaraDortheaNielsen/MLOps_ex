import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def mnist():
    # exchange with the corrupted mnist dataset
    #train = torch.randn(50000, 784)
    test = torch.randn(10000, 784) 

    #train
    train0 = np.load("train_0.npz")
    train1 = np.load("train_1.npz")
    train2 = np.load("train_2.npz")
    train3 = np.load("train_3.npz")
    train4 = np.load("train_4.npz")
    
    t0 = torch.from_numpy(train0.f.images)
    t1 = torch.from_numpy(train1.f.images)
    t2 = torch.from_numpy(train2.f.images)
    t3 = torch.from_numpy(train3.f.images)
    t4 = torch.from_numpy(train4.f.images)
    t_tot = torch.cat((t0,t1,t2,t3,t4), 0)

    l0 = torch.from_numpy(train0.f.labels)
    l1 = torch.from_numpy(train1.f.labels)
    l2 = torch.from_numpy(train2.f.labels)
    l3 = torch.from_numpy(train3.f.labels)
    l4 = torch.from_numpy(train4.f.labels)
    l_tot = torch.cat((l0,l1,l2,l3,l4), 0)

    
    #train = {'images': t_tot, 'labels': l_tot}

    #train
    test0 = np.load("train_0.npz")
    t = torch.from_numpy(train0.f.images)
    l = torch.from_numpy(train0.f.labels)
    #test = {'images': t, 'labels': l}


    


    train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
   

    return train, test


def load_mn():