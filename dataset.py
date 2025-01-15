from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import scipy.sparse as sps
import pickle
import copy
import matplotlib.pylab as plt

#Check (core) data is downloaded, and if not download it
import getData

class dataset(Dataset):
    def __init__(self,x,y,A,B,invariant,device):
        self.u_in = torch.from_numpy(x[0].astype(np.float32)).to(device)
        self.v_in = torch.from_numpy(x[1].astype(np.float32)).to(device)
        self.p_in = torch.from_numpy(x[2].astype(np.float32)).to(device)

        self.u_out = torch.from_numpy(y[0].astype(np.float32)).to(device)
        self.v_out = torch.from_numpy(y[1].astype(np.float32)).to(device)
        self.p_out = torch.from_numpy(y[2].astype(np.float32)).to(device)
        
        self.length = self.u_in.shape[0]

        Ainv = sps.linalg.inv(A).todense()
        self.Ainv = torch.from_numpy(Ainv.astype(np.float32)).to(device)
        self.invariant = invariant
        self.B = torch.from_numpy(B.todense().astype(np.float32)).to(device)
        
    def __getitem__(self,idx):
        x = [self.u_in[idx,:,:], self.v_in[idx,:,:], self.p_in[idx,:,:]]
        y = [self.u_out[idx,:,:], self.v_out[idx,:,:], self.p_out[idx,:,:]]

        return x, y
    
    def __len__(self):
        return self.length

def all_data(stacksize=-1,device='cpu'):
    with open('data_swe.pickle','rb') as file:
        data = pickle.load(file)

        #Pick up data
        p_in = np.array(data['p_c'])
        p_out = np.array(data['p_f'])
        u_in = np.array(data['u_c'])
        u_out = np.array(data['u_f'])
        v_in = np.array(data['v_c'])
        v_out = np.array(data['v_f'])
        A = data['A']
        B = data['B']
        invariant = {'mass': data['mass'],
                     'energy': data['energy']}

    #Stack all simulations
    if stacksize==-1:
        pass
    else:
        p_in = p_in[:,:stacksize,:]
        p_out = p_out[:,:stacksize,:]
        u_in = u_in[:,:stacksize,:]
        u_out = u_out[:,:stacksize,:]
        v_in = v_in[:,:stacksize,:]
        v_out = v_out[:,:stacksize,:]
            
    data_length = p_in.shape[0]

    #Split into training and test data
    split = int(0.7 * data_length) #70% training data

    train_in = u_in[:split,:], v_in[:split,:], p_in[:split,:]
    train_out = u_out[:split,:], v_out[:split,:], p_out[:split,:]
    test_in = u_in[split+1:,:], v_in[split+1:,:], p_in[split+1:,:]
    test_out = u_in[split+1:,:], v_out[split+1:,:], p_out[split+1:,:]

    #Shove in torch.Dataset class
    trainset = dataset(train_in,train_out,A=A,B=B,invariant=invariant,device=device)
    testset = dataset(test_in,test_out,A=A,B=B,invariant=invariant,device=device)

    return trainset, testset
