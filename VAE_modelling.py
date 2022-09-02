### Import libraries


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from livelossplot import PlotLosses
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


### Dataset classes


def train_val_split_vae(X_train_val, y_train_val, other_proteomic):
    """
    Divide the train_val data into training and validation data.
    First step: get the train and test data from X_train_val and y_train_val
    Second step: for the unsupervised part of the training, add train and validation data from unlabeled cell lines of other_proteomic
    """
    # train and val data for the supervised part
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.2, random_state = 42)

    # for the unsupervised part
    np.random.seed(42)
    ind = np.random.choice(other_proteomic.index, size = int(0.8 * len(other_proteomic)), replace = False)
    other_train = other_proteomic.loc[ind]
    other_val = other_proteomic.loc[~other_proteomic.index.isin(ind)]

    X_train_vae = pd.concat([X_train, other_train])
    X_val_vae = pd.concat([X_val, other_val])

    return(X_train, X_val, y_train, y_val, X_train_vae, X_val_vae)

class ProtDataset(Dataset):
    """
    Dataset wrapping proteomic dataset.
    Each cell line is retrieved by indexing tensors along the first dimension.
    """
    def __init__(self, proteomic):
        self.proteomic = proteomic
        
    def __getitem__(self, index):
        return(self.proteomic[index])
    
    def __len__(self) -> int:
        return(len(self.proteomic))

class RegDataset(Dataset):
    """
    Dataset wrapping proteomic dataset.
    Each cell line is retrieved by indexing tensors along the first dimension.
    """
    def __init__(self, proteomic, drug, drug_index):
        self.proteomic = proteomic
        self.drug = drug
        self.drug_index = drug_index
        
    def __getitem__(self, index):
        return(self.proteomic[index], self.drug[index, self.drug_index])

    def __len__(self) -> int:
        return(len(self.proteomic))


### Model classes


class UnsupervisedVAE(nn.Module):
    def __init__(self):
        super(UnsupervisedVAE, self).__init__()

        self.encoder = nn.Sequential(
            # level 1
            nn.Linear(5153, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # level 2: divide between mean and log_var
            nn.Linear(1024, 64*2),
            nn.BatchNorm1d(64*2)
        )

        self.decoder = nn.Sequential(
            # level 2: from the latent space
            nn.Linear(64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # level 1
            nn.Linear(1024, 5153),
            nn.BatchNorm1d(5153),
            # no need for a sigmoid activation because we don't use BCE but MSE - see report to understand why
        )
        
    def reparameterise(self, mu, logvar):
        """
        Use the reparametrisation trick to get the embedding of an input
        mu and logvar are respectively the mean and log variance from the encoder's latent space
        """
        std = logvar.mul(0.5).exp_() # = exp(logvar*0.5) = exp(log(sqrt(var))) = std
        eps = std.data.new(std.size()).normal_()
        return(eps.mul(std).add_(mu))
        
    def forward(self, x):
        """
        First, encode the input data, then use the reparametrisation trick to get an embedding, then decode
        """
        mu_logvar = self.encoder(x).view(-1, 2, 64)
        
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        
        decoded = self.decoder(z) # := mean of the Gaussian's distribution of the output of the decoder
        
        return(decoded, mu, logvar)

class SupervisedVAE(nn.Module):
    def __init__(self, model):
        super(SupervisedVAE, self).__init__()
        
        self.encoder = model.encoder
        
        self.regressor = nn.Linear(64, 1) # just linear regression
        #self.regressor = nn.Sequential(
        #    # hidden regression layer
        #    nn.Linear(64, 10),
        #    nn.BatchNorm1d(10),
        #    nn.ReLU(),
        #    # output layer
        #    nn.Linear(10, 1),
        #    )
        
    def forward(self, x):
        mu_logvar = self.encoder(x).view(-1, 2, 64)
        mu = mu_logvar[:, 0, :]
        
        reg = self.regressor(mu)
        
        return(reg)


### Total loss function


def loss_function(x_hat, x, mu, logvar):
    """
    Loss function that adds the reconstruction loss (chosen to be MSE for reasons exlained in the report) 
    and the KL divergence.
    The KLD is calculated with its simplified form, which comes when the output of the encoder is gaussian. 
    """
    MSE = nn.functional.mse_loss(x_hat, x, reduction='sum') # reduction = sum adviced for VAE because mean can be instable

    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

    return MSE + KLD


