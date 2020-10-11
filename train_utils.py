import torch 
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os 
import torchvision.models as models
import time
import copy
from torchvision import transforms
import random
from PIL import Image


def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dat!mkdir ../input/data/images
        patient_col (str): string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """
    df1_patients_unique = set(df1[patient_col].values)
    df2_patients_unique = set(df2[patient_col].values)

    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)
    leakage = len(patients_in_both_groups) > 0
    return leakage



def initialize_model(model_path=None,pretrained=False,device = torch.device("cpu")):
    """
    Return model with Loaded weights

    Args:
        model_path (model weight  file): path to pretrained model
        pretrained (bool) : True if dowoald model from image net
        device (torch.device("cpu"))
    
    Returns:
        Loaded Model
    """

    model = models.densenet121(pretrained=pretrained)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
                                     nn.Linear(num_ftrs,14 ),
                                     nn.Sigmoid())#model.classifier = torch.nn.Linear(1024,1)
    if model_path!=None:
        model.load_state_dict(torch.load(model_path,map_location=device))
    
    return model

class Weighted_Bce(nn.Module):

    def __init__(self,Positive_weights,Negative_weights,device,eps=0.000001):
        super(Weighted_Bce,self).__init__()
        self.pos_weights = torch.tensor(Positive_weights).detach().to(device)
        self.neg_weights = torch.tensor(Negative_weights).detach().to(device)
        self.eps = eps

    def forward(self,output,target):
        Loss = -(self.pos_weights*target * torch.log(output+self.eps) + self.neg_weights*(1-target)*torch.log(1-output+self.eps))
        return Loss.sum()


def train_model(model, dataloaders, criterion, optimizer,scheduler, num_epochs=25):
    since = time.time()
    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        outputs  = model(inputs)
                        loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)



            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, "--"))

            if phase == 'val':
                scheduler.step(epoch_loss)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model

