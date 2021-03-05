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



def initialize_model(model_path=None,pretrained=False,mode='train',device = torch.device("cpu")):
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
    model.classifier = nn.Linear(num_ftrs,14 )#model.classifier = torch.nn.Linear(1024,1)
    if model_path!=None:
        model.load_state_dict(torch.load(model_path,map_location=device))
    if mode == 'eval':
        model = nn.Sequential(
                                model,
                                nn.Sigmoid())
    return model

def confusion_mat(preds,labels,threshold=0.5):
    preds = np.where(preds<threshold,0,1)
    TP = np.sum(np.logical_and(preds == 1, labels == 1))
    TN = np.sum(np.logical_and(preds == 0, labels == 0))
    FP = np.sum(np.logical_and(preds == 1, labels == 0))
    FN = np.sum(np.logical_and(preds == 0, labels == 1))

    return TP,TN,FP,FN

