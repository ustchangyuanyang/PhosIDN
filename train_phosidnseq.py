import functools
import itertools
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from keras.layers import Dense, Activation, Flatten, Dropout, Reshape
from keras.layers import Conv1D,Conv2D, MaxPooling2D
from keras.models import Sequential,Model
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.optimizers import Adam,SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import copy

def train_for_phosidnseq(train_file_name,sites,predictFrame = 'general',background_weight = None):
    '''

    :param train_file_name:  input of your train file
                                it must be a .csv file and theinput format  is label,proteinName, postion,sites, shortsequence,
    :param site: the sites predict: site = 'S','T' OR 'Y'
    :param predictFrame: 'general' or 'kinase'
    :param background_weight: the model you want load to pretrain new model
                                usually used in kinase training
    :return:
    '''


    win = 71
   
    from methods.dataprocess_train import getMatrixLabel
    [X_train,y_train,ids] = getMatrixLabel(train_file_name, sites, win)


    modelname = "general_{:s}".format(sites)
    if predictFrame == 'general':
        modelname ="general_model_{:s}".format(sites)


    if predictFrame == 'kinase':
        modelname = "kinase_model_{:s}".format(sites)


    from methods.model_phosidnseq import model_phosidnseq

    model = model_phosidnseq(X_train, y_train,
               weights=background_weight)
    model.save_weights(modelname+'.h5',overwrite=True)



