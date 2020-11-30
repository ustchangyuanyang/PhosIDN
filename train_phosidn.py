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
import scipy.io as sio

def get_ppi_features(ids):
    '''
    :param phase: train,test,val
    :param hierarchy:
    :param kinase:
    :return:
    '''

    
    #===========LINE features==============
    # ppi_result_path = "../LINE/vec_2nd_wo_norm.txt"
    # feature_dim = 32
    # f = open(ppi_result_path, 'r')
    # alllines = f.readlines()
    # f.close()
    # feature_dict = {}
    # for i,line in enumerate(alllines):
    #     if i>0:
    #         oneline = line[:-1]
    #         onelist = oneline.split()
    #         protein_name = onelist[0]
    #         features_a = onelist[1:]
    #         features = list(map(lambda x: float(x), features_a))
    #         feature_dict[protein_name] = features

    #==============for SDNE features=============
    #load ppi matrix
    ppi_matrix = sio.loadmat("embedding.mat")
    mat = ppi_matrix["embedding"]
    feature_dim = mat.shape[1]

    feature_dict = {}
    #read protein name
    f = open("protein name of SDNE.txt","r")
    alllines = f.readlines()
    f.close()
    for i,row in enumerate(alllines):
        protein = row[:-2]
        feature_dict[protein] = mat[i,:]

    # print  feature_dict['Q6P3W7']

    ppi_features = []
    zero_vec = np.zeros(feature_dim)

    
    num_of_sites = 0
    for i in range(len(ids)):
        protein = ids[i]
        num_of_sites = num_of_sites + 1
        if protein in feature_dict.keys():
            ppi_features.append(feature_dict[protein])
        else:
            ppi_features.append(zero_vec)

    ppi_features = np.array(ppi_features)
    np.reshape(ppi_features, (num_of_sites, feature_dim))

    return ppi_features
def train_for_phosidn(train_file_name,sites,predictFrame = 'general',background_weight = None):
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
    ppi_features_train = get_ppi_features(ids)

    modelname = "general_{:s}".format(sites)
    if predictFrame == 'general':
        modelname ="general_model_{:s}".format(sites)


    if predictFrame == 'kinase':
        modelname = "kinase_model_{:s}".format(sites)


    from methods.model_phosidn import model_phosidn

    model = model_phosidn(X_train, ppi_features_train, y_train,
               weights=background_weight)
    model.save_weights(modelname+'.h5',overwrite=True)



