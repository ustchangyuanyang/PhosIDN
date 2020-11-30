import functools
import itertools
import os
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import csv
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

def predict_for_phosidn(train_file_name,sites,predictFrame = 'general',
                         hierarchy=None, kinase=None):
    '''
    :param train_file_name: input of your prdict file
                            it must be a .csv file and theinput format  is proteinName, postion,sites, shortseq
    :param sites: the sites predict: site = 'S','T' OR 'Y'
    :param predictFrame: 'general' or 'kinase'
    :param hierarchy: if predictFrame is kinse: you must input the hierarchy:
            group,family,subfamily,kinase to choose corresponding model
    :param kinase: kinase name
    :return:
     a file with the score
    '''


    win = 71
   
    from methods.dataprocess_predict import getMatrixInput
    [X_test,y_test,ids,position] = getMatrixInput(train_file_name, sites, win)
    ppi_features_test = get_ppi_features(ids)

#     print X_test1.shape
#     print len(position)

    from methods.model_phosidn import model_phosidn
    model = model_phosidn(X_test,ppi_features_test, y_test,nb_epoch = 0)

    #load model weight
    if predictFrame == 'general':
        outputfile = 'general_{:s}'.format(site)
        if site == ('S','T'):
            model_weight = './models/PhosIDN/model_general_ST.h5'
        if site == 'Y':
            model_weight = './models/PhosIDN/model_general_Y.h5'


    if predictFrame == 'kinase':
        outputfile = 'kinase_{:s}_{:s}'.format(hierarchy, kinase)
        model_weight = './models/PhosIDN/model_{:s}_{:s}.h5'.format(hierarchy, kinase)
#     print model_weight
    model.load_weights(model_weight)
    predictions_t = model.predict([X_test,ppi_features_test,np.ones(X_test.shape[0])])
    results_ST = np.column_stack((ids, position,predictions_t[:, 1]))

    result = pd.DataFrame(results_ST)
    result.to_csv(outputfile + "prediction_phosphorylation.txt", index=False, header=None, sep='\t',
                  quoting=csv.QUOTE_NONNUMERIC)
if __name__ == '__main__':
    train_file_name = 'test data.csv'
    site = 'S','T'
    predict_for_phosidn(train_file_name, site, predictFrame='kinase',
                         hierarchy='group', kinase='AGC')