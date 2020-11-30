from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.layers import Dense, Activation, Flatten, Dropout, Reshape
from keras.layers import Conv1D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import copy


def model_phosidnseq(X_train, y_train,
              nb_epoch=80,weights=None):

    nb_classes = 2
    img_dim1 = X_train.shape[1:]


    ##########parameters#########

    nb_batch_size = 512

    init_form = 'RandomUniform'   #'glorot_normal' #'
    learning_rate = 0.0003
    nb_dense_block = 1
    nb_layers = 5
    nb_filter = 32
    growth_rate = 24
    # growth_rate = 24
   
    filter_size_block = 7
    filter_size_ori=1
    dense_number = 32
    self_number = 128
    dropout_rate =0.5
    dropout_dense = 0.3
    weight_decay=0.0001



    ###################
    # Construct model #
    ###################
    from phosnet import PhosIDNSeq
    model = PhosIDNSeq(nb_classes, nb_layers,img_dim1, init_form, nb_dense_block,
                       growth_rate,filter_size_block,
                       nb_filter, filter_size_ori,
                       dense_number,self_number,dropout_rate,dropout_dense,weight_decay)
    # Model output

    # choose optimazation
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # model compile
    model.compile(loss='binary_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

    # load weights#
    if weights is not None:
        model.load_weights(weights)
        # model2 = copy.deepcopy(model)
        model2 = model
        model2.load_weights(weights)
        for num in range(len(model2.layers) - 1):
            model.layers[num].set_weights(model2.layers[num].get_weights())

    if nb_epoch > 0 :
      model.fit(X_train, y_train, batch_size=nb_batch_size,
                         # validation_data=([X_val1, X_val2, X_val3, y_val),
                         # validation_split=0.1,
                         epochs= nb_epoch, shuffle=True, verbose=1)


    return model