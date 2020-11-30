from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers import Input, merge, Flatten,Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.regularizers import l2
import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Conv1D,Conv2D, MaxPooling2D

window_size1 = 71

def conv_factory(x, init_form, nb_filter, filter_size_block, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout

    :param x: Input keras network
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras network with b_norm, relu and convolution2d added
    :rtype: keras network
    """
    #x = Activation('relu')(x)
    x = Conv1D(nb_filter, filter_size_block,
                      init=init_form,
                      activation='relu',
                      border_mode='same',
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, init_form, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D

    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool

    """
    #x = Activation('relu')(x)
    x = Conv1D(nb_filter, 1,
                      init=init_form,
                      activation='relu',
                      border_mode='same',
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    #x = AveragePooling2D((2, 2),padding='same')(x)
    x = AveragePooling1D(pool_size=2, padding='same')(x)

    return x


def denseblock(x, init_form, nb_layers, nb_filter, growth_rate,filter_size_block,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones

    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """

    list_feat = [x]
    concat_axis = -1

    for i in range(nb_layers):
        x = conv_factory(x, init_form, growth_rate, filter_size_block, dropout_rate, weight_decay)
        list_feat.append(x)
        x = merge(list_feat, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate
    return x

class Position_Embedding(Layer):



    def __init__(self, size=None, mode='concat', **kwargs):

        self.size = size  

        self.mode = mode

        super(Position_Embedding, self).__init__(**kwargs)



    def call(self, x):

        if (self.size == None) or (self.mode == 'concat'):

            self.size = int(x.shape[-1])


        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)

        position_j = K.expand_dims(position_j, 0)

        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  

        position_i = K.expand_dims(position_i, 2)

        position_ij = K.dot(position_i, position_j)

        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)

        if self.mode == 'sum':

            return position_ij + x

        elif self.mode == 'concat':

            return K.concatenate([position_ij, x], 2)
        
    def compute_output_shape(self, input_shape):

        if self.mode == 'sum':

            return input_shape

        elif self.mode == 'concat':

            return (input_shape[0], input_shape[1], input_shape[2] + self.size)
        
class Self_Attention(Layer):     
    def __init__(self, output_dim, **kwargs):        
        self.output_dim = output_dim        
        super(Self_Attention, self).__init__(**kwargs)   
        
    def build(self, input_shape):        
                    
        self.kernel = self.add_weight(name='kernel',                                      
                                      shape=(3,input_shape[2], self.output_dim),                                      
                                      initializer='uniform',                                      
                                      trainable=True)     
        
        super(Self_Attention, self).build(input_shape)  
           
    def call(self, x):        
        WQ = K.dot(x, self.kernel[0])        
        WK = K.dot(x, self.kernel[1])        
        WV = K.dot(x, self.kernel[2])  
        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))               
        QK = QK / (self.output_dim**0.5)        
        QK = K.softmax(QK)         
        V = K.batch_dot(QK,WV)         
        return V     
    
    def compute_output_shape(self, input_shape):         
        return (input_shape[0],input_shape[1],self.output_dim)

def PhosIDN(nb_classes, nb_layers,img_dim1, img_dim4,init_form, nb_dense_block,
             growth_rate,filter_size_block1,
             nb_filter, filter_size_ori,
             dense_number,self_number,dropout_rate,dropout_dense,weight_decay):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    # first input of 33 seq #
    main_input = Input(shape=img_dim1, name='{:d}_seq_input'.format(window_size1))
    #model_input = Input(shape=img_dim)
    # Initial convolution
    x1 = Conv1D(nb_filter, filter_size_ori,
                      init = init_form,
                      activation='relu',
                      border_mode='same',
                      name='{:d}_initial_conv1D'.format(window_size1),
                      bias=False,
                      W_regularizer=l2(weight_decay))(main_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x1 = transition(x1, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)
    
    
    x = Position_Embedding()(x1)
    
    x = Self_Attention(self_number)(x)
    
    x = Activation('relu')(x)
    
    x = Dropout(dropout_rate)(x)
     
    x = Flatten()(x)
    
    x = Dense(32,
              name ='Dense_2',
              activation='relu',init = init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)
    
    x = Dropout(dropout_dense)(x)
    
    input4 = Input(shape=img_dim4, name='ppi_input')
    
    x4 = Dense(64,
              name ='Dense_ppi1',
              activation='relu',init = init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(input4)
    
    x4 = Dropout(dropout_dense)(x4)
    
    x4 = Dense(32,
              name ='Dense_ppi2',
              activation='relu',init = init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x4)
    
    x4 = Dropout(dropout_dense)(x4)
    
    bias_model = Input(shape=(1,), name='bias')
    
    seq_model = merge([bias_model, x], mode='concat')
    
    seq_model = Reshape((1, 32 + 1))(seq_model)
    
    
    ppi_model = merge([bias_model, x4], mode='concat')
    
    ppi_model = Reshape((1, 32 + 1))(ppi_model)
    
    dot = merge([seq_model, ppi_model], mode='dot', dot_axes=1, name='dot_layer') 
    
    x = Flatten()(dot)
    
    x = Dense(64,
              name ='Dense_3',
              activation='relu',init = init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)
    
    x = Dropout(dropout_dense)(x)
    
    x = Dense(dense_number,
              name ='Dense_4',
              activation='relu',init = init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)
    
    x = Dropout(dropout_dense)(x)
    
    x = Dense(nb_classes,
              name = 'Dense_softmax',
              activation='softmax',init = init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)

    phosidn = Model(input=[main_input,input4,bias_model], output=[x], name="PhosIDN")
    #feauture_dense = Model(input=[main_input, input2, input3], output=[x], name="multi-DenseNet")

    return phosidn

def PhosIDNSeq(nb_classes, nb_layers,img_dim1, init_form, nb_dense_block,
               growth_rate,filter_size_block1,
               nb_filter, filter_size_ori,
               dense_number,self_number,dropout_rate,dropout_dense,weight_decay):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    # first input of 33 seq #
    main_input = Input(shape=img_dim1, name='{:d}_seq_input'.format(window_size1))
    #model_input = Input(shape=img_dim)
    # Initial convolution
    x1 = Conv1D(nb_filter, filter_size_ori,
                      init = init_form,
                      activation='relu',
                      border_mode='same',
                      name='{:d}_initial_conv1D'.format(window_size1),
                      bias=False,
                      W_regularizer=l2(weight_decay))(main_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x1 = transition(x1, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)
    
    
    x = Position_Embedding()(x1)
    
    x = Self_Attention(self_number)(x)
    
    x = Activation('relu')(x)
    
    x = Dropout(dropout_rate)(x)
     
    x = Flatten()(x)
    
    x = Dense(32,
              name ='Dense_2',
              activation='relu',init = init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)
    
    x = Dropout(dropout_dense)(x)
    
    x = Dense(nb_classes,
              name = 'Dense_softmax',
              activation='softmax',init = init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)

    phosidnseq = Model(input=[main_input], output=[x], name="PhosIDNSeq")
    #feauture_dense = Model(input=[main_input, input2, input3], output=[x], name="multi-DenseNet")

    return phosidnseq
