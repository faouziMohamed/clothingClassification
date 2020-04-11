from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import tensorflow as tf
from external.rop import read_only_properties
from dataset import Dtset

"""
This module allow developper to manage the tensorflow.keras.Sequential model
"""

class Model:
    
    """
    Create a sequential model from tensorflow.keras.Sequential.
    The created model will have :
        * 3 Convolution 3D layers
        * 3 MaxPoolinglayers
        * 3 Dropout layers
        * 2 Dense layers :
            - one is a hidden layer with by default 256 neurones
            - second is the ouput layer with by default 10 neurones (output)
            
            
    This default model is compatible with keras.datasets.fashion_mnist dataset.
    Dataset that contain 10 classes for categorization
    """
    def __init__(self,filters=64, kernel_size=(3,3), 
                 padding='same', 
                 output_activation= 'softmax', input_activation='relu',
                 hidden_activation= 'relu',    input_shape=(28,28,1),
                 hidden_neurones  = 256,       ouput_neurones=10,
                 optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=[],verbose=1):

        self.conv2D_filters    = filters
        self.kernel_size       = kernel_size
        self.padding           = padding
        self.output_activation = output_activation
        self.input_activation  = input_activation
        self.hidden_activation = hidden_activation
        self.images_height     = input_shape[0]
        self.images_width      = input_shape[1]
        self.input_shape       = input_shape
        self.hidden_nn         = hidden_neurones
        self.output_neurones   = ouput_neurones
        self.optimizer         = optimizer
        self.loss              = loss
        self.metrics           = metrics
        self.verbose           = verbose
        self.i = 1
        self.__model = None
        self.__compiled = False

        self.new_model()
        self.compile()
    
    #@read_only_properties('__model')
    def new_model(self):
        """
            This methode create a new tensorflow.keras.Sequential model the methode with actual configuration
            and return the model
        """
        self.__compiled = False
        self.__model = Sequential(name=f'Sequential_Model_{self.i}')
        self.i    += 1
        self.__model.add(Conv2D(filters = self.conv2D_filters,
                         kernel_size  = self.kernel_size,
                         padding      = self.padding,
                         input_shape  = self.input_shape,
                         activation   = self.input_activation,
                         name='1rs_Conv2D'
                        ))
        self.__model.add(MaxPooling2D(2,2,name='MaxPooling_1'))
        self.__model.add(Dropout(0.2,name='Dropout_1'))
        
        self.__model.add(Conv2D(filters = self.conv2D_filters,
                         kernel_size  = self.kernel_size,
                         padding      = self.padding,
                         activation   = self.hidden_activation,
                         name='2nd_Conv2D'
                        ))
        self.__model.add(MaxPooling2D(2,2,name='MaxPooling_2'))
        self.__model.add(Dropout(0.2,name='Dropout_2'))
        
        """
        self.__model.add(Conv2D(filters = self.conv2D_filters,
                         kernel_size  = self.kernel_size,
                         padding      = self.padding,
                         activation   = self.hidden_activation,
                         name='3rd_Conv2D'
                        ))
        self.__model.add(MaxPooling2D(2,2,name='MaxPooling_3'))
        self.__model.add(Dropout(0.2,name='Dropout_3'))
        """
        
        self.__model.add(Flatten(name='Flatten_Layer'))
        self.__model.add(Dense(units=self.hidden_nn,
                                activation = self.hidden_activation,name='1st_dense_layer'))
        self.__model.add(Dense(10,activation = self.output_activation,name='2nd_dense_layer'))
        return self.__model
    
    def compile(self):
        self.__model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics)

    def __raise_incompatible_type(self,a,b):
        type_a = str(type(a)).split('\'')[1]
        type_b = str(type(b)).split('\'')[1]
        raise AttributeError(f"{type_a} object cannot be converted to '{type_b}' because '{type_a}' and '{type_b}' are not compatible")

    def __reshape_add_one(self, dataset):
        """
        Reshape numpy.ndarray from the shape(a,b,c) to shape(a,b,c,1)
        """
        x_train = dataset.x_train
        y_train = dataset.y_train
        x_validate = dataset.x_validate
        y_validate = dataset.y_validate
        x_test = dataset.x_test
        y_test = dataset.y_test

        images_width  = x_train.shape[1]
        images_height = x_train.shape[2]

        x_test     =     x_test.reshape(-1, images_width, images_height, 1)
        x_train    =    x_train.reshape(-1, images_width, images_height, 1)
        x_validate = x_validate.reshape(-1, images_width, images_height, 1)

        return (x_train,y_train),(x_validate,y_validate),(x_test,y_test)

    def __reshape_rem_one(self, dataset):
        """
        Reshape numpy.ndarray from the shape(a,b,c) to shape(a,b,c,1)
        """
        x_train = dataset.x_train
        y_train = dataset.y_train
        x_validate = dataset.x_validate
        y_validate = dataset.y_validate
        x_test = dataset.x_test
        y_test = dataset.y_test

        images_width  = x_train.shape[1]
        images_height = x_train.shape[2]

        x_test     =     x_test.reshape(-1, images_width, images_height)
        x_train    =    x_train.reshape(-1, images_width, images_height)
        x_validate = x_validate.reshape(-1, images_width, images_height)

        return (x_train,y_train),(x_validate,y_validate),(x_test,y_test)

    def fit(self, dataset,model=None, epochs=1, batch_size=None, callback=None):
        """
        Fit the model with the given dataset
        """
        #Exception check
        if type(dataset) is not Dtset:
            self.__raise_incompatible_type(dataset,Dtset())

        if (type(model) is not self) and (type(model) is not type(None)) :
            self.__raise_incompatible_type(model,self)

        #Values Check
        if (batch_size is not None) and (batch_size<0):
            batch_size=None
        if epochs < 1 :
            epochs = 1
            
        #Configuring the model
        if (model is None) :

            if self.__model is None:
                model = self.new_model()
                model.compile(optimizer=self.optimizer,
                              loss=self.loss,metrics=self.metrics)
            else:
                if self.__compiled is not True:
                    self.compile()
                model = self.__model
                
        (x_train,y_train),(x_validate,y_validate),(_,_) = self.__reshape_add_one(dataset=dataset)
        if (dataset.x_validate.shape[0] is not 0):
            validation = (x_validate,y_validate)
        else:
            validation = None

        model.fit(
              x=x_train,
              y=y_train,
              batch_size=batch_size,
              validation_data=validation,
              epochs=epochs,
              verbose = self.verbose,
              callbacks=callback)
    
    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self,mod):
        if type(mod) is not type(Sequential()):
            self.__raise_incompatible_type(mod, keras.Sequential())
        self.__model = mod
