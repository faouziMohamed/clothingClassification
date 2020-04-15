from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from external.rop import read_only_properties
from tensorflow   import keras
from dataset      import Dtset
from sys          import exit

import tensorflow as tf
import matplotlib.pyplot as plt

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
                 metrics=[],verbose=1, dataset=Dtset(validate_size=15000, predict_size=1500)):

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
        self.i                 = 1
        self.__model           = None
        self.__compiled        = False
        self.dataset           = dataset
        self.batch_size        = 64
        self.epochs            = 1
        self.prediction        = None

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
        
        
        self.__model.add(Conv2D(filters = self.conv2D_filters,
                         kernel_size  = self.kernel_size,
                         padding      = self.padding,
                         activation   = self.hidden_activation,
                         name='3rd_Conv2D'
                        ))
        self.__model.add(MaxPooling2D(2,2,name='MaxPooling_3'))
        self.__model.add(Dropout(0.2,name='Dropout_3'))
        
        
        self.__model.add(Flatten(name='Flatten_Layer'))
        self.__model.add(Dense(units=self.hidden_nn,
                                activation = self.hidden_activation,name='1st_dense_layer'))
        self.__model.add(Dense(units=256,
                                activation = self.hidden_activation,name='2nd_dense_layer'))
        self.__model.add(Dense(10,activation = self.output_activation,name='3rd_dense_layer'))
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

    def __check_data_and_parameters(self,**kwargs):
        if type(kwargs['dataset']) is not Dtset:
            self.__raise_incompatible_type(kwargs['dataset'],Dtset())

        if (type(kwargs['model']) is not self) and (type(kwargs['model']) is not type(None)) :
            self.__raise_incompatible_type(kwargs['model'],self)

    def __config_model(self,**kwargs):
        #Values Check
        if (kwargs['batch_size'] is not None) and (kwargs['batch_size']<0):
            self.batch_size=None
        if kwargs["epochs"] < 1 :
            self.epochs = 1
            
        #Configuring the model
        if (kwargs['model'] is None) :

            if self.__model is None:
                self.new_model()
                self.compile(optimizer=self.optimizer,
                              loss=self.loss,metrics=self.metrics)
            else:
                if self.__compiled is not True:
                    self.compile()
                    self.__compiled = True

    def fit(self, dataset,model=None, epochs=1, batch_size=None, callback=None):
        """
        Fit the model with the given dataset
        """
        #Exception check
        self.__check_data_and_parameters(dataset=dataset, model=model)
        #check value

        self.__config_model(batch_size=batch_size, epochs=epochs, model=model )
        
        self.dataset = dataset
        #reshape the data by adding 1 on the 4th dimension
        (self.dataset.x_train,self.dataset.y_train),\
        (self.dataset.x_validate,self.dataset.y_validate),\
        (self.dataset.x_test,self.dataset.y_test) = self.__reshape_add_one(dataset=dataset)

        if (self.dataset.x_validate.shape[0] is not 0):
            validation = (self.dataset.x_validate, self.dataset.y_validate)
        else:
            validation = None


        self.fit_history = self.model.fit(
              x=self.dataset.x_train,
              y=self.dataset.y_train,
              batch_size=self.batch_size,
              validation_data=validation,
              epochs=self.epochs,
              verbose = self.verbose,
              callbacks=callback)
        return self.fit_history
    

    def plot_prediction(self, index_test):
        """
        This function plot the prediction for the index 'index_test'
            if the prediction is correct, the name of the predicted image will be writte, in blue color
            otherwise, it will be written in red.
            The argument 'index_test' must be a integer
            
            Before calling this function, make sure the model is trained and evaluate! if it is not the case
            an exception will be raisen. 
        
        """
        if type(index_test) is not int:
            raise ValueError(f"""Error '{index_test}' is not an integer, aborting...""")
        try:
            self.dataset.x_predict = self.dataset.x_predict.reshape(-1,28,28) 
            label                  = int(index_test)
            dataset                = self.dataset
            name                   = dataset.y_predict[label]
            name_predicted         = self.prediction[label].argmax()
        except IndexError:
            raise 
        

        #Drawing the two images
        plt.figure(figsize=(6,6))

        #The image of test
        plt.subplot(1,2,1)
        plt.imshow(dataset.x_predict[label])
        plt.title(dataset.class_names[name])

        #Predicted image
        plt.subplot(1,2,2)
        plt.imshow(dataset.x_predict[label])

        if dataset.class_names[name_predicted] == dataset.class_names[name]:
            plt.title(dataset.class_names[name_predicted],c='blue')
        else:
            plt.title(dataset.class_names[name_predicted],c='red')
        #Drawing in the subplot
        plt.show()


    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self,mod):
        if type(mod) is not type(Sequential()):
            self.__raise_incompatible_type(mod, keras.Sequential())

        mod = Sequential(mod)
        self.__model    = mod

        self.dataset.x_predict = self.dataset.x_predict.reshape(-1,28,28,1) 
        self.prediction = self.__model.predict(self.dataset.x_predict)

