from tensorflow.keras.models import Sequential
from tensorflow.keras        import callbacks
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from external.rop import read_only_properties
from tensorflow   import keras
from dataset      import Dtset


import tensorflow as tf
import matplotlib.pyplot as plt
import os, glob, re

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
                 metrics=[],verbose=1, dataset=Dtset(validate_size=15000, predict_size=1500),
                 try_load_latest_save=True, load_saved_model=None):
        
        
        self.load_latest       = try_load_latest_save
        self.load_saved_model  = load_saved_model
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
        self.dataset           = dataset
        self.batch_size        = 32
        self.epochs            = 1
        self.prediction        = None
        self.callback          = None

        self.__compiled        = False
        self.__fitted          = False
        self.__evaluated       = False
        self.__predicted       = False
        
        #trying to load some model before creating a new model
        load = False
        if load_saved_model is not None:
          if type(load_saved_model) is not str:
            raise TypeError(f"{load_saved_model} must be of type 'str'")
          load = self.__find_model_saved(load_saved_model)
        elif try_load_latest_save is True:
          load = self.__find_model_saved('')
        
        if load is False:
          self.new_model()
          self.compile()
    


    def __find_model_saved(self,path_to_file=''):
      """
      Read model saved in local
      """

      # If user want to load a specific model saved on local
      if path_to_file is not '' and os.path.exists(path_to_file):
        if os.path.isfile(path_to_file):
          self.model = path_to_file
          return True
        else :
          raise ResourceWarning(f"'{path_to_file}' is not a file, it's actually a directory! ...Aborting")
      elif path_to_file is not '' and not os.path.exists(path_to_file):
          raise ResourceWarning(f"'{path_to_file}' is missing, removed or the name is incorrect")
      
      # If the user want to load latest saved model
      if self.load_latest is True:
        list_file = glob.glob('*.h5')
        list_file.sort(reverse=True)
        if len(list_file) is 0:
          return False
        else:  
          self.model = list_file[0]
          return True


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
        self.__model.add(Dropout(0.1,name='Dropout_3'))
        
        
        self.__model.add(Flatten(name='Flatten_Layer'))
        self.__model.add(Dense(units=self.hidden_nn,
                                activation = self.hidden_activation,name='1st_dense_layer'))
        """self.__model.add(Dense(units=64,
                                activation = self.hidden_activation,name='2nd_dense_layer'))"""
        self.__model.add(Dense(10,activation = self.output_activation,name='3rd_dense_layer'))
        return self.__model
    
    def compile(self):
        self.__model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics)
        self.__compiled = True

    def __raise_incompatible_type(self,a,b):
        type_a = str(type(a)).split('\'')[1]
        type_b = str(type(b)).split('\'')[1]
        raise TypeError(f"{type_a} object cannot be converted to '{type_b}' because '{type_a}' and '{type_b}' are not compatible")

    def __reshape_add_one(self, dataset):
        """
        Reshape numpy.ndarray from the shape(a,b,c) to shape(a,b,c,1)
        """

        if dataset is None:
          dataset = self.dataset
          
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

    def __reshape_one(self, data):
        return data.reshape(-1,28,28,1)

    def __reshape_no_one(self,data):
        return data.reshape(-1,28,28)

    def __check_data_and_parameters(self,**kwargs):
        """
        Check if every data structure is correct before configuring the model to be fitted
        """
        if type(kwargs['dataset']) is not Dtset:
            if  kwargs['dataset'] is None:
                pass
            else:
                self.__raise_incompatible_type(kwargs['dataset'], Dtset())

        if (type(kwargs['model']) is not self) and (type(kwargs['model']) is not type(None)) :
            self.__raise_incompatible_type(kwargs['model'],self)
    
    def __create_callback(self,path_to_file='saved_model'):
        """
        Create a callback that will be during the training step to save the best configuration
        """
        list_file = glob.glob('saved_model/*.h5')
        list_file.sort(reverse=True)

        if len(list_file) is 0:
          file_path=f'{path_to_file}/save_1.h5'
        else:
          i = int(re.findall('\d+',list_file[0])[0])
          if i is 0:
            file_path = f'{path_to_file}/save_{1}.h5'
          else:
            file_path = f'{path_to_file}/save_{i+1}.h5'

        self.callback = callbacks.ModelCheckpoint(filepath=file_path,
                                              verbose=0,
                                              monitor='val_accuracy',
                                              save_best_only=True)
        return self.callback

    def __config_model(self,**kwargs):
        """
        Choosing wich value will be used during the training step and prepare the values 
        to be setted to the fit function 
        """
        #Values Check
        if (kwargs['batch_size'] is not None) and (kwargs['batch_size']<0):
            self.batch_size=None
        if kwargs["epochs"] < 1 :
            self.epochs = 1
        else:
            self.epochs = kwargs["epochs"]
            
        #Configuring the model
        if (kwargs['model'] is None) :
            if self.__model is None:
                self.new_model()
                self.compile()
            else:
                if self.__compiled is not True:
                    self.compile()
        else:
            self.model = kwargs['model']

        if kwargs['dataset'] is not None:
            self.dataset = kwargs['dataset']

    def fit(self, dataset=None,model=None, epochs=32, batch_size=None, callback=None):
        """
        Fit the model with the given dataset
        """
        #Exception check
        self.__check_data_and_parameters(dataset=dataset, model=model)
        #check value

        self.__config_model(batch_size=batch_size, epochs=epochs, model=model, dataset=dataset )
        
        #reshape the data by adding 1 on the 4th dimension
        (self.dataset.x_train,self.dataset.y_train),\
        (self.dataset.x_validate,self.dataset.y_validate),\
        (self.dataset.x_test,self.dataset.y_test) = self.__reshape_add_one(dataset=dataset)

        if (self.dataset.x_validate.shape[0] is not 0):
            validation = (self.dataset.x_validate, self.dataset.y_validate)
        else:
            validation = None
          
        if callback is not None:
          self.callback = callback
        else:
          self.__create_callback()


        self.fit_history = self.model.fit(
              x=self.dataset.x_train,
              y=self.dataset.y_train,
              batch_size=self.batch_size,
              validation_data=validation,
              epochs=self.epochs,
              verbose = self.verbose,
              callbacks=self.callback)
        self.__fitted = True
        return self.fit_history
    
    def evaluate(self):
        if self.__fitted is not True:
            raise RuntimeError(f'You must train the model before to evaluate it')
        evaluation = self.__model.evaluate(self.dataset.x_test,self.dataset.y_test)
        self.__evaluated = True;
        return evaluation

    def predict(self):
        self.dataset.x_predict = self.__reshape_one(self.dataset.x_predict)
        if self.__evaluated is not True:
            raise RuntimeError(f'You must train and evaluate the model before making prediction')
        prediction = self.__model.predict(self.dataset.x_predict)
        self.__predicted = True
        return prediction

    def plot_prediction(self, index_test):
        """
        This function plot the prediction for the index 'index_test'
            if the prediction is correct, the name of the predicted image will be writte, in blue color
            otherwise, it will be written in red.
            The argument 'index_test' must be a integer
            
            Before calling this function, make sure the model is trained and evaluate! if it is not the case
            an exception will be raisen. 
        
        """
        if self.__predicted is not True:
            if self.__fitted is not True:
                raise RuntimeError(f'Train and evaluate the model before make prediction')
            if self.__evaluated is not True:
                self.evaluate()

        self.dataset.x_predict = self.__reshape_one(self.dataset.x_predict) 
        self.prediction = self.__model.predict(self.dataset.x_predict)

        if type(index_test) is not int:
            raise ValueError(f"""Error '{index_test}' is not an integer, aborting...""")
        try:
            self.dataset.x_predict = self.__reshape_no_one(self.dataset.x_predict) 
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
            correct = True
            plt.title(dataset.class_names[name_predicted],c='blue')
        else:
            correct = False
            plt.title(dataset.class_names[name_predicted],c='red')
        #Drawing in the subplot
        plt.show()
        return correct


    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self,mod):
        if type(mod) is not type(Sequential()):
            self.__raise_incompatible_type(mod, keras.Sequential())

        #mod = Sequential(mod)
        self.__model   = mod
        self.loss      = mod.loss
        self.optimizer = mod.optimizer
        self.metrics   = mod.metrics
    @property
    def DIRNAME(self):
      return os.getcwd()







        #self.__fitted = True
        #self.__evaluated = True
        #self.dataset.x_predict = self.__reshape_one(self.dataset.x_predict) 
        #self.prediction = self.__model.predict(self.dataset.x_predict)

