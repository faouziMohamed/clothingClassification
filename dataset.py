#!/usr/python3
# -*- coding: utf-8 -*-
from tensorflow import keras
import numpy as np
#from external.rop import read_only_properties
"""
Principals data are managed in this Module 
"""


#@read_only_properties('x','y','x_test', 'y_test','x_validate','y_validate','x_train','y_train')
class Dtset:
    """
        This a class that attribute is readonly. 
        It's allow to get and manage the dataset from tensorflow.keras.datasets

        [The readOnly features is provided by the package rop from 'pip'.]  
    """
    def __init__(self,validate_size=15000,predict_size=1500):
        (__x_tr, __y_tr),(__x_ts, __y_ts) = self.load_data()
        self.class_names = self.classNames()
        #check there is no probleme with value
        self.__raise_error_data(value=predict_size,data='test')
        self.__raise_error_data(value=predict_size,data='predict')

        __x_tr  = self.data_Normalize(__x_tr)
        __x_ts  = self.data_Normalize(__x_ts)


        #Spliting the datato create different value for our model
        ## Value for predict and test data
        self.x_predict  = __x_ts[:predict_size]
        self.y_predict  = __y_ts[:predict_size]
        self.x_test     = __x_ts[predict_size:]
        self.y_test     = __y_ts[predict_size:]

        #Value for validate and train
        self.x_validate = __x_tr[:validate_size]
        self.y_validate = __y_tr[:validate_size]

        self.x_train    = __x_tr[validate_size:]
        self.y_train    = __y_tr[validate_size:]

    def __raise_error_data(self,value,data='test'):
        """
        Raise exception if value given in parameter is not correct for validate data or test data
        """
        #Check if value is correct
        type_value = str(type(value)).split('\'')[1]
        if data is 'test':
            if type(value) is not int:
                raise TypeError(f'The value size must be a integer! not a {type_value}')
       
       #check given predict size is correct
        if value < 1:
            raise IndexError(f'validate_size cannot be negative nor null here {value} < 0')
        if value > 5000:
            raise IndexError(f'predict_size is too big thant test data size'+
            ' here test size = {value} > {10000-value}')

        #check given validate size is correct
        elif data is 'predict':
            if value < 1:
                raise IndexError(f'validate_size cannot be negative nor null here{value} < 0')
            if value > 30000:  
                raise IndexError(f'validate_size is too big thant train data size'+
                ' here validate size = {value} > {60000-value}')

    def load_data(self):
        """
        Load the fashion mnist dataset  from tensorflow.keras.datasets.fashion_mnist
        
        USAGE :
            (train_images, train_labels), (test_images, test_labels) = load_data(lang='fr')
            
        This function return  the mnist dataset.
            The fashion mnist dataset contains two tuples:
            (train_images, train_labels), (test_images, test_labels)   
        """
        fashion_mnist = keras.datasets.fashion_mnist
        return fashion_mnist.load_data()
    
    def classNames(self, lang='fr'):
        """
        Return the mnist dataset class names in the lang choosen between fr(French) and en(English)   
        """
        if lang is 'en':
            return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        return ['T-shirt/Haut', 'Pantalon', 'Tricot', 'Robe', 'Manteau',
               'Sandale', 'Chemise', 'Basket', 'Sac', 'Bottine']

    def load_fashion_mnist(self, lang='fr'):
        """
        Load the fashion mnist dataset  from tensorflow.keras.datasets.fashion_mnist
        
        USAGE :
            dataset,class_names = load_fashion_mnist(lang='fr')
            
        This function return  two tuples:
        1 - the mnist dataset
            The fashion mnist dataset contains two tuples:
            (train_images, train_labels), (test_images, test_labels)
            
        2 - The class name (by default it in french)    
            if lang = 'fr' the class name will be in French,
            if it equal 'en' then it will be in English    
        """
        dataset = self.load_data(lang)
        class_names_ = self.classNames(lang)
            
        return (dataset, class_names_)

    def data_Normalize(self, dataset):
        """
        Divide and return the given dataset (as numpy array) by 255.0
        """
        if type(dataset) is not np.ndarray:
            raise TypeError("The dataset's type must be a 'numpy.ndarray'!")
        return dataset/255.0