#!/usr/python3
# -*- coding: utf-8 -*-
from tensorflow import keras
import numpy as np
from external.rop import read_only_properties
"""
Principals data are managed in this Module 
"""


@read_only_properties('x','y','x_test', 'y_test','x_validate','y_validate','x_train','_y_train')
class Dtset:
    """
        This a class that attribute is readonly. 
        It's allow to get and manage the dataset from tensorflow.keras.datasets

        [The readOnly features is provided by the package rop from 'pip'.]  
    """
    def __init__(self,validate_size=10000):
        (__x, __y),(self.x_test, self.y_test) = self.load_data()
        self.class_names = self.__class_names()

        if validate_size<=0:
            raise IndexError(f'validate_size cannot be negative here {validate_size} < 0')
        if validate_size > 30000:
            raise IndexError(f'validate_size is too big thant train data size'+
                ' here validate size = {validate_size} > {60000-validate_size}')

        self.x_validate = __x[:validate_size]
        self.y_validate = __y[:validate_size]

        self.x_train    = __x[validate_size:]
        self._y_train    = __y[validate_size:]


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

    
    def __class_names(self, lang='fr'):
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
        class_names_ = self.__class_names(lang)
            
        return (dataset, class_names_)

    def data_Normalize(self, dataset):
        """
        Divide and return the given dataset (as numpy array) by 255.0
        """
        if type(dataset) is not np.ndarray:
            raise ValueError("The dataset's type must be a 'numpy.ndarray'!")
        return dataset/255.0