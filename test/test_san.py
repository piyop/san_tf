# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 22:27:15 2020

@author: 005869
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), r'models')))

import unittest
import numpy as np

from tensorflow.keras.utils import plot_model
from tensorflow import keras as kl
import tensorflow as tf

import san_tf as san

# n=10
# images = np.array([[[[x*n+y+1] for y in range(n)] for x in range(n)]])
# tf.image.extract_patches(images, [1,3,3,1], [1,1,1,1], 1, padding)

# class test_funcs(unittest.TestCase):
    
#     def test_view_as_blocks(self):

# if __name__ == '__main__':
#     test = test_funcs()

class test_funcs(unittest.TestCase):
    
    def test_san(self):
        input_shape = (224, 224, 3)
        model = san.san(input_shape,
                    sa_type=1, 
                      layers=(3, 4, 6, 8, 3), 
                      kernels=[3, 7, 7, 7, 7], 
                      num_classes=1000)
        plot_model(model, to_file='model_shapes.png', show_shapes=True)
        
        y = model(np.random.rand(4, 224, 224, 3).astype(np.float32))
        print(y.shape)
    
    
    
if __name__ == '__main__':
    test = test_funcs()
    test.test_san()
    
    # unittest.main()