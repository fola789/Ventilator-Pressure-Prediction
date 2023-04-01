import pandas as pd # For Data Manipulation of Numerical Yables and Time Series 
import numpy as np # For Mathematical operations on arrays
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import normalize, PowerTransformer # Scale input vectors individually to unit norm (vector length).
import lightgbm as lgb # For Distributed gradient boosting framework (XDBoost possible alternative)
import warnings

if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please ensure you have installed TensorFlow correctly')
else:   
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))