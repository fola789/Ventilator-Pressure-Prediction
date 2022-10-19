import pandas as pd # For Data Manipulation of Numerical Yables and Time Series 
import numpy as np # For Mathematical operations on arrays
import sklearn
import warnings
import matplotlib as plt
from IPython.display import display
from sklearn.model_selection import train_test_split, GroupKFold, KFold # Split arrays or matrices into random train and test subsets, K-fold iterator variant with non-overlapping groups, K-fold iterator variant with non-overlapping groups.
from sklearn.metrics import mean_absolute_error # Mean absolute error regression loss.
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import normalize, PowerTransformer # Scale input vectors individually to unit norm (vector length).
#Potential ADDONS 
import optuna #  Automatic hyperparameter optimization software
import lightgbm as lgb # For Distributed gradient boosting framework (XDBoost possible alternative)

if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please ensure you have installed TensorFlow correctly')
else:   
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

train = pd.read_csv('train.csv') # Read a comma-separated values (csv) file into Train DataFrame
test = pd.read_csv('test.csv') # Read a comma-separated values (csv) file into Test DataFrame
submission = pd.read_csv('sample_submission.csv') # Read a comma-separated values (csv) file into Prediction DataFrame
display(train)
display(test)
display(submission)

# All the breath id have same counts i.e. 80.
# Every breath id have unique timestamps. Every breath id contain 80 different timestamps

#To determine the number of distinct values for each feature we can use the following
#To which we further find the following: 
# • 75,450 non-contiguous cycles in Training-Set
# • 50,300 non-contiguous cycles in Test-Set
# • Three different compliances(C) of [10,20,50] mL cm H2O
# • Three different resitance(R) of[5,20,50] cm H2O/L/s
train.nunique().to_frame()
test.nunique().to_frame()
train.R.value_counts().to_frame()
test.C.value_counts().to_frame()

#Data validation in training and test set before any required data cleaning 
train.isnull().sum(axis = 0).to_frame()
test.isnull().sum(axis = 0).to_frame()

breath_one = train.query('breath_id == 1').reset_index(drop = True)
display(breath_one)

breath_one.nunique().to_frame()

breath_one.plot(x="time_step", y="u_in", kind='line',figsize=(12,3), lw=2, title="u_in")
breath_one.plot(x="time_step", y="u_out", kind='line',figsize=(12,3), lw=2, title="u_out")
breath_one.plot(x="time_step", y="pressure", kind='line',figsize=(12,3), lw=2, title="pressure")

train['u_in_cumsum'] = (train['u_in']).groupby(train['breath_id']).cumsum() # get the training by storing theCumulitive sum of percentage the inspiratory solenoid valve by breath id
test['u_in_cumsum'] = (test['u_in']).groupby(test['breath_id']).cumsum()# get the test by storing theCumulitive sum of percentage the inspiratory solenoid valve by breath id

train['u_in_lag'] = train['u_in'].shift(2).fillna(0)# First time_step is always 0.000000 so we shift down 1 row and replace all NaN elements with Replace all NaN elements with 0s.
test['u_in_lag'] = test['u_in'].shift(2).fillna(0)# First u_in is always 0, as the valve is completly closed with no air being let in.

y = train[['pressure']].to_numpy().reshape(-1, 80) # We want the the airway pressure measured in the respiratory circuit, measured in cmH2O and 80 being the total number of step for a full breath. , since we have 80 features, we put -1 to let numpy to infer how many samples are there.

train.drop(['id','breath_id','u_out','pressure',], axis=1, inplace=True) # Drop the following columns using axis=1 and inplace set to true  so that the original data can be modified without creating a copy. *better practice would be to use chaining instead of true inplace
test = test.drop(['id', 'breath_id', 'u_out'], axis=1) # Drop the following columns again

pt = PowerTransformer() # Creates a new instance of the class and assigns this object to the local variable pt.

pt.fit(train) #  parametric, monotonic transformations are applied to make data more Gaussian-like. This is useful for modeling issues related to heteroscedasticity (non-constant variance), or other situations where normality is desired.Fits the training set to transformer
train2 = pt.transform(train) # Apply the power transform to each feature in train dataset using the fitted lambdas.
test2 = pt.transform(test) # Apply the power transform to each feature in test dataset using the fitted lambdas.

train3 = train2.reshape(75450, 80, 6) # 75,450 non-contiguous cycles (each cycle is uniquely labelled with an individual breath_id),  80 time steps of data per breath
test3 = test2.reshape(50300, 80, 6) # 50,300 breaths in the test dataset,  80 time steps of data per breath

kf = KFold(n_splits=5, shuffle=True, random_state=2022) # Split dataset into k consecutive folds
test_preds = []
# scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 200*((75450)*0.8)/1024, 1e-5) # useful to lower the learning rate as the training progresses so an exponential decay function is applied to optimizer step
scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 200*((len(train)*0.8)/1024), 1e-5)

for fold, (train_idx, test_idx) in enumerate(kf.split(train3, y)):
    
    X_train, X_valid = train3[train_idx], train3[test_idx]
    y_train, y_valid = y[train_idx], y[test_idx]
    
    model = keras.models.Sequential([
        keras.layers.Input(shape=(80, 6)),
        keras.layers.Bidirectional(keras.layers.LSTM(200, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(150, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True)),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(scheduler), loss="mae") # alternative loss "mae", although SGD generalizes better than Adam Adam converges faster which is required in ventilator usecase
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=200, batch_size = 1024)
    model.save(f'Fold{fold+1} RNN Weights')
    test_preds.append(model.predict(test.to_numpy().reshape(50300, 80, 6)).squeeze().reshape(-1, 1).squeeze())
    
submission["pressure"] = sum(test_preds)/5
submission.to_csv('submission.csv', index=False)


