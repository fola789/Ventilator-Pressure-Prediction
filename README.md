# Google Brain - Ventilator Pressure Prediction
The development a deep learning model in order to simulate a ventilator connected to a sedated patient's lung and predict the pressure of ventilators given some parameters and a control variable. The [competition](https://www.kaggle.com/c/ventilator-pressure-prediction/overview/description) was organized by Princeton University and & Google Brain at the Kaggle platform.

![Ventilator](images/header.png)
# Summary
What do doctors do when a patient has trouble breathing? They use a ventilator to pump oxygen into a sedated patient's lungs via a tube in the windpipe. But mechanical ventilation is a clinician-intensive procedure, a limitation that was prominently on display during the early days of the COVID-19 pandemic. At the same time, developing new methods for controlling mechanical ventilators is prohibitively expensive, even before reaching clinical trials. High-quality simulators could reduce this barrier.

Current simulators are trained as an ensemble, where each model simulates a single lung setting. However, lungs and their attributes form a continuous space, so a parametric approach must be explored that would consider the differences in patient lungs.

Neural networks and deep learning can better generalize across lungs with varying characteristics than the current industry standard of PID controllers.


![Ventilator](https://github.com/fola789/Ventilator-Pressure-Prediction/blob/main/images/header.png)

# Dataset
Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### 1. Files
+ train.csv - The training set
+ test.csv - The test set
+ sample_submission.csv - A sample submission file in the correct format
### 2. Columns
+ id - globally-unique time step identifier across an entire file
+ breath_id - globally-unique time step for breaths
+ R - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change R by changing the diameter of the straw, with higher R being harder to blow.
+ C - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change C by changing the thickness of the balloon’s latex, with higher C having thinner latex and easier to blow.
+ time_step - the actual time stamp.
+ u_in - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
+ u_out - the control input for the exploratory solenoid valve. Either 0 or 1.
+ pressure - the airway pressure measured in the respiratory circuit, measured in cmH2O

![Ventilator](https://github.com/fola789/Ventilator-Pressure-Prediction/blob/main/images/breathSamples.png)
# Data Summary
From the data we can see that we are provided with the following:
+ 75,450 non-contiguous cycles in the Training-Set
+ 50,300 non-contiguous cycles in the Test-Set
+ Three different compliances(C) of [10,20,50] mL cm H<sub>2</sub>O
+ Three different resitance(R) of[5,20,50] cm H<sub>2</sub>O./L/s
+ Every breath id have 80 unique timestamps.

# Light GBM
### 1. What is LightGBM and how does it work?
Light GBM is a gradient boosting framework that uses tree based learning algorithm.

Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm.
### 2. Advantages of LightGBM
1. **Faster training speed and higher efficiency:** Light GBM use histogram based algorithm i.e it buckets continuous feature values into discrete bins which fasten the training procedure.
2. **Lower memory usage:** Replaces continuous values to discrete bins which result in lower memory usage.
3.**Better accuracy than any other boosting algorithm:** It produces much more complex trees by following leaf wise split approach rather than a level-wise approach which is the main factor in achieving higher accuracy. However, it can sometimes lead to overfitting which can be avoided by setting the max_depth parameter.
4.**Compatibility with Large Datasets:** It is capable of performing equally good with large datasets with a significant reduction in training time as compared to XGBOOST.
5.**Parallel learning supported.**

### 3. LightGBM Intstallation
From Anaconda prompt type the folloiwing command to install in the desired environment.

    conda install -c conda-forge lightgbm

# Results

 ![Results](https://github.com/fola789/Ventilator-Pressure-Prediction/blob/main/images/results.png)
