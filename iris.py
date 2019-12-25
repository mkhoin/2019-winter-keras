# Iris Classification with Neural Network
# December 23, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages.
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


    ####################
    # DATABASE SETTING #
    ####################


# load iris dataset.
iris = load_iris() 
print('\n === SHAPE INFO ===')
print('Data:', iris.data.shape)
print('Label:', iris.target.shape)

print('\nFirst five data: ')
print(iris.data[:5])
print('\nFirst five labels: ')
print(iris.target[:5])


# reshaping label data.
x = iris.data
y = iris.target.reshape(-1, 1) 
print('\nReshaped label dataset:', y.shape)


# one-hot encode the labels.
# then scale the labels.
encoder = OneHotEncoder(sparse = False)
y_scaled = encoder.fit_transform(y)


# Split the data for training and testing.
X_train, X_test, Y_train, Y_test = train_test_split(x, y_scaled, test_size = 0.20)


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# build the model.
model = Sequential()
model.add(Dense(10, input_shape = (4,), activation = 'relu'))
model.add(Dense(10, activation = 'relu'))


# produces 3 probability values for each category.
# they are compared with one-hot encoding labels.
model.add(Dense(3, activation = 'softmax'))


# adam optimizer with learning rate of 0.001.
opt = Adam(lr = 0.001)
model.compile(opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())


# train the model.
model.fit(X_train, Y_train, batch_size = 5, epochs = 30, verbose = 1)


    ####################
    # MODEL EVALUATION #
    ####################


# test on unseen data.
results = model.evaluate(X_test, Y_test, verbose = 1)
print('Final test set accuracy: {:4f}'.format(results[1]))
