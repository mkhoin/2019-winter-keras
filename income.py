# Income Classification with Neural Network
# December 23, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


    ####################
    # DATABASE SETTING #
    ####################


# load datasets and print related info
df = pd.read_csv('dataset/income.csv', header = 0)
print(df.head(5))
print(df.describe())
print(df.dtypes)

print('\nNumber of raw data:', len(df))
high = len(df.loc[df['income'] == ' >50K'])
print('Number of high income data:', high)
print('Number of low income data:', len(df) - high)


# split the DB into training and test sets
df_train, df_test = train_test_split(df, test_size = 0.3, random_state = 5)
print('\nNumber of training data:', df_train.shape[0])
print('Number of test data:', df_test.shape[0])


# prepare income column for classification
# income is 1 if higher than 50k, 0 if lower.
df_train.loc[df_train['income'] == ' >50K'] = 1
df_train.loc[df_train['income'] == ' <=50K'] = 0
df_test.loc[df_test['income'] == ' >50K'] = 1
df_test.loc[df_test['income'] == ' <=50K'] = 0


# prepare training and test datasets
X_train = df_train.loc[:, ~(df_train.columns == 'income')].values
X_test = df_test.loc[:, ~(df_test.columns == 'income')].values
Y_train = df_train['income'].values
Y_test = df_test['income'].values


# scale the variables to help train the neural network
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# Add a first Dense (fully-connected) layer. 
# we always need to supply the input shape of data in the first layer.
# dropout can prevent overfitting and should speed up the training.
model = Sequential()
model.add(Dense(10, input_shape = [X_train.shape[1]], activation = 'relu'))
model.add(Dropout(0.5))


# More layers to allow the network to learn more complex relationships
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.5))


# sigmoid is suited for binary classification.
# softmax for multiple categorical classification.
model.add(Dense(1, activation = 'sigmoid'))


# adam works well in general.
# for this classification task, binary crossentropy makes sense.
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()


# this is where the training starts. training data is loaded here.
# batch size means how many data are loaded in one iteration.
# epochs are the number of training loops. 
# we validate directly on test data. 
# can cause overfitting but is the fastest way for good results.
model.fit(X_train, Y_train, batch_size = 300, epochs = 5, verbose = 1, 
        validation_data = [X_test, Y_test])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
        metrics = ['accuracy'])


    ####################
    # MODEL EVALUATION #
    ####################


# we calculate the accuracy of our model using the test set.
score = model.evaluate(X_test, Y_test, verbose = 1)
print('\nKeras DNN model accuracy = ', score[1])


# display confusion matrix
# the second line converts [0, 1] into true/false
pred = model.predict(X_test)
pred = (pred > 0.5)
print('\n=== CONFUSION MATRIX ===')
print(confusion_matrix(Y_test, pred))


# calculate F1 score using confusion matrix.
print("\nF1 score:", f1_score(Y_test, pred, average = 'micro'))
