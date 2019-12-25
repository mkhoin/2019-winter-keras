# Car Price Prediction with Linear Regression
# December 23, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


    ####################
    # DATABASE SETTING #
    ####################


# read the raw dataset
# 205 data in the raw dataset
print('\n === READING RAW DATABASE ===')
df = pd.read_csv("dataset/car-price.csv", header = 0)
print(df.head(5))
print(df.describe())
print(df.dtypes)


# converting price column from object to int
# it is object-type because of "?"
# dropna() only works with NaN, so the need to replace
# 4 data removed
print('\n === DELETING ? FROM PRICE COLUMN ===')
df["price"].replace('?', np.nan, inplace = True)
df.dropna(subset = ["price"], axis = 0, inplace = True)
df["price"] = df["price"].astype("int")
print(df.describe())


# converting horsepower column from object to int
# it is object-type because of "?"
# dropna() only works with NaN, so the need to replace
# 2 data removed
print('\n === DELETING ? FROM HORSEPOWER COLUMN ===')
df["horsepower"].replace('?',np.nan, inplace = True)
df.dropna(subset = ["horsepower"], axis = 0, inplace = True)
df["horsepower"] = df["horsepower"].astype("int")
print(df.describe())


# plotting positive relation between engine size vs. price
sns.regplot(x = 'engine-size', y = 'price', data = df)
plt.title("Scatterplot of Engine Size vs Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.ylim(0,)
plt.show()


# plotting positive relation between highway-mpg vs. price
sns.regplot(x = 'highway-mpg', y = 'price', data = df)
plt.title("Scatterplot of highway-mpg vs price")
plt.xlabel("highway-mpg")
plt.ylabel("price")
plt.ylim(0,)
plt.show()


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# linear regression with 4 inputs
lm = LinearRegression()
X = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
Y = df["price"]
lm.fit(X, Y)
pred = lm.predict(X)


    ####################
    # MODEL EVALUATION #
    ####################


# print related metrics
print('\n === RELATED METRICS ===')
print('Score: ', lm.score(X, Y))
print('Co-efficients: ', lm.coef_)


# plotting the result
sns.distplot(df["price"], hist = False, color = "r", label = "Actual Value")
sns.distplot(pred, hist = False, color = "b", label = "Fitted Value")
plt.xlim(0,)
plt.ylim(0,)
plt.gca().axes.get_yaxis().set_visible(False)
plt.show()

