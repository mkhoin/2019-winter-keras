# Wholesale Customer Segmentation with K-Means Clustering
# December 23, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering, KMeans


    ####################
    # DATABASE SETTING #
    ####################


# read raw dataset and print the first five data.
data = pd.read_csv('dataset/wholesale.csv')
print(data.head())


# scale the raw dataset and print the first five data.
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns = data.columns)
print(data_scaled.head())


# show dendrogram with threshold line.
# the number of clusters is the number of vertical lines cut.
plt.figure(figsize = (10, 7))  
plt.title("Dendrograms")  
shc.dendrogram(shc.linkage(data_scaled, method = 'ward'))
plt.axhline(y = 6, color = 'r', linestyle = '--')
plt.show()


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# perform k-means clustering and print cluster info.
cluster = KMeans(n_clusters = 4)  
print(cluster.fit_predict(data_scaled))


# scatter plot using the clustering info.
# pick the two axis based on importance. 
plt.figure(figsize = (10, 7))  
plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c = cluster.labels_) 
plt.show()
