# ---------------------------------------------------------- #
# ---------------------- RMIT Vietnam ---------------------- #
# -------------------- Semester B, 2018 -------------------- #
#
# ----- ISYS2090 - File Structures and Database System ----- #
# ------------- Lecturer: Dr. Vladimir Mariano ------------- #
#
# ---------------------- Assignment 3 ---------------------- #
#
# --------- Author: Nguyen Hoang Minh - s3634696
#                   Vo Quoc Vu - s3575819
#                   Ho Minh Tri - s3594986
#                   Le Viet Hoang Dung - s3568452  --------- #
# ---------------------------------------------------------- #

#
# --------------------------- #
# --- Importing libraries --- #
# --------------------------- #
#
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.cluster import KMeans
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

#
# --------------------------------- #
# --- Open the file records.txt --- #
# --------------------------------- #
#
# 'records.txt' contains the information related to the searches. Each line of that file has 3 numbers:
# - The number of synonym(s) per each word that the search engine find for its search string
# - The filter number of top results to be returned
# - The amount of time it takes for each search
reading = open("records.txt", "r")

# This list is the dataset that will be trained and used for prediction later
dataset = []

# Read the file 'recoreds.txt' line by line
line = reading.readline()
while line:
    # Save each line as a new element of the dataset list
    row = line.split()
    dataset.append(row)
    line = reading.readline()
# Close the file when finish reading
reading.close()

# Convert the dataset from list to np_array
learn = np.asarray(dataset)

# Create and assign the Machine Learning tool
# This tool uses K-Means Clustering which group the data into clusters
# 4 clusters will be created
kmeans = KMeans(n_clusters=4)

# Cluster the data
kmeans.fit(learn)

# Print out the centroid of the clusters
print(kmeans.cluster_centers_)

# Create a 3d graph to represent the clusters and the datapoints
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# All the numbers of synonyms
syn_num = learn[:,0]
syn_num = [float(i) for i in syn_num]

# All the filter numbers
filter_num = learn[:,1]
filter_num = [float(i) for i in filter_num]

# All the running time
time = learn[:,2]
time = [float(i) for i in time]

# Generate the graph with the 3 list above as 3 dimensions
ax.scatter(syn_num, filter_num, time, c=kmeans.labels_, cmap='rainbow')

# Include in the graph the centroids which are put in black
ax.scatter(kmeans.cluster_centers_[:,0],
           kmeans.cluster_centers_[:,1],
           kmeans.cluster_centers_[:,2], color='black')

# Put the names for the dimensions
ax.set_xlabel('syn_num')
ax.set_ylabel('filter_num')
ax.set_zlabel('time')

# Save the drawing in an image file
plt.savefig('clusters.png')

# Convert the dataset into dataframe with the columns names
learn = {'syn_num' : syn_num, 'filter_num' : filter_num, 'time' : time}
learn = pd.DataFrame(learn, columns=['syn_num', 'filter_num', 'time'])

# Create the datasets used for prediction
X = learn[['syn_num','filter_num']]
Y = learn['time']

# Generate the tool to suppoert Linear Regression
regr = linear_model.LinearRegression()

# Apply Linear Regression onto the dataset
regr.fit(X, Y)

# Try to make a prediction using given number of synonyms and filter number
new_syn_num = 11
new_filter_num = 15
print ('Time predicted: \n', regr.predict([[new_syn_num, new_filter_num]]))
