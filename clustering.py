import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.cluster import KMeans
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

reading = open("records.txt", "r")

dataset = []

line = reading.readline()
while line:
    row = line.split()
    dataset.append(row)
    line = reading.readline()
reading.close()

learn = np.asarray(dataset)

kmeans = KMeans(n_clusters=4)
kmeans.fit(learn)

print(kmeans.cluster_centers_)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

syn_num = learn[:,0]
syn_num = [float(i) for i in syn_num]
filter_num = learn[:,1]
filter_num = [float(i) for i in filter_num]
time = learn[:,2]
time = [float(i) for i in time]

ax.scatter(syn_num, filter_num, time, c=kmeans.labels_, cmap='rainbow')
ax.scatter(kmeans.cluster_centers_[:,0],
           kmeans.cluster_centers_[:,1],
           kmeans.cluster_centers_[:,2], color='black')
ax.set_xlabel('syn_num')
ax.set_ylabel('filter_num')
ax.set_zlabel('time')
plt.show()


learn = {'syn_num' : syn_num, 'filter_num' : filter_num, 'time' : time}
learn = pd.DataFrame(learn, columns=['syn_num', 'filter_num', 'time'])

X = learn[['syn_num','filter_num']]
Y = learn['time']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

new_syn_num = 11
new_filter_num = 15
print ('Time predicted: \n', regr.predict([[new_syn_num, new_filter_num]]))
