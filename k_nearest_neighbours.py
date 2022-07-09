from collections import Counter

import numpy as np
import heapq

from helper_functions import euclidean_distance,accuracy, plot_scatter


class KNN:
	def __init__(self, k):
		self.k = k 
	
	def fit(self, X_train, y_train):
		## There is no training process for KNN. Just record all the training data.
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		## For each of the point in X, make a prediction using self._predict
		y_hat = [self._predict(x) for x in X_test]
		return np.array(y_hat)

	def _predict(self, x):
		distances = [(euclidean_distance(x, an_x), index) for index, an_x in enumerate(self.X_train)]

		## Store the distances in a heap. So it takes log time to find the
		## k smallest distances. Heapify can be done in linear time. 
		k_nearest_points = heapq.nsmallest(self.k, distances, key=lambda x: x[0])
		k_labels = [self.y_train[index] for _, index in k_nearest_points]
		counter = Counter(k_labels)
		y_hat = counter.most_common(1)[0][0]
		return np.array(y_hat)



if __name__ == '__main__':

	## Test 1
	X_train = np.array([[1,1],[2,1],[2,2],[1,2],[1,-1],[1,-2],[2,-2],[2,-1]])
	y_train = np.array([1,1,1,1,0,0,0,0])
	X_test = np.array([[1.5, 1.5],[1.5, -1.5]])
	y_test = np.array([1, 0])

	## Plot the training data.
	title = 'Training data'
	plot_scatter(X_train, y_train, fix_aspect=True,title=title)
	## Plot the training and testing data. Testing data in grey
	title = 'Training and testing data (testing data in black)'
	plot_scatter(X_train, y_train, X_test, fix_aspect=True, title=title)
	k = 3
	knn = KNN(k)
	knn.fit(X_train, y_train)
	y_hat = knn.predict(X_test)
	## Coloring test point based on predicted label
	title = 'Training and testing data (predicted color of testing data)'
	plot_scatter(X_train, y_train, X_test, y_hat,fix_aspect=True, title=title)
	## Coloring test point based on true label
	title = 'Training and testing data (true color of testing data)'
	plot_scatter(X_train, y_train, X_test, y_test,fix_aspect=True, title=title)

	## Test 2
	from sklearn import datasets
	from sklearn.model_selection import train_test_split

	data_iris = datasets.load_iris()
	X = data_iris.data
	y = data_iris.target
	X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=1234)
	## Plot the training data
	title = 'Training data'
	plot_scatter(X_train[:,0:2], y_train, title=title)
	## Plot the training and testing data, testing data in grey
	title = 'Training and testing data (testing data in black)'
	plot_scatter(X_train[:,0:2], y_train, X_test[:, 0:2], title=title)

	k = 3
	knn = KNN(k)
	knn.fit(X_train, y_train)
	y_hat = knn.predict(X_test)
	## Coloring test point based on predicted label
	title = 'Training and testing data (predicted color of testing data)'
	plot_scatter(X_train, y_train, X_test,  y_hat,title=title)
	## Coloring test point based on true label
	title = 'Training and testing data (true color of testing data)'
	plot_scatter(X_train, y_train, X_test,  y_test,title=title)
	acc = accuracy(y_test, y_hat)
	print(f'Accuracy: {acc*100}%')


	


