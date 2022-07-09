"""
Implement random forest
"""
from collections import Counter

import numpy as np

from decision_tree import DecisionTree


class RandomForest:
	def __init__(self, n_trees, r, max_dep=50, min_ele=2) -> None:
		"""
		n_trees is the number of trees to train, it is also the number of data to sample
		r is the subset of features to train
		"""
		self.n_trees = n_trees 
		self.r = r 
		self.max_dep = max_dep 
		self.min_ele = min_ele
		self.trees = []

	def fit(self, X, y):
		#n_example, n_feature = X.shape
		for _ in range(self.n_trees):
			## Sample n_example data
			X_sample, y_sample = self.sample_bootstrap(X, y)

			tree = DecisionTree(self.max_dep, self.min_ele, num_features=self.r)
			tree.fit(X_sample, y_sample)

			self.trees.append(tree)

	def predict(self, X):
		tree_predicts = np.array([tree.predict(X) for tree in self.trees])
		tree_predicts = tree_predicts.transpose()
		## Get the majority vote
		mojority = [self._most_common_label(votes) for votes in tree_predicts]
		return np.array(mojority)

	def _most_common_label(self, Y):
		counter = Counter(Y)
		return counter.most_common(1)[0][0]

	def sample_bootstrap(self,X,y):
		"""
		Generate samples with bootstrap.
		Samples are generated with replacement. For example, the shape of X
		is m, n = X.shape. (Each row of X is an example). Then sample m rows from
		these m rows (with replacement)
		"""
		num_sample = X.shape[0]
		indexes = np.random.choice(num_sample,num_sample, replace=True)
		return X[indexes], y[indexes]

if __name__ == '__main__':
	# rain = [1,1,1,0,0,0,0,0,0,0]
	# time = [30, 15, 5, 10, 5, 15, 20, 25, 30, 30]
	# X = np.array([rain, time]).transpose()
	# y = np.array([0,0,0,0,0,1,1,1,1,1])
	# r = 2 #math.floor(math.sqrt(X.shape[1]))
	# n_trees = 4
	# max_dep = 10
	# forest = RandomForest(n_trees=n_trees, r=r)
	# X_test = np.array([[1,1], [30, 15]]).transpose()
	# forest.fit(X,y)
	# y_hat = forest.predict(X_test)
	# print(y_hat)
	import math
	from sklearn import datasets
	from sklearn.model_selection import train_test_split	

	def accuracy(y, y_hat):
		acc = np.sum(y == y_hat) / len(y)
		return acc

	data_iris = datasets.load_iris()
	X = data_iris.data
	y = data_iris.target
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
	print(f'Size of training set: {X_train.shape[0]}, Size of test set: {X_test.shape[0]}')
	r = math.floor(math.sqrt(X.shape[1]))
	num_trees = 10
	forest = RandomForest(n_trees=num_trees, r=r)
	forest.fit(X_train, y_train)
	y_hat = forest.predict(X_test)
	acc = accuracy(y_test, y_hat)
	print(f'Accuracy: {acc}')

