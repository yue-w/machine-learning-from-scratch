"""
Adaboost.
References:
1. http://www.cs.cornell.edu/courses/cs4780/2017sp/lectures/lecturenote19.html
2. https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/adaboost.py
3. https://youtu.be/LsK-xG1cLYA
"""
import copy
from logging import raiseExceptions
import math
import numpy as np
from decision_tree import DecisionTree, Node

#### Reuse the codes of DecisionTree
class Stump(DecisionTree):
	def __init__(self, max_depth=1, min_elements=2, num_features=None):
		super().__init__(max_depth, min_elements, num_features)
		self.flip = False

	def predict(self, X):
		## Overrite the predict function. If flip is True, flip prediction
		y_hat = [self._split(self.root, x) for x in X]
		y_hat = np.array(y_hat)
		if self.flip:
			y_hat = -y_hat

		return -np.array(y_hat)

## This will be used for exhaustive search. Write a stump.
class SimpleStump(Node):
	def __init__(self, feature=None, threshold=None, left=None, right=None, *, label=None):
		super().__init__(feature, threshold, left, right, label=label)
		self.flip = False

	def predict(self, X):
		n_sample = X.shape[0]
		X_column = X[:, self.feature]
		y_hat = np.ones(n_sample)
		y_hat[X_column < self.threshold] = -1
		if self.flip:
			y_hat = -y_hat
		return y_hat


class Adaboost:
	def __init__(self, max_boost=200, learnertype=None) -> None:
		"""
		max_boost is the maximum number of boost to do.
		learnertype is a class.
		"""
		self.weaklearner = learnertype
		self.max_boost = max_boost
		self.learners = []
		self.alphas = []

	def fit(self, X, y, method='resample'):
		if method == 'resample': ## Resample based on weight.
			num_example, num_feature = X.shape
			weights = np.full(shape=(num_example,1), fill_value=1/num_example)
			for _ in range(self.max_boost):
				h_tem = self.weaklearner()
				#h_tem = Stump()
				X_resample = self.sample_by_weight(X, weights)
				h_tem.fit(X_resample, y)

				y_hat = h_tem.predict(X)
				y_hat = y_hat.reshape((y_hat.shape[0], 1))
				epsilon = sum(weights[y != y_hat])
				## if epsilon > 0.5, flip prediction of stump
				if epsilon > 0.5:
					h_tem.flip = True
					y_hat = - y_hat
					epsilon = 1 - epsilon
				
				EPS = 1e-10 ## Avoid divide by 0 and log0
				alpha = 0.5 * np.log((1 - epsilon + EPS) / (epsilon + EPS))
				self.alphas.append(alpha)
				self.learners.append(h_tem)
				weights *= np.exp(-alpha * y_hat * y) 
				## Normalization
				weights /= sum(weights)
				assert math.isclose(sum(weights), 1.0)
		
		elif method == 'exhaustive':  # Try all splitting and weight cost by weight
			num_example, num_feature = X.shape
			weights = np.full(shape=(num_example,1), fill_value=1/num_example)
			for _ in range(self.max_boost):
				h_min = None #SimpleStump()
				min_err = float('inf')
            	# greedy search to find best threshold and feature
				for feature_i in range(num_feature):
					X_column = X[:, feature_i]
					thresholds = np.unique(X_column)
					for threshold in thresholds:
						h_tem = SimpleStump(feature=feature_i, threshold=threshold)
						y_hat = h_tem.predict(X)
						y_hat = y_hat.reshape((y_hat.shape[0], 1))
						run_err = sum(weights[y_hat != y]) 
						if run_err > 0.5:
							h_tem.flip = True
							run_err = 1 - run_err
						## record the min error so far	
						if  run_err < min_err:
							h_min = copy.deepcopy(h_tem)
							min_err = run_err
				EPS = 1e-10 ## Avoid divide by 0 and log0
				epsilon = min_err
				alpha = 0.5 * np.log((1 - epsilon + EPS) / (epsilon + EPS))
				self.alphas.append(alpha)
				self.learners.append(h_min)
				y_hat = h_min.predict(X)
				y_hat = y_hat.reshape((y_hat.shape[0], 1))
				weights *= np.exp(-alpha * y_hat * y) 
				## Normalization
				weights /= sum(weights)
				assert math.isclose(sum(weights), 1.0)
		
		else:
			raiseExceptions('Wrong method')

	def sample_by_weight(self, X, weights):
		## probability based on weight
		p = weights.reshape(weights.shape[0])
		indices = np.random.choice(a=X.shape[0], size=X.shape[0], p=p, replace=True)
		X_samle = X[indices]
		return X_samle
	
	def predict(self, X):
		y_hat = np.ones((X.shape[0], 1),dtype=int)
		running_y =  np.zeros((X.shape[0]),dtype=float)
		for learner, alpha in zip(self.learners, self.alphas):
			running_y += learner.predict(X) * alpha
		mask = running_y < 0
		y_hat[mask] = -1
		return y_hat

#%%
# test 1. Handmade example.
if __name__ == '__main__':
	from helper_functions import plot_scatter, precision, recall, accuracy, specificity
	X = np.array([
		### negative examples
		[0.5, 0.1],
		[1, 0.6],
		[1.5, 0.7],
		[3, 2],
		[4, 2.8],
		### positive examples
		[0.5, 3],
		[1, 2.6],
		[2, 3],
		[3, 3.4],
		[4, 5]
	])
	## Add a column of 1 to X
	#X = np.hstack((X, np.ones((X.shape[0],1))))
	y0 = (-1)*np.ones((5, 1), dtype=int)
	y1 = np.ones((5, 1), dtype=int)
	y = np.concatenate((y0, y1))
	#y = y.reshape(y.shape[0])
	plot_scatter(X, y, face_color='lightgray')

	regressor = Adaboost(learnertype=Stump)
	regressor.fit(X, y)
	y_hat = regressor.predict(X)
	y_hat = y_hat.reshape((y_hat.shape[0],-1))
	plot_scatter(X, y_hat,face_color='lightgray')
	print(f'Recall: {recall(y, y_hat)*100}%')
	print(f'Precision: {precision(y, y_hat)*100}%')
	print(f'Accuracy: {accuracy(y, y_hat)*100}%')
	print(f'Specificity: {specificity(y, y_hat, negtave_lable=-1)*100}%')

#%%
#### test 2. Iris.
if __name__ == '__main__':
	from sklearn import datasets
	from sklearn.model_selection import train_test_split
	from helper_functions import plot_scatter, precision, recall, accuracy, specificity
	iris = datasets.load_iris()
	# Use the first two features ('sepal length (cm)', 'sepal width (cm)') for visulization. 
	X = iris['data'][:, 0:2] 
	## Append ones to X
	#X = np.hstack((X, np.ones((X.shape[0],1))))
	# 1 if setosa, else -1
	y = (iris['target'] == 0).astype(np.int) 
	## change negative label from 0 to -1
	mask = y == 0
	y[mask] = -1
	X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=1234)
	## Plot the training data
	title = 'Training data'
	plot_scatter(X_train[:,0:2], y_train, title=title, face_color='lightgray')
	## Plot the training and testing data, testing data in grey
	title = 'Training and testing data (testing data in black)'
	plot_scatter(X_train[:,0:2], y_train, X_test[:, 0:2], title=title,face_color='lightgray')
	regressor = Adaboost(learnertype=Stump)
	y_train = y_train.reshape((y_train.shape[0],1))
	regressor.fit(X_train, y_train)
	y_hat = regressor.predict(X_test)
	y_hat = y_hat.reshape((y_hat.shape[0],-1))
	
	y_test = y_test.reshape((y_test.shape[0],1))
	## Coloring test point based on predicted label
	title = 'Training and testing data (predicted color of testing data)'
	plot_scatter(X_train[:,0:2], y_train, X_test[:,0:2],  \
		y_hat,title=title,face_color='lightgray')

	## Coloring test point based on true label
	title = 'Training and testing data (true color of testing data)'
	plot_scatter(X_train[:,0:2], y_train, X_test[:,0:2],  \
		y_test, title=title,face_color='lightgray')

	print(f'Recall: {recall(y_test, y_hat)*100}%')
	print(f'Precision: {precision(y_test, y_hat)*100}%')
	print(f'Accuracy: {accuracy(y_test, y_hat)*100}%')
	print(f'Specificity: {specificity(y_test, y_hat, -1)*100}%')




