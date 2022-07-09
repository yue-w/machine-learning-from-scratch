"""
Implement Decision Tree from scratch
Reference:
1: https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/decision_tree.py
2: https://stackoverflow.com/a/54074933/17254851

TODO: The current method treat catetorical features as numerical features (divide the feature
by whether the value is larger than or smaller than). 
"""
#%%
from collections import Counter
import numpy as np

#%%
class Node():
	def __init__(self, feature=None, threshold=None, left=None, right=None, *, label=None):
		"""
		feature: the index of the featue to split on
		threshold: the threshold to splt (<= split left, > split right)
		left: left node
		right: right node
		label: keyword-only argument. The label (value of y) of the node 
			   (if it is a leaf, None for non-leaf node).
		"""
		self.feature = feature
		self.threshold = threshold
		self.left = left 
		self.right = right
		self.label = label
	
	def isleaf(self):
		"""
		whether this node is a leaf node
		"""
		return self.label is not None
	
	def plot_tree(self):
		"""
		Plot the tree. 
		Ref: https://stackoverflow.com/a/54074933/17254851
		"""
		lines, *_ = self._plot_aux()
		for line in lines:
			print(line)

	def _plot_aux(self):
		"""Returns list of strings, width, height, and horizontal coordinate of the root."""
		# No child.
		if self.right is None and self.left is None:
			line = '%s' % self.label
			width = len(line)
			height = 1
			middle = width // 2
			return [line], width, height, middle

		text_not_leaf = f'{self.feature}, {self.threshold}' #f'Feature: {self.feature}, Threshold: {self.threshold}'
		# Only left child.
		if self.right is None:
			lines, n, p, x = self.left._plot_aux()
			s = '%s' % text_not_leaf#self.feature
			u = len(s)
			first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
			second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
			shifted_lines = [line + u * ' ' for line in lines]
			return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

		# Only right child.
		if self.left is None:
			lines, n, p, x = self.right._plot_aux()
			s = '%s' % text_not_leaf #self.feature
			u = len(s)
			first_line = s + x * '_' + (n - x) * ' '
			second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
			shifted_lines = [u * ' ' + line for line in lines]
			return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

		# Two children.
		left, n, p, x = self.left._plot_aux()
		right, m, q, y = self.right._plot_aux()
		s = '%s' % text_not_leaf #self.feature
		u = len(s)
		first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
		second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
		if p < q:
			left += [n * ' '] * (q - p)
		elif q < p:
			right += [m * ' '] * (p - q)
		zipped_lines = zip(left, right)
		lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
		return lines, n + m + u, max(p, q) + 2, n + u // 2
	
class DecisionTree():
	def __init__(self, max_depth=10, min_elements=2, num_features=None):
		"""
		max_depth: max depth
		min_elements: if number of nodes is smaller than this number, will not split further.
		num_features: index of the subset of features to split on. If None, use all features.
		"""
		self.root = None
		self.max_depth = max_depth
		self.min_elements = min_elements
		self.num_features = num_features

	def fit(self, X, Y):
		## Number of features to split. Support sub-features (used in random-forest).
		if not self.num_features: 
			self.num_features = X.shape[1]
		self.root = Node()
		self._grow_tree(self.root, X, Y) 

	def _grow_tree(self, node, X, Y, depth=0):
		n_unique_labels = len(np.unique(Y))
		## Stopping crteria
		num_points = X.shape[0]
		if (
			depth > self.max_depth or 
			num_points < self.min_elements or 
			n_unique_labels == 1
			):
			node.label = self._most_common_label(Y)

		else:
			## Find the best split. Return the feature and threshold
			feature, threshold = self._best_split(X, Y)
			node.feature = feature
			node.threshold = threshold

			## Carry out the split
			mask = X[:, feature] <= threshold
			X_left = X[mask]
			Y_left = Y[mask]
			mask = X[:, feature] > threshold
			X_right = X[mask]
			Y_right = Y[mask]

			node.left = Node() 
			node.right = Node()

			## If the best split is to group all node into one side (left or right)
			## then do not carry out this split. Make the current node a leaf
			if len(Y_left) == 0 or len(Y_right) == 0:
				node.label = self._most_common_label(Y)
			else:
				## Recursion
				self._grow_tree(node.left, X_left, Y_left, depth+1)
				self._grow_tree(node.right, X_right, Y_right, depth+1)

	
	def _best_split(self, X, Y):
		"""
		Find the feature and threshould to split.
		Criteria: minimum weighted sum of entropy of left child and right child 
		"""
		min_children_entropy = float('inf')
		threshold_best = None
		feature_best = None
		## Generate the index of features to split on. 
		features = np.random.choice(X.shape[1], self.num_features, replace=False) 
		for feature in features:
			thresholds = np.unique(X[:, feature])
			for threshold in thresholds:
				children_entropy = self._entropy(X, Y, feature, threshold)

				if children_entropy < min_children_entropy:
					min_children_entropy = children_entropy
					threshold_best = threshold
					feature_best = feature

		return feature_best, threshold_best


	def _entropy(self,X, Y, feature, threshold):
		mask_left = X[:, feature] <= threshold
		Y_left = Y[mask_left]
		mask_right = X[:, feature] > threshold
		Y_right = Y[mask_right]

		## Ignore split that makes left or right empty. (return inf)
		if len(Y_left) == 0 or len(Y_right) == 0:
			return 1#float('inf')

		left_entropy = self._child_entropy(Y_left)
		left_weight = len(Y_left) / len(Y)


		right_entropy = self._child_entropy(Y_right) 
		right_weight = len(Y_right) / len(Y)

		entropy = left_entropy * left_weight + right_entropy * right_weight
		return entropy
	
	def _child_entropy(self, Y):
		## if Y is 2D array, reshape it to get value
		Y_r = Y.reshape(Y.shape[0])
		counter = Counter(Y_r)
		ps = np.zeros(len(counter))
		for i, ele in enumerate(counter):
			ps[i] = counter[ele]/len(Y)

		return -sum([p * np.log2(p) for p in ps if p > 0])

	def _most_common_label(self, Y):
		## if Y is 2D array, reshape it to get value
		Y_r = Y.reshape(Y.shape[0])
		counter = Counter(Y_r)
		return counter.most_common(1)[0][0]

	def predict(self, X):
		# self.root.feature
		# self.root.threshold
		y_hat = [self._split(self.root, x) for x in X]
		return np.array(y_hat)
	
	def _split(self, node, x):
		if node.isleaf():
			return node.label
		else:
			feature = node.feature
			threshold = node.threshold
			if x[feature] <= threshold:
				return self._split(node.left, x)
			else:
				return self._split(node.right, x)

	def __str__(self):
		return (
			f'A decision tree with parameters:\n' 
			f'max depth: {self.max_depth}\n'
			f'min element in a node: {self.min_elements}\n'
			f'number of features to split: {self.num_features}\n'
			f'first node -> feature (index) to split :{self.root.feature}, threshold: {self.root.threshold}'
			)


#%%
if __name__ == "__main__":
	rain = [1,1,1,0,0,0,0,0,0,0]
	time = [30, 15, 5, 10, 5, 15, 20, 25, 30, 30]
	X = np.array([rain, time]).transpose()
	Y = np.array([0,0,0,0,0,1,1,1,1,1])
	dt = DecisionTree()
	dt.fit(X, Y)
	dt.root.plot_tree()
	# x = [0, 30]
	# y_hat = dt.predict(x)
	# print(f'Prediction of {x} is :{y_hat}')

#%%
## Test 2
## Read data
if __name__ == "__main__":
	from sklearn import datasets
	from sklearn.model_selection import train_test_split
	from helper_functions import accuracy, recall, precision, specificity
	data_iris = datasets.load_iris()
	X = data_iris.data
	Y = data_iris.target
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1234)
	print(f'Size of training set: {X_train.shape[0]}, Size of test set: {X_test.shape[0]}')
	#%%
	## Train
	dt = DecisionTree(max_depth=3)
	dt.fit(X_train, Y_train)
	dt.root.plot_tree()
	# %%
	## Test
	Y_hat = dt.predict(X_test)
	accy = accuracy(Y_test, Y_hat)
	print(f'Accuracy is: {accy*100}%')
	print(dt)
	print(f'Recall: {recall(Y_test, Y_hat)*100}%')
	print(f'Precision: {precision(Y_test, Y_hat)*100}%')
	print(f'Accuracy: {accuracy(Y_test, Y_hat)*100}%')
	print(f'Specificity: {specificity(Y_test, Y_hat)*100}%')