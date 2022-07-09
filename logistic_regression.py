#%%
import numpy as np
from helper_functions import sigmoid,plot_scatter,p_to_label, recall, precision
from helper_functions import specificity, accuracy

#%%
class LogisticRegression:
	def __init__(self, lr=0.001, n_iters=1000) -> None:
		self.lr = lr 
		self.n_iters = n_iters
		self.w = None
	
	def fit(self, X, y):
		## m: number of examples. n: number of features (include the added 1)
		m, n = X.shape
		assert m == len(y)
		## We have added 1 to X, so ther is no b.
		## w is a column vector
		self.w = np.random.normal(loc=0, scale=0.1, size=(n, 1))
		for _ in range(self.n_iters):
			y_hat = self.predict(X)
			grd = self.grad(y_hat, y, X)
			self.w -= self.lr * grd

	def grad(self, y_hat, y, X):
		"""
		Compute gradient of w.
		"""
		## m: number of examples.
		m = X.shape[0]
		dw = np.matmul(X.T, (y_hat-y)) / m

		return dw

	def predict(self, X):
		z = np.dot(X, self.w)
		y_hat = sigmoid(z)
		return y_hat



#%%
### test 1. Handmade example.
if __name__ == '__main__':
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
	X = np.hstack((X, np.ones((X.shape[0],1))))

	y0 = np.zeros((5, 1), dtype=int)
	y1 = np.ones((5, 1), dtype=int)
	y = np.concatenate((y0, y1))
	plot_scatter(X[:, 0:2], y)

	lr = 0.05
	count_iter = 1000
	regressor = LogisticRegression(lr,count_iter)
	regressor.fit(X, y)
	probs = regressor.predict(X)
	y_hat = p_to_label(probs)
	w1, w2, w3 = regressor.w
	slope = - w1 / w2
	intercept = - w3 / w2
	line = {'slope':slope, 'intercept':intercept }
	plot_scatter(X[:, 0:2], y_hat, line=line)
	y_test = y




#%%
#### test 2. Iris.
if __name__ == '__main__':
	from sklearn import datasets
	from sklearn.model_selection import train_test_split
	iris = datasets.load_iris()
	# Use the first two features ('sepal length (cm)', 'sepal width (cm)') for visulization. 
	X = iris['data'][:, 0:2] 
	## Append ones to X
	X = np.hstack((X, np.ones((X.shape[0],1))))
	# 1 if setosa, else 0
	y = (iris['target'] == 0).astype(np.int) 
	y = y.reshape((len(y), 1))
	X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=1234)
	## Plot the training data
	title = 'Training data'
	plot_scatter(X_train[:,0:2], y_train, title=title)
	## Plot the training and testing data, testing data in grey
	title = 'Training and testing data (testing data in black)'
	plot_scatter(X_train[:,0:2], y_train, X_test[:, 0:2], title=title)
	regressor = LogisticRegression(lr=0.01)
	regressor.fit(X_train, y_train)
	props = regressor.predict(X_test)
	y_hat = p_to_label(props)
	
	## Coloring test point based on predicted label
	w1, w2, w3 = regressor.w
	slope = - w1 / w2
	intercept = - w3 / w2
	line = {'slope':slope, 'intercept':intercept }
	title = 'Training and testing data (predicted color of testing data)'
	plot_scatter(X_train[:,0:2], y_train, X_test[:,0:2],  y_hat,title=title, line=line)

	## Coloring test point based on true label
	title = 'Training and testing data (true color of testing data)'
	plot_scatter(X_train[:,0:2], y_train, X_test[:,0:2],  y_test,title=title)

	print(f'Recall: {recall(y_test, y_hat)*100}%')
	print(f'Precision: {precision(y_test, y_hat)*100}%')
	print(f'Accuracy: {accuracy(y_test, y_hat)*100}%')
	print(f'Specificity: {specificity(y_test, y_hat)*100}%')

# %%
