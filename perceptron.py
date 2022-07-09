"""
Perceptron.
Reference: 
1. https://youtu.be/t2ym2a3pb_Y
2. https://youtu.be/iZTeva0WSTQ
"""
#%%
from helper_functions import step, plot_scatter
import numpy as np


class Perceptron:
	def __init__(self, lr=1, max_loop=1e4) -> None:
		self.lr = lr 
		self.w = None 
		self.max_loop = max_loop
		## Record total iteration count.
		self.ite_count = 0
	
	def fit(self, X, y):
		## m examples, n featurs (including added 1)
		m, n = X.shape 
		assert m == len(y)
		## Initialize self.w as 0 vector
		self.w = np.zeros((n, 1))

		## Loop until all predictions are correct or has reached
		## max iteration 
		while self.ite_count < self.max_loop:
			y_hat = step(np.matmul(X, self.w))
			flag = y_hat != y 
			if sum(flag) == 0:
				break
			self.ite_count += 1
			## use vectorization
			self.w -=  self.lr * np.matmul(X.T, (y_hat - y))
		assert self.ite_count < self.max_loop, "Error, reached max iteration!"

	def predict(self, X):
		y_hat = step(np.matmul(X, self.w))
		return y_hat

#%%
## test 1. Handmade example.
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

	lr = 1
	count_iter = 1000
	regressor = Perceptron(lr,count_iter)
	regressor.fit(X, y)
	y_hat = regressor.predict(X)
	
	w1, w2, w3 = regressor.w
	slope = - w1 / w2
	intercept = - w3 / w2
	line = {'slope':slope, 'intercept':intercept }
	plot_scatter(X[:,0:2], y_hat, line=line)
	print(f'Total iteration number: {regressor.ite_count}')

### test 2. a larger dummy sample.
if __name__ == '__main__':
	from sklearn import datasets
	from sklearn.model_selection import train_test_split

	X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, \
		cluster_std=1.05, random_state=2)
	y = y.reshape((len(y), 1))
	## Add a column of 1 to X
	X = np.hstack((X, np.ones((X.shape[0],1))))

	plot_scatter(X[:, 0:2], y)

	lr = 1
	count_iter = 1000
	regressor = Perceptron(lr,count_iter)
	regressor.fit(X, y)
	y_hat = regressor.predict(X)
	
	w1, w2, w3 = regressor.w
	slope = - w1 / w2
	intercept = - w3 / w2
	line = {'slope':slope, 'intercept':intercept }
	plot_scatter(X[:,0:2], y_hat, line=line)
	print(f'Total iteration number: {regressor.ite_count}')
