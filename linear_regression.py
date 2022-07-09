"""
Linear regression
"""
from calendar import c
import numpy as np 
import matplotlib.pyplot as plt
class LinearRegression:
	def __init__(self, lr=0.001, count_iter=1000):
		self.lr = lr
		self.count_iter = count_iter
		self.W = None
		self.b = None

	def fit(self, X, y):
		n_example, n_feature = X.shape
		assert n_example == y.shape[0], "Number of X and y do not match"

		self.W = np.random.normal(loc=0,scale=1,size=(n_feature, 1))

		for _ in range(self.count_iter):
			y_hat = np.dot(X, self.W)
			grd = self.grad(y_hat, y, X)
			self.W -= self.lr * grd
	
	def grad(self, y_hat, y, X):
		"""
		Gradient.
		"""
		## m: number of examples.
		m = X.shape[0]
		dw = np.matmul(X.T, (y_hat-y)) / m

		return dw

	def predict(self, X):
		y_hat = np.dot(X, self.W) 
		return y_hat

if __name__ == '__main__':
	from sklearn.model_selection import train_test_split
	m = 200
	X = np.linspace(start=0, stop=5, num=m)
	X = X.reshape(m, 1)
	ones = np.ones((m, 1))
	X = np.concatenate((X, ones), axis=1)
	## Add a column of ones to X

	k1 = 2
	k2 = 1
	#y = k1 * X[:,0] + k2 + np.random.normal(loc=0, scale=0.5, size=(X.shape[0],1))
	y = k1 * X[:,0] + k2 + np.random.normal(loc=0, scale=0.5, size=X.shape[0])
	y = y.reshape((m, 1)) 
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)


	lr = 0.05
	count_iter = 1000
	regressor = LinearRegression(lr,count_iter)
	regressor.fit(X_train, y_train)
	y_hat = regressor.predict(X_test)
	
	# Plot
	fig, ax = plt.subplots()
	ax.scatter(X_train[:, 0], y_train[:,0],c='g', s=30)
	ax.scatter(X_test[:, 0], y_test[:,0],c='r', s=30)
	## plot the fitted line
	# ax.plot(X_test[:, 0], y_hat, linestyle='--',c='black')
	# plot the fitted line using a point and the slop
	ax.axline(xy1=(X_test[0][0], y_test[0][0]), slope=regressor.W[0], linestyle='--',color='black')
	plt.show()

