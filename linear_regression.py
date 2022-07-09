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
		self.b = 0

		for _ in range(self.count_iter):
			y_hat = np.dot(X, self.W) + self.b
			y_diff = (y_hat - y)
			dW =  1/n_example * np.dot(X.T, y_diff)
			db = 1/n_example * np.sum(y_diff) 
			self.W -= self.lr * dW
			self.b -= self.lr * db 

	def predict(self, X):
		y_hat = np.dot(X, self.W) + self.b
		return y_hat

if __name__ == '__main__':
	from sklearn.model_selection import train_test_split
	num = 200
	X = np.linspace(start=0, stop=5, num=num)
	X = X.reshape(num,1)
	k = 2
	b = 1
	y = k * X + b + np.random.normal(loc=0, scale=0.5, size=(X.shape[0],1))
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

	#y_real = k*X_train[:,0]+b
	#ax.plot(X_train[:, 0], y_real, c='black')
	lr = 0.05
	count_iter = 1000
	regressor = LinearRegression(lr,count_iter)
	regressor.fit(X_train, y_train)
	y_hat = regressor.predict(X_test)
	
	# Plot
	fig, ax = plt.subplots()
	ax.scatter(X_train[:, 0], y_train[:,0],c='g', s=30)
	ax.scatter(X_test[:, 0], y_test[:,0],c='r', s=30)
	ax.plot(X_test[:, 0], y_hat, c='black')
	plt.show()

