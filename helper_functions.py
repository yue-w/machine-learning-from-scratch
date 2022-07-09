
from tkinter import Y
from typing import Set
import math
import re 

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors

import numpy as np



def euclidean_distance(v1, v2):
	"""
	Compute the Euclidean distance of two vectors v1 and v2.
	v1 and v2 should be an 1-d numpy array
	"""
	return np.sqrt(np.sum((v1 - v2) ** 2))

def accuracy(y, y_hat):
	acc = np.sum(y == y_hat) / len(y)
	return acc

def precision(y, y_hat):
	mask_actual_positive = y == 1
	TP = sum(y_hat[mask_actual_positive] == 1)
	pred_positive = sum(y_hat==1)
	prec = TP/pred_positive
	return prec

def recall(y, y_hat):
	mask_actual_positive = y == 1
	TP = sum(y_hat[mask_actual_positive] == 1)
	actual_positive = sum(y==1)
	rec = TP/actual_positive
	return rec

def specificity(y, y_hat, negtave_lable=0):
	"""
	negtave_lable may be 0 or -1
	"""
	mask_TP = y == negtave_lable
	TN = sum(y_hat[mask_TP] == negtave_lable)
	actual_negative = sum( y == negtave_lable)
	spec = TN / actual_negative
	return spec


def gaussian(x, miu=0, sigma=1):
	#rst = 1 / (sigma * math.sqrt(2 * math.pi)) * np.exp(-0.5 * np.power(x-miu,2) / np.power(sigma,2))
	"""
	Calculate the Gaussian function of a number.
	"""
	rst = 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-0.5 * (x-miu)**2 / (sigma**2))
	return rst

def p_to_label(y):
	"""
	Convert probability to label. 0 if y <= 0.5, 1 if y > 0.5
	"""
	labels = [1 if i > 0.5 else 0 for i in y]
	return np.array(labels).reshape((len(labels), 1))

def plot_scatter(X_train, y_train, X_test=None,  y_test=None,fix_aspect=False,\
	title='', line=None, face_color=None):
	"""
	Scatter plot of categorical data. 
	The training data are distinghished by shape and color 
	If y_test is None, plot the label of test in black, otherwise, plot it based on
	its label.
	"""
	## Plot training data.
	## Get the total number of categories in the data.
	fig, ax = plt.subplots()
	## if y label is -1, the default color is white, change backgound color to
	## reveal the white white marker
	if face_color:
		ax.set_facecolor(face_color)

	unique_labels = np.unique(y_train)
	for i, c in enumerate(unique_labels):
		marker = Line2D.filled_markers[i]
		color = list(colors.BASE_COLORS.keys())[c]
		mask = (y_train == c)
		mask = mask.reshape(len(mask))
		X = X_train[mask]
		scatter = ax.scatter(X[:, 0], X[:, 1], c=color, marker=marker)

	## Plot the testing point. If y_hat is given, plot the color based on label.
	## If y_hat is None, plot the testing point in unfilled marker.
	if X_test is not None:
		if y_test is not None:
			unique_labels = np.unique(y_test)
			for i, c in enumerate(unique_labels):
				#marker = Line2D.filled_markers[i]
				color = list(colors.BASE_COLORS.keys())[c]
				mask = (y_test == c)
				mask = mask.reshape(len(mask))
				X = X_test[mask]
				ax.scatter(X[:, 0], X[:, 1], c=color, marker='+')
		else:
			## if y_test is not given, plot X_test with an unfilled marker in black
			ax.scatter(X_test[:, 0], X_test[:, 1], c='black', marker='+')
	if line:
		slope = line['slope']
		intercept = line['intercept']
		abline(slope, intercept)


	if fix_aspect:	
		# plt.xlim(0.5, 2.5) 
		# plt.ylim(-2.5, 2.5) 
		plt.gca().set_aspect('equal', adjustable='box')
	# produce a legend with the unique colors from the scatter
	#legend1 = ax.legend(*scatter.legend_elements())
	#ax.add_artist(legend1)
	plt.title(title)
	plt.show()


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')
	
def tokenize(text: str) -> Set[str]:
	"""
	Extract word and numbers from a string. Remove duplicates.
	"""
	text = text.lower()
	all_words = re.findall("[a-z0-9']+", text)
	return set(all_words)

def sigmoid(z):
	"""
	compute sigmoid function.
	z is a numpy array
	"""
	return 1 / (1 + np.exp(-z))

def step(x):
	"""
	return a vector with the same shape of x.
	return 1 if x >= 0, return 0 otherwise.
	"""
	return np.where(x>=0, 1, 0)

if __name__ == '__main__':
	"""
	
	v1 = np.array([0, 0])
	v2 = np.array([3, 4])
	assert euclidean_distance(v1, v2) == 5
	v1 = np.array([0, 4])
	v2 = np.array([3, 0])
	assert euclidean_distance(v1, v2) == 5

	import scipy.stats
	miu = 0
	sigma = 1
	v = -1
	print(gaussian(v))
	print( scipy.stats.norm(miu, sigma).pdf(v))
	assert math.isclose(gaussian(v), scipy.stats.norm(miu, sigma).pdf(v))
	y = np.array([1,1,0,0])
	y_hat = np.array([1,0,1,0])
	#assert precision(y, y_hat) == 2/3
	assert recall(y, y_hat) == 1/2
	z = np.array([0, 10, -10])
	print(sigmoid(z))
	"""





