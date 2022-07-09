"""
References:
1: https://www.cs.cornell.edu/courses/cs4780/2017sp/lectures/lecturenote05.html
2. SMS Spam Collection Dataset UCI: https://www.kaggle.com/uciml/sms-spam-collection-dataset
3. https://www.youtube.com/watch?v=BqUmKsfSWho&list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E&index=5
4. https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/naive_bayes.py
"""
#%%
## Standard library
from collections import defaultdict
import math 
## Third parth library
import numpy as np
## My library
from helper_functions import gaussian, accuracy, recall,precision, specificity
from helper_functions import tokenize, plot_scatter 

#%%
class NaiveBayes:
	def __init__(self, case='gaussian', **kwargs) -> None:
		"""
		type: type of features, 2 types are supported:
			type='gaussian': the features are continuous. Each feature is an
			independent Gaussian distribution. Each label falls into one of K categories.
			type='Multinomial': Each feature represents a count. An example of 
			this is the count of a specific word in a document. (Spam filter). Each
			label is categorical.
		"""
		self.case = case
		if self.case == 'multinomial':
			self.vocabulary = set()
			self.spam_word_count = defaultdict(int)
			self.ham_word_count = defaultdict(int)
			self.spam_count = 0
			self.ham_count = 0

			if 'smooth_factor' in kwargs:
				self.smooth_factor = kwargs['smooth_factor']
			else:
				self.smooth_factor = 0.5
	
	def fit(self, X, y):
		assert len(X) == len(y), "Error! Length of X and y are different."
		if self.case == 'gaussian':
			self._fit_gaussian(X, y)

		if self.case == 'multinomial':
			self._fit_multinomial(X, y)

	def _fit_multinomial(self, X, y):
		"""
		Only support two classes. For example, spam or not.
		Only support spam detection for now.
		"""
		n_examples = X.shape[0]
		# unique_labels = np.unique(y)
		# self.unique_labels = unique_labels
		for index, message in X.items():
			## Count spam and ham messages
			if y[index] == 1:
				self.spam_count += 1
			else:
				self.ham_count += 1
			
			words = tokenize(message)
			for word in words:
				self.vocabulary.add(word)
				## For each word, count occurance. Do this for both spam and ham.
				if y[index] == 1:
					self.spam_word_count[word] += 1
				else:
					self.ham_word_count[word] += 1
		self.p_y_spam = self.spam_count / n_examples
		self.p_y_ham = self.ham_count / n_examples
		

	def _fit_gaussian(self, X, y):
		n_examples, n_features = X.shape
		unique_labels = np.unique(y)
		self.unique_labels = unique_labels
		self.X_miu = np.zeros((len(unique_labels), n_features))
		self.X_var = np.zeros((len(unique_labels), n_features))
		self.p_y = np.zeros_like(self.unique_labels, dtype=float)
		
		## For each label
		for index, c in enumerate(self.unique_labels):
			mask = (y == c)
			X_label_c = X[mask]
			self.p_y[index] = sum(mask) / float(n_examples)
			## For each feature
			for k in range(n_features):
				X_c = X_label_c[:, k]
				self.X_miu[index,k] = X_c.mean()
				self.X_var[index, k] = X_c.var()
				
	
	def predict(self, X):
		"""
		For each example in X, return the predicted label.
		"""
		if self.case == 'gaussian':
			y_hat = [self._predict_gaussian(x) for x in X]
			return np.array(y_hat)
		
		elif self.case == 'multinomial':
			y_hat = [self._predict_multinomial(x) for x in X]
			return np.array(y_hat)
			
	def _predict_gaussian(self, x):
		## ps is the probabilities of all labels. Return the largest one.
		ps = np.zeros_like(self.unique_labels, dtype=float) 
		n_features = len(x)
		for i, c in enumerate(self.unique_labels):
			## p(y|x) = p(x|y)p(y)/p(x),
			## argmax[p(y|x)] = argmax[p(x|y)p(y)] 
			ps[i] += np.log(self.p_y[i])
			## Naive Bayes assumption is used here.
			## Feature labels are independent given label.
			for f in range(n_features):
				miu_x = self.X_miu[c,f]
				var_x = self.X_var[c,f]
				p_x_i = gaussian(x[f], miu_x, math.sqrt(var_x))
				ps[i] += np.log(p_x_i)

		index_max_label = np.argmax(ps)
		max_label = self.unique_labels[index_max_label]
		return max_label
	
	def _predict_multinomial(self, x):
		#probabilities = np.zeros(2, dtype=float) 
		p_spam, p_ham = self._probability(x)
		if p_spam > p_ham:
			return 1
		else:
			return 0

	def _probability(self, x):
		words = tokenize(x)
		message_count = defaultdict(int)
		for word in words:
			message_count[word] += 1

		log_p_spam = log_p_ham = 0.0
		log_p_spam += np.log(self.p_y_spam)
		log_p_ham += np.log(self.p_y_ham)
		len_vocabulary = len(self.vocabulary)
		for v in self.vocabulary:
			## (# of word xi appeared in spams + k) / (# of words in all spams combined + k*len_vocabulary)
			## where k is the smooth factor. think about the smooth factor as each word occurred at least
			## k times.
			theta_alpha_spam = (self.spam_word_count[v] + self.smooth_factor) / (self.spam_count + len_vocabulary * self.smooth_factor)
			theta_alpha_ham = (self.ham_word_count[v] + self.smooth_factor) / (self.ham_count + len_vocabulary * self.smooth_factor)

			x_alpha = message_count[v]
			log_p_spam += x_alpha * np.log(theta_alpha_spam)
			log_p_ham += x_alpha * np.log(theta_alpha_ham)
		return log_p_spam, log_p_ham

#%%
if __name__ == '__main__':	
	## Test 1: Gaussian
	n_examples = 100
	n_features = 2
	mean1 = [1, 1]
	scale1 = 1
	cluster1 = np.random.normal(loc=mean1, scale=scale1, size=(n_examples, n_features))
	y_1 = np.zeros(len(cluster1), dtype=int)
	mean2 = [5, 2]
	scale2 = 1
	cluster2 = np.random.normal(loc=mean2, scale=scale2, size=(n_examples, n_features))
	y_2 = np.ones(len(cluster2), dtype=int)
	X_train = np.concatenate((cluster1, cluster2), axis=0)
	y_train = np.concatenate((y_1, y_2), axis=0)
	title = "Training data (two clusters of normal distribution)"
	plot_scatter(X_train, y_train,title=title)

	X_test = np.array([mean1, mean2])
	y_test = np.array([0, 1])
	title = 'Training and testing data (testing data in black)'
	plot_scatter(X_train, y_train,X_test, title=title)

	## Make a prediction
	naive_bayes = NaiveBayes()
	naive_bayes.fit(X_train, y_train)
	y_hat = naive_bayes.predict(X_test)
	## Plot testing data by predicted label
	title = 'Training and testing data (predicted color of testing data)'
	plot_scatter(X_train, y_train, X_test,  y_hat,title=title)

	## Plot testing data by true label
	## Coloring test point based on true label
	title = 'Training and testing data (true color of testing data)'
	plot_scatter(X_train, y_train, X_test,  y_test,title=title)
	print(f'Recall: {recall(y_test, y_hat)*100}%')
	print(f'Precision: {precision(y_test, y_hat)*100}%')
	print(f'Accuracy: {accuracy(y_test, y_hat)*100}%')
	print(f'Specificity: {specificity(y_test, y_hat)*100}%')
	

#%%

if __name__ == '__main__':
	#### Test 2
	
	
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

	## Make a prediction
	naive_bayes = NaiveBayes()
	naive_bayes.fit(X_train, y_train)
	y_hat = naive_bayes.predict(X_test)
	## Plot testing data by predicted label
	title = 'Training and testing data (predicted color of testing data)'
	plot_scatter(X_train, y_train, X_test,  y_hat,title=title)

	## Plot testing data by true label
	## Coloring test point based on true label
	title = 'Training and testing data (true color of testing data)'
	plot_scatter(X_train, y_train, X_test,  y_test,title=title)
	print(f'Recall: {recall(y_test, y_hat)*100}%')
	print(f'Precision: {precision(y_test, y_hat)*100}%')
	print(f'Accuracy: {accuracy(y_test, y_hat)*100}%')
	print(f'Specificity: {specificity(y_test, y_hat)*100}%')


#%%
#### Test Spam.
if __name__ == '__main__':
	import pandas as pd
	from sklearn.model_selection import train_test_split
	data = pd.read_csv('data/spam.csv', usecols=[0,1])
	## Rename the dataframe
	data = data.rename(columns={'v1':'category','v2':'message'})
	## Label 'Spam' to '1' and 'ham' to '0'
	data['label'] = np.where(data['category']=='spam',1, 0)
	## Split train and test
	X_train, X_test, y_train, y_test = train_test_split(data.loc[:,'message'],data.loc[:,'label'],train_size=0.8,random_state=4321)
	kwargs = {'smooth_factor': 0.5}
	naive_bayes = NaiveBayes(case='multinomial',**kwargs)
	naive_bayes.fit(X_train, y_train)
	y_hat = naive_bayes.predict(X_test)
	
	print(f'Recall: {recall(y_test, y_hat)*100}%')
	print(f'Precision: {precision(y_test, y_hat)*100}%')
	print(f'Accuracy: {accuracy(y_test, y_hat)*100}%')
	print(f'Specificity: {specificity(y_test, y_hat)*100}%')




# %%
