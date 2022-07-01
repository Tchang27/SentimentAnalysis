from process import Processor
from neuralnet import NeuralNet
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text_regex = r'''[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+'''
stemmer = PorterStemmer()
stemmer_cache = {}
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.remove('not')

def process_token(t: str) -> str:
	"""
	Small helper to stem individual tokens (either get from cache or
	stem and add to cache)
	:param t: Token to process
	:return: Stemmed token
	"""
	if t in stemmer_cache:
		return stemmer_cache[t]
	else:
		temp = lemmatizer.lemmatize(t)
		temp = stemmer.stem(temp)
		stemmer_cache[t] = temp
		return temp

class Model():
	#constructor trains model
	def __init__(self):
		'''
		Constructor of model, trains neural net to be able to
		conduct sentiment analysis on user inputs
		'''
		self.processor = Processor()
		matrix, annotations = self.processor.get_matrix_with_annotations()
		training = np.array(matrix)
		y = np.array(annotations)
		y = pd.get_dummies(y)
		y = y.iloc[:,1].values
		X_train, X_test, y_train, y_test = train_test_split(training,y,test_size = 0.50, random_state=0)

		#reformat for training/testing
		clean_y_train = []
		clean_y_test = []
		for item in y_train:
			clean_y_train.append([item])
		for item1 in y_test:
			clean_y_test.append([item1])
		clean_y_train = np.array(clean_y_train)
		clean_y_test = np.array(clean_y_test)

		#train model
		n = len(matrix[0])
		self.nn = NeuralNet([n,32,32,8,1], 0.01)
		self.nn.fit(X_train, clean_y_train, epochs=100)
		
		#get accuraacy on training and testing
		accuracy = 0
		for (x, target) in zip(X_train, clean_y_train):
			# make a prediction on the data point and display the result
			# to our console
			pred = self.nn.predict(x)[0][0]
			step = 1 if pred > 0.5 else 0
			if step == target[0]:
				accuracy += 1
		accuracy = accuracy / len(X_train)
		print("Accuracy of model on training data: " + str(accuracy))

		accuracy = 0
		for (x, target) in zip(X_test, clean_y_test):
			# make a prediction on the data point and display the result
			# to our console
			pred = self.nn.predict(x)[0][0]
			step = 1 if pred > 0.5 else 0
			if step == target[0]:
				accuracy += 1
		accuracy = accuracy / len(X_test)
		print("Accuracy of model on partitioned testing data: " + str(accuracy))
	
		neg_acc = self.test_model('data/neg', 0)
		print("accuracy of model on neg test data: " + str(neg_acc))

		pos_acc = self.test_model('data/pos', 1)
		print("accuracy of model on pos test data: " + str(pos_acc))
		
	
	def test_model(self, dir, target):
		'''
		Function that reads in test folder of txt files and calculates
		accuracy of model.

		Param:
		dir - filepath to testing data
		target - target label 
		'''
		#test data for pos
		test_data = []
		for root, dirs, files in os.walk(dir):
			for file in files:
				if file.endswith('.txt'):
					with open(os.path.join(root, file), 'r') as f:
						text = f.read()
						test_data.append(text.lower())

		#stem/lemmatize data and remove stopwords
		test_corp = []
		for item in test_data:
			tokens = re.findall(text_regex, item)
			tokens = [process_token(token) for token in tokens if token not in STOP_WORDS]
			clean_data = ' '.join(tokens)
			test_corp.append(clean_data)

		#vectorize using the same cv as used in processing original data
		external_test = self.processor.cv.transform(test_corp).toarray()
		test_labels = [[target] for i in range(len(test_data))]

		test_mat = np.array(external_test)
		test_labels = np.array(test_labels)

		#calculate accuracy
		accuracy = 0
		for (x, target) in zip(test_mat, test_labels):
			pred = self.nn.predict(x)[0][0]
			step = 1 if pred > 0.5 else 0
			if step == target[0]:
				accuracy += 1

		accuracy = accuracy / len(test_data)
		return accuracy


if __name__ == "__main__":
	m = Model()
	user_input = input("\nWrite a sentence: ")
	while user_input != ".quit":
		u = user_input.lower()
		tokens = re.findall(text_regex, u)
		tokens = [process_token(token) for token in tokens if token not in STOP_WORDS]

		clean_data = ' '.join(tokens)
		u_corpus = [clean_data]
		u_X_test = m.processor.cv.transform(u_corpus).toarray()
		pred = m.nn.predict(u_X_test)[0][0]

		print("Model output: " + str(pred))
		step = 1 if pred > 0.5 else 0
		if step == 1:
			print("Prediction: positive")
		else:
			print("Prediction: negative")
		user_input = input("\nWrite a sentence: ")
