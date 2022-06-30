from process import Processor
from neuralnet import NeuralNet
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split


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
		X_train, X_test, y_train, y_test = train_test_split(training,y,test_size = 0.15, random_state=0)

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
		self.nn = NeuralNet([n,n,8,4,2,1], 0.01)
		self.nn.fit(X_train, clean_y_train, epochs=1)
		
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

		#running model on another test data
		neg_accuracy = self.test_model('data/neg', 0)
		print("Accuracy of model on neg testing data: " + str(neg_accuracy))
		pos_accuracy = self.test_model('data/pos', 1)
		print("Accuracy of model on pos testing data: " + str(pos_accuracy))
		
	
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
						test_data.append(text)

		test_proc = Processor(test_data)
		matrix, throwaway = test_proc.get_matrix_with_annotations()
		test_labels = [[target] for i in range(len(test_data))]

		test_data = np.array(matrix)
		test_labels = np.array(test_labels)

		accuracy = 0
		for (x, target) in zip(test_data, test_labels):
			# make a prediction on the data point and display the result
			# to our console
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
		data = []
		data.append(str(user_input))
		proc = Processor(data)

		matrix, unused = proc.get_matrix_with_annotations()
		vector_data = np.array(matrix)
		shape = np.shape(vector_data)
		padded_arr = np.zeros((1, 250))
		padded_arr[:shape[0],:shape[1]] = vector_data
		pred = m.nn.predict(padded_arr)[0][0]

		print("Model output: " + str(pred))
		step = 1 if pred > 0.5 else 0
		if step == 1:
			print("Prediction: positive")
		else:
			print("Prediction: negative")
		user_input = input("\nWrite a sentence: ")
