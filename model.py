from tkinter import Y
from process import Processor
from neuralnet import NeuralNet
import numpy as np
import os


class Model():
	#constructor trains model
	def __init__(self):
		'''
		Constructor of model, trains neural net to be able to
		conduct sentiment analysis on user inputs
		'''
		self.processor = Processor([])
		matrix, annotations = self.processor.get_matrix_with_annotations()
		training = np.array(matrix)
		y = np.array(annotations)

		n = len(matrix[0])
		self.nn = NeuralNet([n,int(n/2),int(n/2),1], 0.01)
		self.nn.fit(training, y, epochs=2500)

		accuracy = 0
		for (x, target) in zip(training, y):
			# make a prediction on the data point and display the result
			# to our console
			pred = self.nn.predict(x)[0][0]
			step = 1 if pred > 0.5 else 0
			if step == target[0]:
				accuracy += 1
		accuracy = accuracy / len(training)
		print("Accuracy of model on training data: " + str(accuracy))


		#running model on test data
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
		data.append(user_input)
		proc = Processor(data)

		matrix, unused = proc.get_matrix_with_annotations()
		vector_data = np.array(matrix)
		pred = m.nn.predict(vector_data)[0][0]

		step = 1 if pred > 0.5 else 0
		if step == 1:
			print("Prediction: positive")
		else:
			print("Prediction: negative")
		user_input = input("\nWrite a sentence: ")
