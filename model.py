from process import Processor
from neuralnet import NeuralNet
import numpy as np

processor = Processor()
matrix, annotations = processor.get_matrix_with_annotations()
training = np.array(matrix)
y = np.array(annotations)

n = len(matrix[0])
nn = NeuralNet([n,n,1], 0.1)
nn.fit(training, y, epochs=20000)

for (x, target) in zip(training, y):
	# make a prediction on the data point and display the result
	# to our console
	pred = nn.predict(x)[0][0]
	step = 1 if pred > 0.5 else 0
	print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(
		x, target[0], pred, step))
