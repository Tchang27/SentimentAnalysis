from __future__ import annotations
from process import Processor
from neuralnet import NeuralNet
import numpy as np

processor = Processor()
neural_net = NeuralNet()

matrix, annotations = processor.get_matrix_with_annotations()
