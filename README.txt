Overview:
Neural network that performs sentiment analysis. It is trained on a dataset of movie reviews labeled either positive or
negative. The model is trained using backpropagation, and the learning rate and epochs can be adjusted. Once the model 
finishes training, the user can type in sentences for the model to evaluate.

Program Structure:
Model 1:
process.py - class that processes text data and converts it into vectors that can be used to train the neural net
neuralnet.py - class for the neural net 
model.py - class that trains the neural net using the processed data and handles user/console interaction

Model 2:
complete_model.py - using python libraries, creates a sentiment analysis model

How to Use:
Run model.py
Once the model is trained, it will prompt the user to write a sentence then evaluate whether it is positive or negative
Quit the program by typing .quit

Known Issues and Bugs:
Currently overfiting the training data, achieving around 9% accuracy 
on training data and 55% accuracy on testing data
data 

Acknowledgements:
Data gathered from:
  https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  https://ai.stanford.edu/~amaas/data/sentiment/
