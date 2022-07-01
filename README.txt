Overview:
Neural network that performs sentiment analysis. It is trained on a dataset of movie reviews labeled either positive or
negative. The model is trained using backpropagation, and the learning rate and epochs can be adjusted. Once the model 
finishes training, the user can type in sentences for the model to evaluate.

Program Structure:
Model 1: Built from scratch
process.py - class that processes text data and converts it into vectors that can be used to train the neural net
neuralnet.py - class for the neural net 
model.py - class that trains the neural net using the processed data and handles user/console interaction

Model 2: Built using machine learning libraries
complete_model.py - using python libraries, creates a sentiment analysis model

How to Use:
Run model.py
Once the model is trained, it will prompt the user to write a sentence then evaluate whether it is positive or negative
Quit the program by typing .quit

Known Issues and Bugs:
Currently Model 1 is overfiting the training data, achieving around 55% accuracy on testing data
data 

Results:
Model 1:
Built from scratch, using just numpy, model 1 is much slower and slightly lower in accuracy
compared to Model 2. Testing data from the same source as the training data yield accuracy 
around 77%. When predicting sentiments of user inputs, it tends to struggle.

Model 2:
Using sklearn, the model gets around 84% on the testing data, which was partitioned from the 
training set and not used during learning. When predicting new inputs from the user, it fares
better than Model 1

Conclusions:
Both the model created from scratch and the model created from python's machine learning 
libraries tend to perform sentiment analysis well on testing data sourced from the same 
dataset as the training data, but poorly on external datasets. This indicates the models 
are learning hidden patterns within the training dataset rather than generalized sentiment patterns.

Acknowledgements:
Data gathered from:
  https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  https://ai.stanford.edu/~amaas/data/sentiment/
