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
Model 1 has a higher risk of overfiting the training data
The best configuration to minimize overfitting:
architecture = [n,32,32,8,1]
alpha = 0.01
epochs = 100

Results:
Model 1:
Built from scratch, using just numpy, model 1 is slower to train than Model 2. Testing data from the
same source as the training data yield accuracy around 86% after 100 iterations. Testing accuracy 
on external data was 91% on negative sentiments and 92% on posiive sentiments, meaning that there
were 9% false positives and 8% false negatives.

Model 2:
Using sklearn, the model gets around 84% on the testing data, which was partitioned from the 
training set and not used during learning. Testing accuracy on external data was 86% on negative
sentiments and 84% on positive sentiments, meaning there were 14% false positives and 16% false
negatives.

Conclusions:
Both the models tend to perform sentiment analysis well on testing data sourced from the same 
dataset as the training data. Model 1 performed better on the external dataset compared to 
Model 2.

Acknowledgements:
Data gathered from:
  https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
Code resources:
  https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/
  https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
  https://realpython.com/python-ai-neural-network/
  https://towardsdatascience.com/deep-learning-with-python-neural-networks-complete-tutorial-6b53c0b06af0
