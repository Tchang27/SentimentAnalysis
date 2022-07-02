Overview:
Neural network that performs sentiment analysis. It is trained on a dataset of movie reviews labeled either positive or
negative. The model is trained using backpropagation, and the learning rate and epochs can be adjusted. Once the model 
finishes training, the user can type in sentences for the model to evaluate.

Program Structure:
  Model 1: Built from scratch
  process.py - class that processes text data and converts it into vectors that can be used to train the neural net
  neuralnet.py - class for the neural net 
  model.py - class that trains the neural net using the processed data, measures its performance
    on testing data, and handles user/console interaction

  Model 2: Built using machine learning libraries
  complete_model.py - using python libraries, creates a sentiment analysis model

How to Use:
  Setting up Training and Testing Data:
    Data source from https://ai.stanford.edu/~amaas/data/sentiment/
    In the root directory, create a folder named 'data', then upload the 'train' and 'test'
    folders from the dataset into the folder.

  Run model.py
    Once the model is trained, it will prompt the user to write a sentence then evaluate whether it is positive or negative
    Quit the program by typing .quit

  Run complete_model.py
    Once the model is trained, it will prompt the user to write a sentence then evaluate whether it is positive or negative
    Quit the program by typing .quit

Known Issues and Bugs:
  Model 1 has a higher risk of overfiting the training data.
  The best configuration to minimize overfitting:
  architecture = [n,32,32,8,1]
  alpha = 0.01
  epochs = 300

Results:
  Model 1:
  Built from numpy, model 1 is slower to train than Model 2. Testing accuracy 
  on the testing data was 85% on negative sentiments and 85% on posiive sentiments, 
  meaning that there were 15% false positives and 15% false negatives.

  Model 2:
  Testing accuracy on the testing dataset was 86% on negative sentiments and 84% on 
  positive sentiments, meaning there were 14% false positives and 16% false negatives.

Conclusions:
  Both the models tend to perform sentiment analysis well on testing data. Model 1 
  took longer to train, but performed similarly to Model 2.

Citations:
Data gathered from:
  https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  https://ai.stanford.edu/~amaas/data/sentiment/
    @InProceedings{maas-EtAl:2011:ACL-HLT2011,
    author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
    title     = {Learning Word Vectors for Sentiment Analysis},
    booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
    month     = {June},
    year      = {2011},
    address   = {Portland, Oregon, USA},
    publisher = {Association for Computational Linguistics},
    pages     = {142--150},
    url       = {http://www.aclweb.org/anthology/P11-1015}
    }
Code resources:
  https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/
  https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
  https://realpython.com/python-ai-neural-network/
  https://towardsdatascience.com/deep-learning-with-python-neural-networks-complete-tutorial-6b53c0b06af0
