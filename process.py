import math
import re
import csv
from tokenize import Token
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras import regularizers



STOP_WORDS = set(stopwords.words('english'))

'''
Class that processes sentiment annotated data for training neural net
'''
class Processor:
    def __init__(self, array):
        self.data = []
        self.annotations = []
        self.text_regex = r'''[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+'''
        
        #read in csv
        if len(array) == 0:
            with open('data/IMDB_Dataset.csv', 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    self.data.append(row[0])
                    self.annotations.append(row[1])
        else:
            self.data = array
        #parse data
        self.parse_all()

        #vectorize and standardize data for training
        for i in range(len(self.annotations)):
            if self.annotations[i] == 'positive':
                self.annotations[i] = [1]
            else:
                self.annotations[i] = [0]
        self.data = self.vectorize(self.data)

    def parse_all(self):
        '''
        Preprocesses the csv file data
        '''
        stemmer = PorterStemmer()
        stemmer_cache = {}
        #each review is a list consisting of text and sentiment
        for i, review in enumerate(self.data):
            self.parse_text(i, review, stemmer, stemmer_cache)

    def parse_text(self, index, review, stemmer, stemmer_cache):
        '''
        Tokenize, stem, stop all words in the text 
        Assign 1 for positive, 0 for negative sentiments
        '''
        clean_data = ''
        tokens = re.findall(self.text_regex, review)

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
                temp = stemmer.stem(t.lower())
                stemmer_cache[t] = temp
                return temp

        tokens = [process_token(token) for token in tokens if token not in STOP_WORDS]

        for token in tokens:
            clean_data += ' ' + token
        self.data[index] = clean_data

    def vectorize(self, data):
        max_words = 1000
        max_len = 100
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)
        data = pad_sequences(sequences, maxlen=max_len)

        data = np.divide(data, 100)

        return data
        
    def get_matrix_with_annotations(self):
        return self.data, self.annotations

