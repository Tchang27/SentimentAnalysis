import re
import csv
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer


STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.remove('not')

'''
Class that processes sentiment annotated data for training neural net
'''
class Processor:
    def __init__(self, array=[]):
        self.data = []
        self.annotations = []
        self.text_regex = r'''[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+'''
        
        self.cv = CountVectorizer(max_features=2500)

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
                self.annotations[i] = 1
            else:
                self.annotations[i] = 0
        self.data = self.vectorize(self.data)

    def parse_all(self):
        '''
        Preprocesses the csv file data
        '''
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stemmer_cache = {}
        #each review is a list consisting of text and sentiment
        for i, review in enumerate(self.data):
            self.parse_text(i, review, stemmer, lemmatizer, stemmer_cache)

    def parse_text(self, index, review, stemmer, lemmatizer, stemmer_cache):
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
                temp = lemmatizer.lemmatize(temp)
                stemmer_cache[t] = temp
                return temp

        tokens = [process_token(token) for token in tokens if token not in STOP_WORDS]

        for token in tokens:
            clean_data += ' ' + token
        self.data[index] = clean_data

    def vectorize(self, data):
        vec_mat = self.cv.fit_transform(data).toarray()
        return vec_mat
        '''
        max_words = 2000
        max_len = 200
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)
        data = pad_sequences(sequences, maxlen=max_len)

        data = np.divide(data, 200)

        return data
        '''
        
        
    def get_matrix_with_annotations(self):
        return self.data, self.annotations

