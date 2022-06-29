import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#read in and format files
df = pd.read_csv('data/IMDB_Dataset.csv')
df['sentiment'] = [1 if sentiment == 'positive' else 0 for sentiment in df['sentiment']]


#initial data analysis
print('Inital sentiment counts: ')
print(df['sentiment'].value_counts())

#processing words and creating corpus
corpus = []
text_regex = r'''[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+'''
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stemmer_cache = {}
STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.remove('not')

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
                #temp = stemmer.stem(t.lower())
                temp = lemmatizer.lemmatize(t)
                stemmer_cache[t] = temp
                return temp

def parse_text(idx):
        '''
        Tokenize, stem, stop all words in the text 
        Assign 1 for positive, 0 for negative sentiments
        '''
        clean_data = ''
        tokens = re.findall(text_regex, df['review'][idx])

        tokens = [process_token(token) for token in tokens if token not in STOP_WORDS]

        for token in tokens:
            clean_data += ' ' + token
        corpus.append(clean_data)

for i in tqdm(range(0, len(df['review']))):
    parse_text(i)

#create model
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y = pd.get_dummies(df['sentiment'])
y = y.iloc[:,1].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state=0)
model = MultinomialNB().fit(X_train, y_train)

#make predicitions
predictions = model.predict(X_test)
print('Model accuracy: ' + str(accuracy_score(y_test, predictions)))
print(classification_report(predictions, y_test))

user_input = input("\nWrite a sentence: ")
while user_input != ".quit":
    u = user_input.lower()
    tokens = re.findall(text_regex, u)
    tokens = [process_token(token) for token in tokens if token not in STOP_WORDS]

    clean_data = ' '.join(tokens)
    u_corpus = [clean_data]
    u_X_test = cv.transform(u_corpus).toarray()
    u_pred = model.predict(u_X_test)
    u_pred = 'positive' if u_pred > 0.5 else 'negative'
    print("Prediction: " + str(u_pred))

    user_input = input("\nWrite a sentence: ")