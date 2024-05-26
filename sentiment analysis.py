# Natural Language Processing

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# import the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Cleaning the texts
import re
from string import punctuation
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    ps = PorterStemmer()
    review = re.sub("[^a-zA-Z]", ' ', text)
    review = review.lower()
    review = review.split()
    stop_words = set(stopwords.words('english'))
    review = [word for word in review if not word in stop_words]
    review = [ps.stem(word) for word in review]
    review = " ".join(review)
    
    return review

dataset['clean_review'] = dataset['Review'].map(preprocess_text)

corpus = []
for i in range(0, 1000):
    ps = PorterStemmer()
    review = re.sub("[^a-zA-Z]", ' ', dataset["Review"][i])
    review = review.lower()
    review = review.split()
    stop_words = set(stopwords.words('english'))
    review = [word for word in review if not word in stop_words]
    review = [ps.stem(word) for word in review]
    review = " ".join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1400)

# Defining dependent and independent variables
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
print(X.shape)


vocab = cv.vocabulary_
nam = cv.get_feature_names_out()
stop = cv.get_stop_words()


# Splitting the dataset into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)




# Training the model using Naive bayes
from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()


gb.fit(X_train, y_train)

# Predicting the Test set results
yhat = gb.predict(X_test)


# Evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix

ac = accuracy_score(y_test, yhat)


cm = confusion_matrix(y_test, yhat)





