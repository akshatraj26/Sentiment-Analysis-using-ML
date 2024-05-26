# Natural Language Processing

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

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
    
    
# Creating the Bag of Words model using Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1400)

# Creating a bag of model using tfidfvectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(max_features=1400)



X_tfidf = tf.fit_transform(corpus).toarray()
y_tfidf = dataset.iloc[:, 1].values

# Defining dependent and independent variables
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values



vocab = cv.vocabulary_
nam = cv.get_feature_names_out()
stop = cv.get_stop_words()


# Splitting the dataset into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X_tfidf, y_tfidf, test_size=0.25, random_state=0)




# Training the model using ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


names = ["Decision Tree",
            "Adaboost",
            "RandomForest",
            "LogisticRegression",
            "Multinomial",
            "Bernoulli",
            'Support Vector']

accuracy_cv = {}
confusion_cv = {}
models = [DecisionTreeClassifier(), AdaBoostClassifier(), RandomForestClassifier(),
          AdaBoostClassifier(),LogisticRegression(), MultinomialNB(), BernoulliNB(),
          SVC()]

for name, model in zip(names, models):
    mod = model.fit(X_train, y_train)
    y_pred = mod.predict(X_test)
    ac = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    accuracy_cv[name] = ac
    confusion_cv[name] = cm
    



accuracy_tfidf = {}
confusion_tfidf = {}


for name, model in zip(names, models):
    
    mod = model.fit(X_train_tf, y_train_tf)
    y_pred = mod.predict(X_test_tf)
    ac = accuracy_score(y_test_tf, y_pred)
    cm = confusion_matrix(y_test_tf, y_pred)
    
    accuracy_tfidf[name] = ac
    confusion_tfidf[name] = cm
    
    
    
# After analysing we see that Bernoulli
bn = BernoulliNB()
bn.fit(X_train_tf, y_train_tf)

y_pred = bn.predict(X_test_tf)
ac = accuracy_score(y_test_tf, y_pred)
ac


# Testing the Bernoulli model
sent = ['Food was best']
X_tfi = tf.transform(sent).toarray()
prediction = bn.predict(X_tfi)

prediction






