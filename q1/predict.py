import codecs
import csv
import os
from math import log

import nltk
import numpy as np
import pandas as pd
from hazm import Stemmer, Lemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, find, vstack, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import chi2, SelectKBest, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


# nltk.download('punkt')

def main():
    train = pd.read_csv("train.csv", low_memory=False)
    train['date'] = pd.to_datetime(train['date'])
    train['date'] = train['date'].apply(lambda x: x.month)
    test = pd.read_csv("test.csv", low_memory=False)
    test['date'] = pd.to_datetime(test['date'])
    test['date'] = test['date'].apply(lambda x: x.month)
    X_train = train.drop(['sales'], axis=1)
    y_train = train['sales']
    X_test = test
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)
    y_test = classifier.predict(X_test)
    write_ans_to_csv(y_test)

def write_ans_to_csv(classes, test_data):
    with open('ans.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'sales']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index, c in enumerate(classes):
            writer.writerow({'id': test_data["id"][index], 'sales': c})


main()
