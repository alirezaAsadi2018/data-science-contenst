import codecs
import csv
import os
from math import log


import numpy as np
import pandas as pd
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import xgboost as xgb


# nltk.download('punkt')

def main():
    data = pd.read_csv("train_users.csv", low_memory=False)
    test_data = pd.read_csv("test_users.csv", low_memory=False)
    test_original = test_data.copy()
    
    categoryVal = data["title_fa_product"].unique()
    categoryValCount = len(categoryVal)
    category_dict = {}
    for i in range(0,categoryValCount):
        category_dict[categoryVal[i]] = i + 1
    data["title_fa_enc"] = data["title_fa_product"].map(category_dict).astype(int)
    
    #data = pd.get_dummies(data, columns=['title_fa_category'])

    
    
    data['advantages'].fillna(method = 'ffill', inplace = True)
    data['disadvantages'].fillna(method = 'ffill', inplace = True)
    data['title'] = data['title'].fillna('')
    data = data.dropna(axis=0)
    
    
    categoryVal = test_data["title_fa_product"].unique()
    categoryValCount = len(categoryVal)
    category_dict = {}
    for i in range(0,categoryValCount):
        category_dict[categoryVal[i]] = i + 1
    test_data["title_fa_enc"] = test_data["title_fa_product"].map(category_dict).astype(int)
    
    #test_data = pd.get_dummies(test_data, columns=['title_fa_category'])

    
    
    test_data['advantages'].fillna(method = 'ffill', inplace = True)
    test_data['disadvantages'].fillna(method = 'ffill', inplace = True)
    test_data['title'] = test_data['title'].fillna('')
    test_data = test_data.dropna(axis=0)
    
    
    X_train = data
    X_test = test_data
    library_classifier(X_train, X_test, test_original)
    
    
def preprocess(df):

    categoryVal = df["title_fa_product"].unique()
    categoryValCount = len(categoryVal)
    category_dict = {}
    for i in range(0,categoryValCount):
        category_dict[categoryVal[i]] = i + 1
    df["title_fa_enc"] = df["title_fa_product"].map(category_dict).astype(int)
    
    # Cleaning title into integers
    categoryVal = df["title"].unique()
    categoryValCount = len(categoryVal)
    category_dict = {}
    for i in range(0,categoryValCount):
        category_dict[categoryVal[i]] = i + 1
    df["title_enc"] = df["title"].map(category_dict).astype(int)
    
    
    
    #df['category_enc_log'] = np.log(df['category_enc'])
    df['advantages'].fillna(method = 'ffill', inplace = True)
    df['disadvantages'].fillna(method = 'ffill', inplace = True)
    df['title'] = df['title'].fillna('')
    df = df.dropna(axis=0)
    
    
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    zipped_series = pd.Series({vocab[i]:zipped[i] for i in vector[vector_index].indices})
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]
    
def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape(0)):
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
    return list(set(filter_list))

def library_classifier(data, test_data, test_original):
    comment_vectorizer = TfidfVectorizer(max_features=10)
    title_vectorizer = TfidfVectorizer(max_features=10)
    un_comments = data['comment'].unique()
    corpus_comment = [str(un_comments[i]).replace("\r\n", "")
                      for i in range(len(un_comments))]
    un_titles = data['title'].unique()
    corpus_title = [str(un_titles[i]).replace("\r\n", "")
                    for i in range(len(un_titles))]

    test_un_comments = test_data['comment'].unique()
    test_corpus_comment = [str(test_un_comments[i]).replace("\r\n", "")
                           for i in range(len(test_un_comments))]
    test_un_titles = test_data['title'].unique()
    test_corpus_title = [str(test_un_titles[i]).replace("\r\n", "")
                         for i in range(len(test_un_titles))]

    comment_vectorizer.fit_transform(corpus_comment)
    a = comment_vectorizer.transform(corpus_comment)
    title_vectorizer.fit_transform(corpus_title)
    b = title_vectorizer.transform(corpus_title)
    df1 = pd.DataFrame.sparse.from_spmatrix(a)
    df2 = pd.DataFrame.sparse.from_spmatrix(b)
    frames = [df1, df2, data]

    X_train = pd.concat(frames)
    

    a = comment_vectorizer.transform(test_corpus_comment)
    b = title_vectorizer.transform(test_corpus_title)
    df1 = pd.DataFrame.sparse.from_spmatrix(a)
    df2 = pd.DataFrame.sparse.from_spmatrix(b)
    frames = [df1, df2, test_data]
    X_test = pd.concat(frames)
    
    
    y_train = data["rate"]
    X_train = data.drop(['rate', 'verification_status', 'title_fa_category', 'title_fa_product', 'disadvantages', 'advantages', 'comment', 'id', 'title'], axis=1)
    X_test = test_data.drop(['title_fa_category', 'title_fa_product', 'disadvantages', 'advantages', 'comment', 'id', 'title'], axis=1)
    xgbr = xgb.XGBRegressor(verbosity=0)
    xgbr.fit(X_train, y_train)
    y_test = xgbr.predict(X_test)
    print(y_test)
    write_ans_to_csv(y_test, test_original)
    
    score = xgbr.score(X_train, y_train)  
    print("Training score: ", score)
    

def write_ans_to_csv(classes, test_data):
    with open('ans.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index, c in enumerate(classes):
            writer.writerow({'id': test_data["id"][index], 'rate': c})


main()
