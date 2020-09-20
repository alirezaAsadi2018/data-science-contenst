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


# nltk.download('punkt')

def main():
    data = pd.read_csv("train_users.csv", low_memory=False)
    test_data = pd.read_csv("test_users.csv", low_memory=False)
    stops, stemmer = read_stops()
    # y_test = my_classifier(data, test_data)
    # y_test = classify_with_vectorization(data, test_data)
    y_test = library_classifier(data, test_data)
    print(y_test)
    write_ans_to_csv(y_test, test_data)

def classify_with_vectorization(data, test_data):
    corpus = [str(l).replace("\r\n", "") for l in data['comment']]
    count_vect = CountVectorizer()
    X = count_vect.fit_transform(corpus)
    vocabulary = count_vect.get_feature_names()
    X_train_counts = count_vect.fit_transform(corpus)
    condprob = [[], []]
    data_belonging_to_0 = data.loc[data["verification_status"] == 0, "comment"].tolist()
    corpus = [str(l).replace("\r\n", "") for l in data_belonging_to_0]
    X_train_counts = count_vect.fit_transform(corpus)
    vocabulary_0 = count_vect.get_feature_names()
    feature_freq = X_train_counts.sum(axis=0).tolist()[0]
    for i in range(len(vocabulary_0)):
        p = (feature_freq[i] + 1) / (len(vocabulary_0) + len(vocabulary))
        condprob[0] += [p]

    data_belonging_to_1 = data.loc[data["verification_status"] == 1, "comment"].tolist()
    corpus = [str(l).replace("\r\n", "") for l in data_belonging_to_1]
    X_train_counts = count_vect.fit_transform(corpus)
    vocabulary_1 = count_vect.get_feature_names()
    prior = [len(data_belonging_to_0) / len(data), len(data_belonging_to_1) / len(data)]
    feature_freq = X_train_counts.sum(axis=0).tolist()[0]
    for i in range(len(vocabulary_1)):
        p = (feature_freq[i] + 1) / (len(vocabulary_1) + len(vocabulary))
        condprob[1] += [p]

    classes = []
    X_new_counts = count_vect.transform([str(i) for i in test_data["comment"]])
    test_vocabulary = count_vect.get_feature_names()
    for q in X_new_counts:
        score = log(prior[0], 10)
        for v in find(q)[1]:
            if test_vocabulary[v] in set(vocabulary_0):
                feature_index = vocabulary_0.index(test_vocabulary[v])
                score += log(condprob[0][feature_index], 10)
        score_0 = score
        score = log(prior[1], 10)
        for v in find(q)[1]:
            if test_vocabulary[v] in set(vocabulary_1):
                feature_index = vocabulary_1.index(test_vocabulary[v])
                score += log(condprob[1][feature_index], 10)
        c = 0 if score_0 > score else 1
        classes += [c]
    return classes


def my_classifier(data, test_data):
    comment_lines = [str(l).replace("\r\n", "") for l in data['comment']]
    comment_vocabulary = extract_vocabulary(comment_lines)
    title_lines = [str(l).replace("\r\n", "") for l in data['title']]
    title_vocabulary = extract_vocabulary(title_lines)
    prior, condprob_comment, condprob_title = train_multinomial_NB(data, comment_vocabulary,
                                                                   title_vocabulary)
    classes = []
    for i in range(len(test_data)):
        comment = test_data["comment"][i]
        title = test_data["title"][i]
        c = apply_multinomial_NB(comment, title, prior, condprob_comment, condprob_title)
        classes += [c]
    return classes


def library_classifier(data, test_data):
    y_train = data["verification_status"]
    comment_vectorizer = TfidfVectorizer()
    title_vectorizer = TfidfVectorizer()
    corpus_comment = [str(data['comment'][i]).replace("\r\n", "")
                      for i in range(len(data['comment']))]
    corpus_title = [str(data['title'][i]).replace("\r\n", "")
                    for i in range(len(data['title']))]

    test_corpus_comment = [str(test_data['comment'][i]).replace("\r\n", "")
                           for i in range(len(test_data['comment']))]
    test_corpus_title = [str(test_data['title'][i]).replace("\r\n", "")
                         for i in range(len(test_data['title']))]

    comment_vectorizer.fit_transform(corpus_comment)
    a = comment_vectorizer.transform(corpus_comment)
    title_vectorizer.fit_transform(corpus_title)
    b = title_vectorizer.transform(corpus_title)
    c = np.array(data['is_buyer']).reshape((len(data), 1))
    X_train = hstack([a, b, c])

    fit = SelectKBest(score_func=chi2, k=2500).fit(X_train, y_train)
    X_train = fit.transform(X_train)

    a = comment_vectorizer.transform(test_corpus_comment)
    b = title_vectorizer.transform(test_corpus_title)
    c = np.array(test_data['is_buyer']).reshape((len(test_data), 1))
    X_test = hstack([a, b, c])
    X_test = fit.transform(X_test)

    # classifier = RandomForestClassifier(verbose=1, max_depth=400, random_state=0, n_jobs=4)
    classifier = BernoulliNB()
    classifier.fit(X_train, y_train)
    y_test = classifier.predict(X_test)
    return y_test


def read_stops():
    #read stopwords file
    stops = []
    with codecs.open('stopwords.txt', encoding='utf-8') as reader:
        stops = set(reader.read().split(os.linesep))
    stemmer = Stemmer()
    return stops, stemmer


def write_ans_to_csv(classes, test_data):
    with open('ans.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'verification_status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index, c in enumerate(classes):
            writer.writerow({'id': test_data["id"][index], 'verification_status': c})


def knn(vectors, query_vector, k):
    distance_list = []
    for c, vector in vectors:
        distance = 0
        for i in len(query_vector):
            distance += (query_vector[i] - vector[i]) ** 2
        distance_list += [(c, distance)]
    sorted_points = sorted(distance_list, key=lambda tup: tup[1])
    freq = [0, 0]
    for tup in sorted_points:
        if k == 0:
            break
        k -= 1
        freq[tup[0]] += 1
    return 0 if freq[0] > freq[1] else 1


def train_multinomial_NB(data, comment_vocabulary, title_vocabulary):
    # c=class = 0 or 1
    prior = []
    condprob_comment = []
    condprob_title = []
    for c in range(2):
        comments_belonging_to_c = data.loc[data["verification_status"] == c, "comment"].tolist()
        titles_belonging_to_c = data.loc[data["verification_status"] == c, "title"].tolist()
        if c == 0:
            comments_belonging_to_c = comments_belonging_to_c[:20000]
            titles_belonging_to_c = titles_belonging_to_c[:20000]
        Nc = len(comments_belonging_to_c)
        prior += [Nc / len(data)]
        dict = {w: 0 for w in comment_vocabulary}
        count = 0
        words_in_this_class = set()
        for senetence in comments_belonging_to_c:
            for word in word_tokenize(str(senetence)):
                words_in_this_class.add(word)
                if word in comment_vocabulary:
                    dict[word] += 1
        probability = {}
        for v in comment_vocabulary:
            p = (dict[v] + 1) / (len(words_in_this_class) + len(comment_vocabulary))
            probability[v] = p
        condprob_comment += [probability]

        dict = {w: 0 for w in title_vocabulary}
        count = 0
        words_in_this_class = set()
        for senetence in titles_belonging_to_c:
            for word in word_tokenize(str(senetence)):
                words_in_this_class.add(word)
                if word in title_vocabulary:
                    dict[word] += 1
        probability = {}
        for v in title_vocabulary:
            p = (dict[v] + 1) / (len(words_in_this_class) + len(title_vocabulary))
            probability[v] = p
        condprob_title += [probability]

    return prior, condprob_comment, condprob_title


def apply_multinomial_NB(comment, title, prior, condprob_comment, condprob_title):
    comment_words = extract_vectors_from_document(comment)
    title_words = extract_vectors_from_document(title)
    scores = []
    # c=class = 0 or 1
    for c in range(2):
        score = log(prior[c], 10)
        for w in comment_words:
            if w in condprob_comment[c]:
                score += log(condprob_comment[c][w], 10)

        for w in title_words:
            if w in condprob_title[c]:
                score += log(condprob_title[c][w], 10)
        scores += [score]
    return 0 if scores[0] > scores[1] else 1


def extract_vocabulary(lines):
    words_vector = set(str(lines).split(" "))
    # for word in word_tokenize(str(lines)):
    #     if word not in stops:
    #         words_vector.add(word)
    return words_vector


def extract_vectors_from_document(string):
    words_vector = set(str(string).split(" "))
    # for word in word_tokenize(str(string)):
    #     if word not in stops:
    #         words_vector.add(stemmer.stem(word))
    return words_vector


main()
