#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:02:19 2019

@author: takatoyamada
"""
from django.core.management.base import BaseCommand
import numpy as np
import pickle
import MeCab
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

vectorizer = CountVectorizer(min_df=10)
mecab = MeCab.Tagger('mecabrc')


class Command(BaseCommand):
    def handle(self, *args, **options):
        with open('./app/python/pickle/words_list.pickle', mode='rb')as f:
            words_list = pickle.load(f)
        X = vectorizer.fit_transform(words_list).toarray()
        y = make_label()
        (X_train, X_test, y_train, y_test) = train_test_split(
                                               X, y, test_size=0.3)
        valid(X_train, y_train, X_test, y_test)
        with open('./app/python/pickle/X_naive.pickle', mode='wb')as f:
            pickle.dump(X_train, f)
        with open('./app/python/pickle/y_naive.pickle', mode='wb')as f:
            pickle.dump(y_train, f)


def make_label():
    """Make label of X

    Args:
        None

    Returns:
        Label(8 categories, 100 data in each category)

    """
    for i in range(8):
        num = i+1
        y_ = np.full(100, num)
        if num == 1:
            y = y_
        else:
            y = np.concatenate((y, y_))
    return y


def np_log(x):
    return np.log(np.clip(a=x, a_min=1e-10, a_max=x))


def calcu_Pc(y_train):
    P = np.zeros((1, 8))
    for i in range(8):
        num = i+1
        P[0][i] = np.count_nonzero(y_train == num) / len(y_train)
    Pc = np_log(P)
    return Pc


def calc_Pwc(x, X_train, y_train):
    Pwc = np.ones((1, 8))
    for i in range(8):
        num = i+1
        X_i = X_train[np.where(y_train == num)]
        for j in range(len(x)):
            if x[j] >= 1:
                Pwc[0][i] += np_log((np.count_nonzero(X_i.T[j] >= 1)+1) / (
                              np.sum(X_i)+X_train.shape[1]))
    return Pwc


def predict(log_Pc, log_Pwc):
    pre = np.argmax(log_Pc+log_Pwc)
    return pre+1


def valid(X_train, y_train, X_valid, y_valid):
    """Calculate accuracy rate from validation data

    Args:
        X_train : train data
        y_train : train label
        X_valid : validation data
        y_valid : validation label

    Returns:
        None

    """
    
    log_Pc = calcu_Pc(y_train)
    m = len(y_valid)
    predicts = np.zeros((1, m))
    for i in range(m):
        x = X_valid[i]
        log_Pwc = calc_Pwc(x, X_train, y_train)
        predicts[0][i] = predict(log_Pc, log_Pwc)
    accu = np.count_nonzero(predicts[0] == y_valid) / m
    print("<naive bayes>")
    print("accuracy rate:%s" % str(accu))
