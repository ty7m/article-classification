#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:22:26 2019

@author: takatoyamada
"""
import numpy as np
import requests
import MeCab
import pickle
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 

mecab = MeCab.Tagger('mecabrc')
vectorizer = CountVectorizer(min_df = 60)

def extraction_article(url):
    text_list=[]
    soup = BeautifulSoup(requests.get(url).content,'html.parser') # 
    div = soup.find('div', class_="main article_main")
    for p in div.find_all('p'):
        text = p.string
        text_list.append(str(text))
    return text_list   
 
def tokenize(text):
    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0] == '名詞':
            yield node.surface.lower()
        node = node.next
        
def get_words(contents):
    ret = []
    for content in contents:
        ret.append(get_words_main(content))
    return ret

def get_words_main(content):
    return [taken for taken in tokenize(content)]

def make_label():
    for i in range(8):
        num = i+1
        y_ = np.full(100,num)
        if num ==1:
            y = y_
        else:
            y = np.concatenate((y,y_)) 
    return y

def np_log(x):
    return np.log(np.clip(a=x,a_min=1e-10,a_max=x))

def calcu_Pc(X,y):
    P = np.zeros((1,8))
    for i in range(8):
        num = i+1
        P[0][i] = np.count_nonzero(y==num)/len(y)
    Pc = np_log(P) 
    return Pc

def calc_Pwc(x,X_train,y_train):
    Pwc = np.ones((1,8))
    for i in range(8):
        num = i+1
        X_i = X_train[np.where(y_train==num)]
        for j in range(len(x)):
            if x[j] >= 1:
                Pwc[0][i] += np_log((np.count_nonzero(X_i.T[j]>=1)+1)/(np.sum(X_i)+X_train.shape[1]))
    return Pwc

def predict(log_Pc,log_Pwc):
    pre = np.argmax(log_Pc+log_Pwc)
    return pre

def main(url): 
    """Predict which category url belong to

    Args:
        url (str): url you want to know which category belong to

    Returns:
        None

    """
    with open('./app/python/pickle/words_list.pickle', mode = 'rb')as f:
            words_list = pickle.load(f)  
    X = vectorizer.fit_transform(words_list).toarray()
    y = make_label()
    (X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size=0.3)
    text_list=extraction_article(url)
    contents_ = [''.join(text_list)]  
    words = [' '.join(get_words(contents_)[-1])]
    x = vectorizer.transform(words).toarray()[-1]
    log_Pc = calcu_Pc(X_train,y_train)
    log_Pwc = calc_Pwc(x,X_train,y_train)
    pre = predict(log_Pc,log_Pwc)
    return pre

main("https://gunosy.com/articles/R3GIm")