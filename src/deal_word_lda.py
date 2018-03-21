# -*- coding: UTF-8 -*-
'''tfidf-lsi-svm stack for gender'''

from __future__ import division
__author__ = "TF大Q"

import pickle
import os
import codecs
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import jieba
from numpy import *
import codecs
from gensim import models,corpora
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.externals import joblib

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class user_predict:
    def __init__(self, train_document, text_document,valid_document):
        self.train_document = train_document
        self.text_document = text_document
        self.valid_document = valid_document

    # -----------------------准确值计算-----------------------
    def myAcc(self,y_true,y_pred):
        #最大数的索引
        y_pred = np.argmax(y_pred,axis=1)

        return np.mean(y_true == y_pred)

    # -----------------------split word---------------------
    def SplitWord(self,sentences):  ##分词并去除停用词和单个词
        stopwordfile = open('./chi_selection-master/哈工大停用词表.txt', 'r')
        data = stopwordfile.readlines()
        stopwords = []
        for word in data:
            stopwords.append(word)
        words = jieba.cut_for_search(sentences)
        wordlist = []
        for word in words:
            wordlist.append(word)
        words = []
        for word in wordlist:
            if word.__len__() >= 2 and word.encode('utf-8') not in stopwords:
                words.append(word)
        return words

    # -----------------------load data-----------------------
    def load_data(self,doc):
        ys = {}
        list_age = []
        list_area = []
        list_gender = []
        list_total = []

        # 对应标签导入词典
        f = codecs.open(doc)
        temp = json.loads(f.read().replace("\n",""))
        print len(temp)

        doc_name = doc.replace(".csv","")

        for i in range(len(temp)):
            #每个user更新一次
            list_query = []
            list_gender.append(temp[i]['label'])
            for j in range(len(temp[i]['features']['followers'])):
                for k in ['info','nickname','user_type','area']:
                    list_query.append(temp[i]['features']['followers'][j][k])
            str_query = ''.join(list_query)

            words = [word for word in jieba.cut(str_query)]
            list_total.append(' '.join(words))

        print list_total.__len__()

        f2 = codecs.open(doc_name + ".txt","w+")
        f2.write(str(list_total.__len__()))
        f2.write("\n")
        for i in list_total:
            f2.write(i)
            f2.write("\n")

        return list_total


if __name__ == '__main__':

    user_predict = user_predict("gender_train.csv","gender_test.csv","gender_valid.csv")
    list_name = ["gender_train.csv","gender_test.csv","gender_valid.csv"]
    for i in list_name:
        user_predict.load_data(i)





