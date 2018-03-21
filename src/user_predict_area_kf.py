# coding:utf-8
from __future__ import division
__author__ = "TF大Q"

import jieba
import pandas as pd
from sklearn import svm
import json
from numpy import *
import jieba.posseg as pseg
import os
import codecs
import pickle
import math
from gensim import models,corpora
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import sys
reload(sys)
sys.setdefaultencoding('utf8')

if __name__ == "__main__":
    list_bad = []
    list_good = []

    # -----------------------split word---------------------
    def SplitWord(sentences):  ##分词并去除停用词和单个词
        stopwordfile = open('/home/gaorui/PycharmProjects/untitled3/glove_study/weibo_comment/chi_selection-master/哈工大停用词表.txt', 'r')
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
    def load_data(doc):
        ys = {}
        list_age = []
        list_area = []
        list_gender = []
        list_total = []

        # 对应标签导入词典
        f = codecs.open(doc)
        temp = json.loads(f.read().replace("\n",""))
        print len(temp)

        for i in range(len(temp)):
            #每个user更新一次
            list_query = []
            list_gender.append(temp[i]['label'])
            for j in range(len(temp[i]['features']['followers'])):
                for k in ['info','nickname','user_type','area']:
                    list_query.append(temp[i]['features']['followers'][j][k])
            str_query = ''.join(list_query)

            # words = self.SplitWord(str_query)
            # list_total.append(' '.join(words))

            words = [word for word in jieba.cut(str_query)]
            list_total.append(' '.join(words))

        print list_total.__len__()
        #标签转化,男:0,女:1
        list_tag = []
        for i in list_gender:

            list_tag.append(int(i))
        print "data have read "
        return list_total,list_tag

    def prepare_tfidf(doc):
        list_total,list_tag = load_data(doc)
        stop_word = []

        texts = [[word for word in document.lower().split() if word not in stop_word]
                 for document in list_total]

        dictionary = corpora.Dictionary(texts)  # 生成词典



        from sklearn.feature_extraction.text import TfidfVectorizer
        # 用TFIDF的方法计算词频,sublinear_tf 表示学习率
        tfv = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf=True,stop_words=stop_word)
        # 对文本中所有的用户对应的所有的评论里面的单词进行ＴＦＩＤＦ的计算，找出每个词对应的tfidf值
        X_sp = tfv.fit_transform(list_total)
        print "X_sp: ",X_sp.shape
        X_sp = X_sp.toarray()

        return X_sp,list_tag


# -----------------------my score-----------------------

    def myAcc(y_true,y_pred):
        #最大数的索引
        y_pred = np.argmax(y_pred,axis=1)
        return np.mean(y_true == y_pred)

#------------------------my mean count------------------

    def mymean(list_predict_score):
        num_total = 0
        for num in list_predict_score:
            num_total += num
        return num_total/(len(list_predict_score))

    # -----------------------grid search--------------------
    def mygridsearch(train_features, train_labels, model, param_grid):
        grid_search = GridSearchCV(model, param_grid, n_jobs=-1, verbose=1)
        grid_search.fit(train_features, train_labels)
        best_parameters = grid_search.best_estimator_.get_params()
        # for para, val in list(best_parameters.items()):
        #     print para, val
        return best_parameters

    #------------------------my mean count------------------

    def mymean(list_predict_score):
        num_total = 0
        for num in list_predict_score:
            num_total += num
        return num_total/(len(list_predict_score))

    # ----------------------my selectKBest---------------------

    def myselctKBest(X,y,k_best):
        #最优k值
        X_new= SelectKBest(chi2, k=k_best).fit_transform(X, y)
        return X_new

# ------------------------------begin to predict------------

    X_sp_train,X_train_tag = prepare_tfidf("area_train.csv")
    X_sp_test,X_test_tag = prepare_tfidf("area_test.csv")
    X_sp_valid ,X_valid_tag= prepare_tfidf("area_valid.csv")

    TR = X_sp_train.__len__()
    TE = X_sp_test.__len__()
    TV = X_sp_valid.__len__()
    n = 5
    k_best = 300

    X_train = X_sp_train
    y_train = X_train_tag
    y_train = np.array(y_train)
    print X_train.shape

# # -----------------------ka fang ------------------------------------

    #加入卡方校验
    X_new_train = myselctKBest(X_train,y_train,k_best)
    print X_new_train.shape




    X_text = X_sp_test
    y_text = X_test_tag
    y_text = np.array(y_text)
    #加入卡方校验
    X_new_text = myselctKBest(X_text,y_text,k_best)
    print X_new_text.shape

    X_valid = X_sp_valid[:TV]
    y_valid = X_valid_tag[:TV]
    y_valid = np.array(y_valid)
    #加入卡方校验
    X_new_valid = myselctKBest(X_valid,y_valid,k_best)
    print X_new_valid.shape


    # kfold折叠交叉验证
    list_myAcc = []
    # for i, (tr, te) in enumerate(KFold(len(y_train), n_folds=n)):
    #     print "第"+  str(i ) +"次循环"
    clf = svm.SVC(probability=True)

    # gridsearch 找最优模型
    param_grid = {'C': [0.8, 1, 2, 3, 4, 5], 'decision_function_shape': ['ovo','ovr']}
    best_parameters = mygridsearch(X_valid, y_valid, clf, param_grid)

    print 'kernel: ', best_parameters['kernel']
    print 'C: ', best_parameters['C']

    clf = svm.SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], probability=True)

    # 逻辑回归训练模型
    clf.fit(X_new_train, y_train)
    # 用模型预测
    y_pred_te = clf.predict_proba(X_new_text)

    print np.argmax(y_pred_te,axis=1)
    print "**"*50
    print y_text

    # #获取准确率
    print myAcc(y_text, y_pred_te)
    list_myAcc.append(myAcc(y_text, y_pred_te))

    print "kf + 支持向量机　准确率平均值为: "
    print  mymean(list_myAcc)




