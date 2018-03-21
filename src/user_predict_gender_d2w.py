# -*- coding: UTF-8 -*-
''' doc2vc-svm stack for gender'''

from __future__ import division
__author__ = "TF大Q"

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import codecs
import gensim
import pandas as pd
import jieba
import json
from datetime import datetime
from collections import namedtuple
from gensim.models import Doc2Vec
from collections import OrderedDict
import subprocess
from sklearn import svm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score,KFold
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn.externals import joblib

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

        for i in range(len(temp)):
            #每个user更新一次
            list_query = []
            list_gender.append(temp[i]['label'])
            for j in range(len(temp[i]['features']['followers'])):
                # not area
                for k in ['info','nickname','user_type']:
                    list_query.append(temp[i]['features']['followers'][j][k])
            str_query = ''.join(list_query)

            # words = self.SplitWord(str_query)
            # list_total.append(' '.join(words))

            words = [word for word in jieba.cut_for_search(str_query)]
            list_total.append(' '.join(words))

        #标签转化,男:0,女:1
        list_tag = []
        for i in list_gender:

            list_tag.append(int(i))
        print "data have read "
        return list_total,list_tag

    # -------------------------prepare lsi svd -----------------------
    def prepare_lsi(self,doc):


        list_total,list_tag = self.load_data(doc)


        stop_word = []

        #构建语料库
        X_doc = []
        TaggededDocument = gensim.models.doc2vec.TaggedDocument
        for i in range(list_total.__len__()):
            word_list = list_total[i]
            document = TaggededDocument(word_list, tags=[i])
            X_doc.append(document)

        return X_doc,list_total,list_tag

    def train_lsi_model(self,doc):

        X_doc,list_total,list_tag = self.prepare_lsi(doc)
        #训练模型
        model_dm = Doc2Vec(X_doc,dm=0, size=300, negative=5, hs=0, min_count=1, window=30,sample=1e-5,workers=8,alpha=0.025,min_alpha=0.025)
        joblib.dump(model_dm,"model_d2v_dm.model")
        print "d2w模型训练完成"

        return model_dm


    def train_lsi(self,doc):

        if(os.path.exists("model_d2v_dm.model")):

            #load train model
            model_dm = joblib.load("model_d2v_dm.model")
        else:
            #load train model
            model_dm = self.train_lsi_model(doc)

        #prepare data
        X_doc,list_total,list_tag = self.prepare_lsi(doc)

        for i in range(10):
            # 一个用户作为一个文件去进行d2v的计算
            model_dm.train(X_doc, total_examples=model_dm.corpus_count, epochs=2)
            X_d2v = np.array([model_dm.docvecs[i] for i in range(len(list_total))])

        print X_d2v.shape

        list_side = X_d2v
        print " doc2vec 矩阵构建完成----------------"

        return list_total,list_tag,list_side



    # -----------------------grid search--------------------
    def mygridsearch(self,train_features, train_labels, model, param_grid):
        grid_search = GridSearchCV(model, param_grid, n_jobs=-1, verbose=1)
        grid_search.fit(train_features, train_labels)
        best_parameters = grid_search.best_estimator_.get_params()
        return best_parameters

    #------------------------my mean count------------------

    def mymean(self,list_predict_score):
        num_total = 0
        for num in list_predict_score:
            num_total += num
        return num_total/(len(list_predict_score))

    # ----------------------my selectKBest---------------------

    def myselctKBest(self,X,y,k_best):
        #最优k值
        X_new= SelectKBest(chi2, k=k_best).fit_transform(X, y)
        return X_new

    # ------------------------------begin to predict------------

    def predict(self):

        train_list_total,train_list_tag,train_list_side = self.train_lsi(self.train_document)
        print "train model done -------------------"

        text_list_total,text_list_tag,text_list_side = self.train_lsi(self.text_document)
        print "text model done  -------------------"

        valid_list_total,valid_list_tag,valid_list_side = self.train_lsi(self.valid_document)
        print "valid model done--------------------"

        TR = train_list_total.__len__()
        TE = text_list_total.__len__()
        TV = valid_list_total.__len__()

        n = 5
        # train_list_side = mat(train_list_side)
        # text_list_side = mat(text_list_side)
        # valid_list_side = mat(valid_list_side)

        X_train = train_list_side[:TR]
        y_train = train_list_tag[:TR]
        y_train = np.array(y_train)

        print "train shape :---------------------"
        print X_train.shape

        X_valid = valid_list_side[:TV]
        y_valid = valid_list_tag[:TV]
        y_valid = np.array(y_valid)

        print "valid shape :---------------------"
        print X_valid.shape

        X_text = text_list_side[:TE]
        y_text = text_list_tag[:TE]
        y_text = np.array(y_text)

        print "text shape :---------------------"
        print X_text.shape

        # kfold折叠交叉验证
        list_myAcc = []
        # for i, (tr, te) in enumerate(KFold(len(y_train), n_folds=n)):
        #     print "第"+  str(i ) +"次循环"
        clf = svm.SVC(probability=True)

        # gridsearch 找最优模型
        param_grid = {'C': [2, 3, 4, 5, 10, 20], 'kernel': ['linear']}
        best_parameters = self.mygridsearch(X_valid, y_valid, clf, param_grid)

        print 'kernel: ', best_parameters['kernel']
        print 'C: ', best_parameters['C']

        clf = svm.SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], probability=True)

        # 逻辑回归训练模型
        clf.fit(X_train, y_train)
        # 用模型预测
        y_pred_te = clf.predict_proba(X_text)

        # #获取准确率
        print self.myAcc(y_text, y_pred_te)
        list_myAcc.append(self.myAcc(y_text, y_pred_te))

        print "doc2vec + 支持向量机　准确率平均值为: "
        print  self.mymean(list_myAcc)


if __name__ == '__main__':

    user_predict = user_predict("gender_train.csv","gender_test.csv","gender_valid.csv")
    user_predict.predict()





