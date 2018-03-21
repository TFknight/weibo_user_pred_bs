# -*- coding: UTF-8 -*-
'''lda_tfidf-svm stack for age'''

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

        #标签转化,男:0,女:1
        list_tag = []
        for i in list_gender:

            list_tag.append(int(i))
        print "data have read "
        return list_total,list_tag

# -------------------------prepare lda svd -----------------------
    def prepare_lsi(self,doc):

        #给训练集用的
        list_total,list_tag = self.load_data(doc)

        stop_word = []

        texts = [[word for word in document.lower().split() if word not in stop_word]
                 for document in list_total]

        #train dictionary done
        dictionary = corpora.Dictionary(texts)  # 生成词典
        # 用TFIDF的方法计算词频,sublinear_tf 表示学习率
        tfv = TfidfVectorizer(min_df=1, max_df=0.95, sublinear_tf=True,stop_words=stop_word)
        # 对文本中所有的用户对应的所有的评论里面的单词进行ＴＦＩＤＦ的计算，找出每个词对应的tfidf值
        X_sp = tfv.fit_transform(list_total)
        corpus = [dictionary.doc2bow(text) for text in texts]
        #train model done
        tfidf_model = models.TfidfModel(corpus)
        joblib.dump(tfidf_model,"tfidf_model.model")

        corpus_tfidf = tfidf_model[corpus]

        lda_model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=200)

        joblib.dump(dictionary,"tfidf_dictionary.dict")
        print "训练集lda -----"
        joblib.dump(lda_model,"tfidf_lda.model")


        return tfidf_model,dictionary

    def train_lsi(self,doc):

        if not (os.path.exists("tfidf_model.model")):

            print "prepare model"
            #load train model
            tfidf_model,dictionary = self.prepare_lsi(doc)

            #load data
            list_total,list_tag = self.load_data(doc)
            stop_word = []
            texts = [[word for word in document.lower().split() if word not in stop_word]
                     for document in list_total]

            corpus = [dictionary.doc2bow(text) for text in texts]

        else:
            print "use model"
            #load train valid text
            tfidf_model = joblib.load("tfidf_model.model")
            dictionary = joblib.load("tfidf_dictionary.dict")

            #load data
            list_total,list_tag = self.load_data(doc)
            stop_word = []
            texts = [[word for word in document.lower().split() if word not in stop_word]
                     for document in list_total]

            corpus = [dictionary.doc2bow(text) for text in texts]


        lda_model = joblib.load("tfidf_lda.model")
        corpus_tfidf = tfidf_model[corpus]

        list_side = []

        corpus_lsi = lda_model[corpus_tfidf]
        nodes = list(corpus_lsi)

        #first build array
        for i in range(len(nodes)):
            list_zero = []
            for j in range(200):
                list_zero.append(0)
            list_side.append(list_zero)


        for i in range(len(nodes)):
            list_d = []
            for j in range(len(nodes[i])):
                list_side[i][j] = nodes[i][j][1]


        print mat(list_side).shape
        print "lda 矩阵构建完成----------------"

        return list_total,list_tag,list_side


    # -----------------------grid search--------------------
    def mygridsearch(self,train_features, train_labels, model, param_grid):
        grid_search = GridSearchCV(model, param_grid, n_jobs=-1, verbose=1)

        grid_search.fit(train_features, train_labels)
        # best_parameters = grid_search.best_estimator_.get_params()
        best_parameters = grid_search.best_params_
        return best_parameters

    #------------------------my mean count------------------

    def mymean(self,list_predict_score):
        num_total = 0
        for num in list_predict_score:
            num_total += num
        return num_total/(len(list_predict_score))

    #------------------------my reg score-------------------

    def get_regre_score(self,result_list):
        sum_fenzi = 0
        for line in result_list:
            sum_fenzi += np.square(line[0]-line[1])
        result = 1.0 * sum_fenzi/(result_list.__len__())
        return result

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
        train_list_side = mat(train_list_side)
        text_list_side = mat(text_list_side)
        valid_list_side = mat(valid_list_side)

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
        clf = svm.SVR()

        # # gridsearch 找最优模型
        param_grid = {'C': [0.8, 1, 2, 3, 4, 5]}
        best_parameters = self.mygridsearch(X_valid, y_valid, clf, param_grid)

        # print 'kernel: ', best_parameters['decision_function_shape']
        print 'C: ', best_parameters['C']

        clf = svm.SVR(kernel='linear', C=best_parameters['C'])

        # 逻辑回归训练模型
        clf.fit(X_train, y_train)
        # 用模型预测
        y_pred_te = clf.predict(X_text)

        # y_pred_te = np.argmax(y_pred_te,axis=1)
        print y_pred_te
        print "**"*20
        print y_text

        list_a_total = []
        for i in range(len(y_pred_te)):
            list_a = []
            list_a.append(y_pred_te[i])
            list_a.append(y_text[i])
            list_a_total.append(list_a)

        result = list_a_total
        print self.get_regre_score(result)

        # # #获取准确率
        # print self.myAcc(y_text, y_pred_te)
        # list_myAcc.append(self.myAcc(y_text, y_pred_te))
        #
        # print "doc2vec + 回归 + 支持向量机　准确率平均值为: "
        # print  self.mymean(list_myAcc)


if __name__ == '__main__':

    user_predict = user_predict("age_train.csv","age_test.csv","age_valid.csv")
    user_predict.predict()





