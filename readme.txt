本项目是用作用户年龄，性别，地区预测实验的baseline

user_predict_area_tfidf.py 是用tfidf+lsi＋ＳＶＭ进行地区预测　
user_predict_area_d2w.py 是用d2w＋ＳＶＭ进行地区预测　
user_predict_gender_kf.py 是用卡方校验+tfidf＋svm进行性别预测
user_predict_gender_lda.py 是用lda+tfidf＋svm进行性别预测
text_hstack.py 是用dbow + dm　两种doc2vec模型融合进行预测


