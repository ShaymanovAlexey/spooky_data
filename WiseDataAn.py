import unicodecsv as csv
import codecs
from datetime import datetime
import numpy as np
import pandas as pd
import nltk
import time
import enchant
import re
# function for making ngrams
from nltk.util import ngrams
import collections
import sys
import math
from collections import Counter
from textblob import TextBlob
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import string
import xgboost as xgb
from sklearn import ensemble, metrics, model_selection,svm, naive_bayes
from scipy.sparse import hstack
from sklearn.pipeline import  FeatureUnion
from matplotlib.pyplot import  colormaps as cm
from collections import Counter
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
import spacy
from sklearn.model_selection import GridSearchCV



nlp = spacy.load('en')
stop_words = set(stopwords.words('english'))

#import only English dictionary
eng_dict = enchant.Dict("en_US")


PATH_TRAIN = './input/train.csv'
PATH_TEST = './input/test.csv'

fl = codecs.open(PATH_TRAIN, 'r','utf-8')
l = fl.readline()

#find semicolon and comma in samples

part_speech_dict = {}

names_frequntly = {}

most_frequently_pattern = {}

list_of_names = []

NGRAMS = 5

COUNT_GRAMS = 300

cur_time = time.time()


l_most_grams = {}

keys_author_head = []

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

grouped_df = train_df.groupby('author')

## Number of words in the text ##
train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))

train_df["unique_words"] = train_df["text"].apply(lambda x:len(set(str(x).split())))
test_df["unique_words"] = test_df["text"].apply(lambda x: len(set(str(x).split())))

#we see distribution of words
# sns.stripplot(x='author', y='num_words', data=train_df, jitter=True, palette="Set2", split=True,linewidth=1,edgecolor='gray')
# plt.xlabel('Author Name', fontsize=12)
# plt.ylabel('Number of words in text', fontsize=12)
# plt.title("Number of words by author", fontsize=15)
#plt.show()


train_df["num_punctuations"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test_df["num_punctuations"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

#we see distribution of punctuation
# sns.stripplot(x='author', y='num_punctuations', data=train_df, jitter=True, palette="Set2", split=True,linewidth=1,edgecolor='gray')
# plt.xlabel('Author Name', fontsize=12)
# plt.ylabel('Number of punctuations  in text', fontsize=12)
# plt.title("Number of punctuations  by author", fontsize=15)
#plt.show()


train_df["num_questions"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in ['?']]) )
test_df["num_questions"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in ['?']]) )

train_df["num_three_dots"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in ['...']]) )
test_df["num_three_dots"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in ['...']]) )

#we see distribution of punctuation
# sns.stripplot(x='author', y='num_questions', data=train_df, jitter=True, palette="Set2", split=True,linewidth=1,edgecolor='gray')
# plt.xlabel('Author Name', fontsize=12)
# plt.ylabel('Number of questions  in text', fontsize=12)
# plt.title("Number of questions  by author", fontsize=15)
#plt.show()

train_df["num_exclam"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in ['!']]) )
test_df["num_exclam"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in ['!']]) )

#we see distribution of punctuation
# sns.stripplot(x='author', y='num_exclam', data=train_df, jitter=True, palette="Set2", split=True,linewidth=1,edgecolor='gray')
# plt.xlabel('Author Name', fontsize=12)
# plt.ylabel('Number of exclamation   in text', fontsize=12)
# plt.title("Number of exclamation   by author", fontsize=15)
#plt.show()

train_df["num_dots"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in ['.']]) )
test_df["num_dots"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in ['.']]) )

#we see distribution of punctuation
# sns.stripplot(x='author', y='num_dots', data=train_df, jitter=True, palette="Set2", split=True,linewidth=1,edgecolor='gray')
# plt.xlabel('Author Name', fontsize=12)
# plt.ylabel('Number of dots  in text', fontsize=12)
# plt.title("Number of dots  by author", fontsize=15)
#plt.show()

train_df["num_semicolon"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in [';']]) )
test_df["num_semicolon"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in [';']]) )
#we see distribution of punctuation
# sns.stripplot(x='author', y='num_semicolon', data=train_df, jitter=True, palette="Set2", split=True,linewidth=1,edgecolor='gray')
# plt.xlabel('Author Name', fontsize=12)
# plt.ylabel('Number of semicolon  in text', fontsize=12)
# plt.title("Number of semicolon  by author", fontsize=15)
#plt.show()

train_df["num_comma"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in [',']]) )
test_df["num_comma"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in [',']]) )
#we see distribution of punctuation
# sns.stripplot(x='author', y='num_comma', data=train_df, jitter=True, palette="Set2", split=True,linewidth=1,edgecolor='gray')
# plt.xlabel('Author Name', fontsize=12)
# plt.ylabel('Number of comma  in text', fontsize=12)
# plt.title("Number of comma  by author", fontsize=15)
#plt.show()

#not in text
train_df["num_dash"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in ['-']]) )
test_df["num_dash"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in ['-']]) )

#we see distribution of punctuation
# sns.stripplot(x='author', y='num_dash', data=train_df, jitter=True, palette="Set2", split=True,linewidth=1,edgecolor='gray')
# plt.xlabel('Author Name', fontsize=12)
# plt.ylabel('Number of dash  in text', fontsize=12)
# plt.title("Number of dash  by author", fontsize=15)
#plt.show()


#we see distribution of mean word
train_df["mean_word_len"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df["mean_word_len"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
# sns.stripplot(x='author', y='mean_word_len', data=train_df, jitter=True, palette="Set2", split=True,linewidth=1,edgecolor='gray')
# plt.xlabel('Author Name', fontsize=12)
# plt.ylabel('Number of mean word length  in text', fontsize=12)
# plt.title("Number of mean word length  by author", fontsize=15)
#plt.show()
#print(train_df["mean_word_len"])

train_df["pure_text"] = train_df["text"].apply(lambda x:re.sub('[.,"?;&]+', '', x))
train_df["mean_pure_word_len"] = train_df["pure_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df["pure_text"] = test_df["text"].apply(lambda x:re.sub('[.,"?;&]+', '', x))
test_df["mean_pure_word_len"] = test_df["pure_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
# sns.stripplot(x='author', y='mean_pure_word_len', data=train_df, jitter=True, palette="Set2", split=True,linewidth=1,edgecolor='gray')
# plt.xlabel('Author Name', fontsize=12)
# plt.ylabel('Number of mean pure word length  in text', fontsize=12)
# plt.title("Number of mean pure word length  by author", fontsize=15)
#plt.show()

# time1 = time.time()
# #print(train_df[["mean_word_len","mean_pure_word_len"]])
train_df["pure_more_text"] = train_df["pure_text"].apply(lambda x: [w for w in str(x).lower().split() if w not in stop_words])
train_df["pure_more_text"] = train_df["pure_more_text"].apply(lambda x: ' '.join(str(e) for e in x))
train_df["part_of_speech"] = train_df["pure_more_text"].apply(lambda x: [w.tag_ for w in nlp(x)])
train_df["part_of_speech"] = train_df["part_of_speech"].apply(lambda x: ' '.join(str(e) for e in x))
# train_df["part_of_speech_dep"] = train_df["pure_more_text"].apply(lambda x: [w.dep_ for w in nlp(x)])
# train_df["part_of_speech_dep"] = train_df["part_of_speech_dep"].apply(lambda x: ' '.join(str(e) for e in x))
train_df["mean_pure_more_word_len"] = train_df["pure_more_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
#
test_df["pure_more_text"] = test_df["pure_text"].apply(lambda x: [w for w in str(x).lower().split() if w not in stop_words])
test_df["pure_more_text"] = test_df["pure_more_text"].apply(lambda x: ' '.join(str(e) for e in x))
test_df["part_of_speech"] = test_df["pure_more_text"].apply(lambda x: [w.tag_ for w in nlp(x)])
test_df["part_of_speech"] = test_df["part_of_speech"].apply(lambda x: ' '.join(str(e) for e in x))
# test_df["part_of_speech_dep"] = test_df["pure_more_text"].apply(lambda x: [w.dep_ for w in nlp(x)])
# test_df["part_of_speech_dep"] = test_df["part_of_speech_dep"].apply(lambda x: ' '.join(str(e) for e in x))
test_df["mean_pure_more_word_len"] = test_df["pure_more_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# sns.stripplot(x='author', y='mean_pure_more_word_len', data=train_df, jitter=True, palette="Set2", split=True,linewidth=1,edgecolor='gray')
# plt.xlabel('Author Name', fontsize=12)
# plt.ylabel('Number of mean pure more word length in text', fontsize=12)
# plt.title("Number of mean pure more word length  by author", fontsize=15)
#plt.show()
#get more features for most frequency words

def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=23, child=1, colsample=0.4,params={}):
    param = {}
    param['eta'] = 0.03
    param['objective'] = 'multi:softprob'
    param['max_depth'] = 3
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = child
    param['subsample'] = 0.8
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 2000

    plst = list(param.items()) + list(params.items())
    print(plst)
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, verbose_eval=40)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    return pred_test_y, pred_test_y2, model

#for only features
#kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
#cv_scores = []
#pred_full_test = 0
#pred_train = np.zeros([train_df.shape[0], 3])
#for dev_index, val_index in kf.split(train_X):
#    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
#    dev_y, val_y = train_y[dev_index], train_y[val_index]
#    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0)
#    pred_full_test = pred_full_test + pred_test_y
#    pred_train[val_index,:] = pred_val_y
#    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

#print("cv scores : ", cv_scores)

#fig, ax = plt.subplots(figsize=(12,12))
#xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
#plt.show()

def runSVM(train_X, train_y, test_X, test_y, test_X2):
    model = svm.SVC()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

#-----------------------------------------------------------------Count Vectorizer n-grams -----------------------------------------------------------------------------------------------------
author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
train_y = train_df['author'].map(author_mapping_dict)
cols_to_drop = ['id', 'text', 'pure_text', 'mean_word_len', 'pure_more_text','num_three_dots', 'num_exclam', 'num_dash']
target = 'author'
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,4))
full_tfidf = tfidf_vec.fit(train_df['pure_more_text'].values.tolist() + test_df['pure_more_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['pure_more_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['pure_more_text'].values.tolist())


#for vector data sparce matrix
kf = model_selection.KFold(n_splits=6, shuffle=True, random_state=10)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
true_train = np.zeros([train_df.shape[0]])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    true_train[val_index] = val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
pred_full_test = pred_full_test / 6.
print("Mean cv score : ", np.mean(cv_scores))

# conf_repres_tr_y = true_train
# conf_repres_pre_y = np.argmax(pred_train, axis=1)
#
# fig2 = plt.figure()
# mat = confusion_matrix(conf_repres_tr_y, conf_repres_pre_y)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=author_mapping_dict.keys(), yticklabels=author_mapping_dict.keys())
# plt.xlabel('true label')
# plt.ylabel('predicted label')
#plt.show()

train_df["cv_ngram_1_4_eap"] = pred_train[:,0]
train_df["cv_ngram_1_4_hpl"] = pred_train[:,1]
train_df["cv_ngram_1_4_mws"] = pred_train[:,2]
test_df["cv_ngram_1_4_eap"] = pred_full_test[:,0]
test_df["cv_ngram_1_4_hpl"] = pred_full_test[:,1]
test_df["cv_ngram_1_4_mws"] = pred_full_test[:,2]
#
#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------Tfid  Vectorizer n-grams ----------------------------------------------------------------------------
author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
train_y = train_df['author'].map(author_mapping_dict)
cols_to_drop = ['id', 'text', 'pure_text', 'mean_word_len', 'pure_more_text','num_three_dots', 'num_exclam', 'num_dash']
target = 'author'
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

tfidf_vec = TfidfVectorizer(ngram_range=(1,3))
full_tfidf =tfidf_vec.fit_transform(train_df['text'].values.tolist()+test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
features = tfidf_vec.get_feature_names()

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
true_train = np.zeros([train_df.shape[0]])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=10)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    true_train[val_index] = val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# conf_repres_tr_y = true_train
# conf_repres_pre_y = np.argmax(pred_train, axis=1)
#
# fig3 = plt.figure()
# mat = confusion_matrix(conf_repres_tr_y, conf_repres_pre_y)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=author_mapping_dict.keys(), yticklabels=author_mapping_dict.keys())
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()

train_df["tv_ngram_1_4_word_eap"] = pred_train[:,0]
train_df["tv_ngram_1_4_word_hpl"] = pred_train[:,1]
train_df["tv_ngram_1_4_word_mws"] = pred_train[:,2]
test_df["tv_ngram_1_4_word_eap"] = pred_full_test[:,0]
test_df["tv_ngram_1_4_word_hpl"] = pred_full_test[:,1]
test_df["tv_ngram_1_4_word_mws"] = pred_full_test[:,2]


n_comp = 150
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))

train_svd.columns = ['svd_tv_ngram_word_' + str(i) for i in range(n_comp)]
test_svd.columns = ['svd_tv_ngram_word_' + str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)

del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
#
# #------------------------------------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------Count Vectorizer char grams -----------------------------------------------------------------------------------------------------

# author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
# train_y = train_df['author'].map(author_mapping_dict)
# cols_to_drop = ['id', 'text', 'pure_text', 'mean_word_len', 'pure_more_text','num_three_dots', 'num_exclam', 'num_dash']
# target = 'author'
# train_X = train_df.drop(cols_to_drop+['author'], axis=1)
# test_X = test_df.drop(cols_to_drop, axis=1)
#
# tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,4), analyzer='char')
# full_tfidf = tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
# train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
# test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
#
#
# #for vector data sparce matrix
# kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=10)
# cv_scores = []
# pred_full_test = 0
# pred_train = np.zeros([train_df.shape[0], 3])
# true_train = np.zeros([train_df.shape[0]])
# for dev_index, val_index in kf.split(train_X):
#     dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
#     dev_y, val_y = train_y[dev_index], train_y[val_index]
#     pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
#     pred_full_test = pred_full_test + pred_test_y
#     pred_train[val_index,:] = pred_val_y
#     true_train[val_index] = val_y
#     cv_scores.append(metrics.log_loss(val_y, pred_val_y))
# print("Mean cv score : ", np.mean(cv_scores))
# pred_full_test = pred_full_test / 10.
#
# # conf_repres_tr_y = true_train
# # conf_repres_pre_y = np.argmax(pred_train, axis=1)
# #
# # fig4 = plt.figure()
# # mat = confusion_matrix(conf_repres_tr_y, conf_repres_pre_y)
# # sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
# #             xticklabels=author_mapping_dict.keys(), yticklabels=author_mapping_dict.keys())
# # plt.xlabel('true label')
# # plt.ylabel('predicted label')
# # plt.show()
#
# train_df["cv_ngram1_4_char_eap"] = pred_train[:,0]
# train_df["cv_ngram1_4_char_hpl"] = pred_train[:,1]
# train_df["cv_ngram1_4_char_mws"] = pred_train[:,2]
# test_df["cv_ngram1_4_char_eap"] = pred_full_test[:,0]
# test_df["cv_ngram1_4_char_hpl"] = pred_full_test[:,1]
# test_df["cv_ngram1_4_char_mws"] = pred_full_test[:,2]



#-------------------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------Count Vectorizer char_b grams -----------------------------------------------------------------------------------------------------
author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
train_y = train_df['author'].map(author_mapping_dict)
cols_to_drop = ['id', 'text', 'pure_text', 'mean_word_len', 'pure_more_text','num_three_dots', 'num_exclam', 'num_dash']
target = 'author'
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,4), analyzer='char_wb')
full_tfidf = tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())


#for vector data sparce matrix
kf = model_selection.KFold(n_splits=6, shuffle=True, random_state=10)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
true_train = np.zeros([train_df.shape[0]])
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    true_train[val_index] = val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 6.

# conf_repres_tr_y = true_train
# conf_repres_pre_y = np.argmax(pred_train, axis=1)
#
# fig5 = plt.figure()
# mat = confusion_matrix(conf_repres_tr_y, conf_repres_pre_y)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=author_mapping_dict.keys(), yticklabels=author_mapping_dict.keys())
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()

train_df["cv_ngram1_4_char_wb_eap"] = pred_train[:,0]
train_df["cv_ngram1_4_char_wb_hpl"] = pred_train[:,1]
train_df["cv_ngram1_4_char_wb_mws"] = pred_train[:,2]
test_df["cv_ngram1_4_char_wb_eap"] = pred_full_test[:,0]
test_df["cv_ngram1_4_char_wb_hpl"] = pred_full_test[:,1]
test_df["cv_ngram1_4_char_wb_mws"] = pred_full_test[:,2]

#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------Tfid  Vectorizer char n-grams ----------------------------------------------------------------------------
author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
train_y = train_df['author'].map(author_mapping_dict)
cols_to_drop = ['id', 'text', 'pure_text', 'mean_word_len', 'pure_more_text','num_three_dots', 'num_exclam', 'num_dash']
target = 'author'
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

tfidf_vec = TfidfVectorizer(ngram_range=(1,7),analyzer = 'char')
full_tfidf =tfidf_vec.fit_transform(train_df['pure_more_text'].values.tolist()+test_df['pure_more_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['pure_more_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['pure_more_text'].values.tolist())
features = tfidf_vec.get_feature_names()

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
true_train = np.zeros([train_df.shape[0]])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=10)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    true_train[val_index] = val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# conf_repres_tr_y = true_train
# conf_repres_pre_y = np.argmax(pred_train, axis=1)
#
# fig6 = plt.figure()
# mat = confusion_matrix(conf_repres_tr_y, conf_repres_pre_y)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=author_mapping_dict.keys(), yticklabels=author_mapping_dict.keys())
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()

train_df["tv_ngram_1_4_char_eap"] = pred_train[:,0]
train_df["tv_ngram_1_4_char_hpl"] = pred_train[:,1]
train_df["tv_ngram_1_4_char_mws"] = pred_train[:,2]
test_df["tv_ngram_1_4_char_eap"] = pred_full_test[:,0]
test_df["tv_ngram_1_4_char_hpl"] = pred_full_test[:,1]
test_df["tv_ngram_1_4_char_mws"] = pred_full_test[:,2]
#
#
# n_comp = 200
# svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
# svd_obj.fit(full_tfidf)
# train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
# test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
#
# train_svd.columns = ['svd_char_' + str(i) for i in range(n_comp)]
# test_svd.columns = ['svd_char_' + str(i) for i in range(n_comp)]
# train_df = pd.concat([train_df, train_svd], axis=1)
# test_df = pd.concat([test_df, test_svd], axis=1)
#
# del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

#---------------------------------------------------------------HashingVectorizer ---------------------------------------------------------------------------------------

author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
train_y = train_df['author'].map(author_mapping_dict)
cols_to_drop = ['id', 'text', 'pure_text', 'mean_word_len', 'pure_more_text','num_three_dots', 'num_exclam', 'num_dash']
target = 'author'
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

tfidf_vec = HashingVectorizer(ngram_range=(1,3),analyzer = 'word',non_negative=True, stop_words='english')
full_tfidf =tfidf_vec.fit_transform(train_df['pure_more_text'].values.tolist()+test_df['pure_more_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['pure_more_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['pure_more_text'].values.tolist())


cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
true_train = np.zeros([train_df.shape[0]])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=10)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    true_train[val_index] = val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# conf_repres_tr_y = true_train
# conf_repres_pre_y = np.argmax(pred_train, axis=1)
#
# fig6 = plt.figure()
# mat = confusion_matrix(conf_repres_tr_y, conf_repres_pre_y)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=author_mapping_dict.keys(), yticklabels=author_mapping_dict.keys())
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()

train_df["hv_ngram_1_4_word_eap"] = pred_train[:,0]
train_df["hv_ngram_1_4_word_hpl"] = pred_train[:,1]
train_df["hv_ngram_1_4_word_mws"] = pred_train[:,2]
test_df["hv_ngram_1_4_word_eap"] = pred_full_test[:,0]
test_df["hv_ngram_1_4_word_hpl"] = pred_full_test[:,1]
test_df["hv_ngram_1_4_word_mws"] = pred_full_test[:,2]
#
#
# n_comp = 200
# svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
# svd_obj.fit(full_tfidf)
# train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
# test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
#
# train_svd.columns = ['svd_char_' + str(i) for i in range(n_comp)]
# test_svd.columns = ['svd_char_' + str(i) for i in range(n_comp)]
# train_df = pd.concat([train_df, train_svd], axis=1)
# test_df = pd.concat([test_df, test_svd], axis=1)
#
# del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

#-----------------------------------------------------------------Count Vectorizer pos tags n-grams -----------------------------------------------------------------------------------------------------
author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
train_y = train_df['author'].map(author_mapping_dict)
cols_to_drop = ['id', 'text', 'pure_text', 'mean_word_len', 'pure_more_text','num_three_dots', 'num_exclam', 'num_dash',"part_of_speech"]
target = 'author'
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

tfidf_vec = CountVectorizer()
full_tfidf = tfidf_vec.fit(train_df["part_of_speech"].values.tolist() + test_df["part_of_speech"].values.tolist())
train_tfidf = tfidf_vec.transform(train_df["part_of_speech"].values.tolist())
test_tfidf = tfidf_vec.transform(test_df["part_of_speech"].values.tolist())


#for vector data sparce matrix
kf = model_selection.KFold(n_splits=6, shuffle=True, random_state=10)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
true_train = np.zeros([train_df.shape[0]])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    true_train[val_index] = val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
pred_full_test = pred_full_test / 10.
print("Mean cv score : ", np.mean(cv_scores))

# conf_repres_tr_y = true_train
# conf_repres_pre_y = np.argmax(pred_train, axis=1)
#
# fig2 = plt.figure()
# mat = confusion_matrix(conf_repres_tr_y, conf_repres_pre_y)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=author_mapping_dict.keys(), yticklabels=author_mapping_dict.keys())
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()

train_df["cv_pos_eap"] = pred_train[:,0]
train_df["cv_pos_hpl"] = pred_train[:,1]
train_df["cv_pos_mws"] = pred_train[:,2]
test_df["cv_pos_eap"] = pred_full_test[:,0]
test_df["cv_pos_hpl"] = pred_full_test[:,1]
test_df["cv_pos_mws"] = pred_full_test[:,2]

#
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# #-----------------------------------------------------------------Tfid  Vectorizer dep  n-grams ----------------------------------------------------------------------------
# author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
# train_y = train_df['author'].map(author_mapping_dict)
# cols_to_drop = ['id', 'text', 'pure_text', 'mean_word_len', 'pure_more_text','num_three_dots', 'num_exclam', 'num_dash']
# target = 'author'
# train_X = train_df.drop(cols_to_drop+['author'], axis=1)
# test_X = test_df.drop(cols_to_drop, axis=1)
#
# tfidf_vec = TfidfVectorizer(ngram_range=(1,7))
# full_tfidf = tfidf_vec.fit(train_df["part_of_speech_dep"].values.tolist() + test_df["part_of_speech_dep"].values.tolist())
# train_tfidf = tfidf_vec.transform(train_df["part_of_speech_dep"].values.tolist())
# test_tfidf = tfidf_vec.transform(test_df["part_of_speech_dep"].values.tolist())
#
# cv_scores = []
# pred_full_test = 0
# pred_train = np.zeros([train_df.shape[0], 3])
# true_train = np.zeros([train_df.shape[0]])
# kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=10)
# for dev_index, val_index in kf.split(train_X):
#     dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
#     dev_y, val_y = train_y[dev_index], train_y[val_index]
#     pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
#     pred_full_test = pred_full_test + pred_test_y
#     pred_train[val_index,:] = pred_val_y
#     true_train[val_index] = val_y
#     cv_scores.append(metrics.log_loss(val_y, pred_val_y))
# print("Mean cv score : ", np.mean(cv_scores))
# pred_full_test = pred_full_test / 5.
#
# conf_repres_tr_y = true_train
# conf_repres_pre_y = np.argmax(pred_train, axis=1)
#
# # fig3 = plt.figure()
# # mat = confusion_matrix(conf_repres_tr_y, conf_repres_pre_y)
# # sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
# #             xticklabels=author_mapping_dict.keys(), yticklabels=author_mapping_dict.keys())
# # plt.xlabel('true label')
# # plt.ylabel('predicted label')
# # plt.show()
# train_df["pred_cv_tfidf_word_eap"] = pred_train[:,0]
# train_df["pred_cv_tfidf_word_hpl"] = pred_train[:,1]
# train_df["pred_cv_tfidf_word_mws"] = pred_train[:,2]
# test_df["pred_cv_t_tfidf_word_eap"] = pred_full_test[:,0]
# test_df["pred_cv_t_tfidf_word_hpl"] = pred_full_test[:,1]
# test_df["pred_cv_t_tfidf_word_mws"] = pred_full_test[:,2]
#
# #
# # n_comp = 200
# # svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
# # svd_obj.fit(full_tfidf)
# # train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
# # test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
# #
# # train_svd.columns = ['svd_word_' + str(i) for i in range(n_comp)]
# # test_svd.columns = ['svd_word_' + str(i) for i in range(n_comp)]
# # train_df = pd.concat([train_df, train_svd], axis=1)
# # test_df = pd.concat([test_df, test_svd], axis=1)
# #
# # del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
#
# #------------------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------Tfid  Vectorizer part of speech  n-grams ----------------------------------------------------------------------------
author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
train_y = train_df['author'].map(author_mapping_dict)
cols_to_drop = ['id', 'text', 'pure_text', 'mean_word_len', 'pure_more_text','num_three_dots', 'num_exclam', 'num_dash']
target = 'author'
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

tfidf_vec = TfidfVectorizer(ngram_range=(1,3))
full_tfidf = tfidf_vec.fit(train_df["part_of_speech"].values.tolist() + test_df["part_of_speech"].values.tolist())
train_tfidf = tfidf_vec.transform(train_df["part_of_speech"].values.tolist())
test_tfidf = tfidf_vec.transform(test_df["part_of_speech"].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
true_train = np.zeros([train_df.shape[0]])
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=10)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    true_train[val_index] = val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 10.

# conf_repres_tr_y = true_train
# conf_repres_pre_y = np.argmax(pred_train, axis=1)
#
# fig3 = plt.figure()
# mat = confusion_matrix(conf_repres_tr_y, conf_repres_pre_y)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=author_mapping_dict.keys(), yticklabels=author_mapping_dict.keys())
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()

train_df["tv_pos_eap"] = pred_train[:,0]
train_df["tv_pos_hpl"] = pred_train[:,1]
train_df["tv_pos_mws"] = pred_train[:,2]
test_df["tv_pos_eap"] = pred_full_test[:,0]
test_df["tv_pos_hpl"] = pred_full_test[:,1]
test_df["tv_pos_mws"] = pred_full_test[:,2]

#
# n_comp = 200
# svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
# svd_obj.fit(full_tfidf)
# train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
# test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
#
# train_svd.columns = ['svd_word_' + str(i) for i in range(n_comp)]
# test_svd.columns = ['svd_word_' + str(i) for i in range(n_comp)]
# train_df = pd.concat([train_df, train_svd], axis=1)
# test_df = pd.concat([test_df, test_svd], axis=1)
#
# del full_tfidf, train_tfidf, test_tfidf

#-----------------------------------------------------------------------------------------------------------------------------------------------------

#value more readable

cols_to_drop = ['id', 'text', 'pure_text', 'mean_word_len', 'pure_more_text','num_three_dots', 'num_exclam', 'num_dash',"part_of_speech"]

target = 'author'
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)


#subsample
param_test = [0.6,0.8,0.3]


kf = model_selection.KFold(n_splits=3, shuffle=True, random_state=17)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
params_data = {}

for keys in param_test:
    param_test1 = {'subsample':keys}
    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0,colsample=0.7,params=param_test1)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    #print("cv scores : ", cv_scores)
    params_data[keys]=np.mean(cv_scores)
    cv_scores.clear()
print(params_data)

pred_test_data = pd.DataFrame(pred_test_y)
pred_test_data['id'] = test_df['id']
pred_test_data = pred_test_data.set_index('id')
pred_test_data.columns = ['EAP', 'HPL', 'MWS']
pred_test_data.to_csv('subFinal.csv', index=True)

fig, ax = plt.subplots(figsize=(12,12))
xgb.plot_importance(model, max_num_features=100, height=0.8, ax=ax)
plt.show()
#train_tfidf  = hstack(((train_tfidf).toarray(),np.array(train_df['id'])[:,None]))
#f1 = pd.DataFrame(train_tfidf.toarray(), columns=tfidf_vec.get_feature_names())

'''
#for vector data sparce matrix
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=10)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_tfidf, seed_val=0)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    break


print(pred_test_data)
print("cv scores : ", cv_scores)
'''
