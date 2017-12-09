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

#import only English dictionary
eng_dict = enchant.Dict("en_US")


spookyData = pd.read_csv('spookyDataModSpeech')

del spookyData['Unnamed: 0']

spookyData['author'] = spookyData['author'].map(lambda x: re.sub("\"","",x))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
predict = [x for x_index,x in enumerate(spookyData.columns) if x not in ['index', 'Id', 'author']]
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from matplotlib.pyplot import  colormaps as cm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier

class XgboostHelperFunc:
    def __init__(self, params = None, metrics = 'neg_mean_squared_error'):
        self.xgboo = XGBClassifier(**params)
        self.metrics = metrics

    @property
    def metrics(self):
        return self.__metrics

    @metrics.setter
    def metrics(self, metrics):
        self.__metrics = metrics

    def train_foo(self, x_train, y_train):
        self.xgboo.fit(x_train,y_train)

    def fit_foo(self, x, y):
        return self.xgboo.fit(x,y)

    def predict_foo(self, x):
        return self.xgboo.predict(x)

    def show_feature_importance(self, x, y):
        map1 = self.xgboo.fit(x,y).booster().get_score(importance_type='weight')
        plt.bar(range(len(map1)), map1.values(), color='b')
        plt.xticks(range(len(map1)), map1.keys(), rotation=25)
        plt.show()

    def tuning_grid_cv(self, param_test, x_train, y_train):
        gsearch = GridSearchCV(estimator = self.xgboo, param_grid = param_test, n_jobs = -1, scoring = self.metrics, iid=False)
        gsearch.fit(x_train, y_train)
        print(gsearch.best_score_)




param_test = {'learning_rate' :0.1, 'n_estimators' :40, 'max_depth':1, 'min_child_weight':3, 'seed' :27}

model1 = XgboostHelperFunc(params=param_test)

target = 'author'
cur_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(spookyData[predict], spookyData[target], test_size = 0.2, random_state = 10)
print(y_test)

model1.train_foo(X_train,y_train)
dtrain_predict = model1.predict_foo(X_test)

spend_time = -cur_time  + time.time()
print(model1.show_feature_importance)
print("spend time ", spend_time)
#Print model report:
print("\nModel Report")
#print("AUC Score (Train): %f" % roc_auc_score(y_test, dtrain_predprob))
print("Accuracy : %.4g" % accuracy_score(y_test.values, dtrain_predict))
