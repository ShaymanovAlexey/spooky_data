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


PATH_TRAIN = './input/train.csv'
PATH_TEST = './input/test.csv'

fl = codecs.open(PATH_TRAIN, 'r','utf-8')
l = fl.readline()

#find semicolon and comma in samples

part_speech_dict = {}

names_frequntly = {}

most_frequently_pattern = {}

list_of_names = []

NGRAMS = 3

COUNT_GRAMS = 100

cur_time = time.time()

def find_symbols(line):
    return sum([1 for x in line if x in [',']]), sum([1 for x in line if x in [';']]),sum([1 for x in line if x in ['?']])

spookyData = pd.DataFrame({'Id': ['id26305','id17569','id11008'], 'comma_freq': 0, 'semicolon_freq': 0,'question':0, 'author':""})


for index_line, line in enumerate(fl):
    # разбиваем строчку на три строковые переменные

    Id, Sample, Author = line.strip().split('","')
    _,Id = Id.strip().split("\"")
    Id = str(Id)
    comma_count, sem_count, quest_count = find_symbols(Sample)
    list_colums = list(spookyData.columns)
    Sample = re.sub('\"', '', Sample)
    # define part of speech

    #create new raw in table
    spookyData.loc[index_line] = 0

    Sample = Sample.lower()

    if most_frequently_pattern.get(Author) == None:
        most_frequently_pattern[Author] = []

    most_frequently_pattern[Author].append(Sample)



    for names_check in Sample.split():
        if not eng_dict.check(names_check) and names_check[-1] not in[',',';',':','"', '?','\'','.' , 'Æ',  'Å',  'ç',  'ἶ'] and names_check[0] not in[',',';',':','"', '?','\'','.', 'Æ',  'Å',  'ç',  'ἶ']:
            if (eng_dict.suggest(names_check) != [] and eng_dict.suggest(names_check)[0][0].islower()) or (len(names_check) >3 and names_check[-3:] == 'ish' or names_check[-2:] == 'in' ):
                continue
            else:
                #print(names_check)
                if names_check not in list_colums:
                    spookyData[names_check] = 0
                    spookyData.loc[index_line,names_check] = 1
                    list_of_names.append(names_check)
                else:
                    spookyData.loc[index_line, names_check] += 1

    spookyData.loc[index_line, 'Id'] = Id
    spookyData.loc[index_line, 'comma_freq']= comma_count
    spookyData.loc[index_line, 'semicolon_freq']= sem_count
    spookyData.loc[index_line, 'question']=  quest_count
    spookyData.loc[index_line, 'author']=  Author

spookyData.loc[:,(spookyData != 0).any(axis=0)]
file_name = 'spookyData'
spookyData.to_csv(file_name, encoding='utf-8')
print("all_merged")
sys.exit()


spend_time = -cur_time  + time.time()
print("time spent ", int(spend_time)/60, spend_time%60)

l_most_grams = {}

keys_author_head = []

for keys_author in most_frequently_pattern.keys():
    text_author  = ' '.join(str(e) for e in most_frequently_pattern[keys_author])
    text_author = re.sub('\.', ' ', text_author)
    tokenized = text_author.split()
    esThreeGrams = ngrams(tokenized, NGRAMS)
    esThreeFreq = collections.Counter(esThreeGrams)
    l_most_freq = esThreeFreq.most_common(COUNT_GRAMS)
    l_most_freq = [l1[0] for l1 in l_most_freq]
    l_most_grams[keys_author] = l_most_freq
    keys_author_head.append(keys_author)
    print("keys_author", keys_author)

for data_list in l_most_grams[keys_author_head[1]]:
        if (data_list in l_most_grams[keys_author_head[0]]) and (data_list in l_most_grams[keys_author_head[2]]):
            l_most_grams[keys_author_head[0]].remove(data_list)
            l_most_grams[keys_author_head[1]].remove(data_list)
            l_most_grams[keys_author_head[2]].remove(data_list)
        elif data_list in l_most_grams[keys_author_head[0]]:
            l_most_grams[keys_author_head[0]].remove(data_list)
            l_most_grams[keys_author_head[1]].remove(data_list)
        elif data_list in l_most_grams[keys_author_head[2]]:
            l_most_grams[keys_author_head[2]].remove(data_list)
            l_most_grams[keys_author_head[1]].remove(data_list)



for data_list in l_most_grams[keys_author_head[1]]:
        if (data_list in l_most_grams[keys_author_head[0]]) and (data_list in l_most_grams[keys_author_head[2]]):
            l_most_grams[keys_author_head[0]].remove(data_list)
            l_most_grams[keys_author_head[1]].remove(data_list)
            l_most_grams[keys_author_head[2]].remove(data_list)

        elif data_list in l_most_grams[keys_author_head[0]]:
            l_most_grams[keys_author_head[0]].remove(data_list)
            l_most_grams[keys_author_head[1]].remove(data_list)

        elif data_list in l_most_grams[keys_author_head[2]]:
            l_most_grams[keys_author_head[2]].remove(data_list)
            l_most_grams[keys_author_head[1]].remove(data_list)


for data_list in l_most_grams[keys_author_head[1]]:
        if (data_list in l_most_grams[keys_author_head[0]]) and (data_list in l_most_grams[keys_author_head[2]]):
            l_most_grams[keys_author_head[0]].remove(data_list)
            l_most_grams[keys_author_head[1]].remove(data_list)
            l_most_grams[keys_author_head[2]].remove(data_list)

        elif data_list in l_most_grams[keys_author_head[0]]:
            l_most_grams[keys_author_head[0]].remove(data_list)
            l_most_grams[keys_author_head[1]].remove(data_list)

        elif data_list in l_most_grams[keys_author_head[2]]:
            l_most_grams[keys_author_head[2]].remove(data_list)
            l_most_grams[keys_author_head[1]].remove(data_list)



def jaccard_distance(a, b):
    """Calculate the jaccard distance between sets A and B"""
    a = set(it for x_t in a for it in x_t)
    b = set(b)
    return 1.0 * len(a & b) / len(a | b)

author_all_grams = []

for author_gram in l_most_grams[keys_author_head[0]]:
    author_gram = str(author_gram)
    spookyData[author_gram]= 0
    author_all_grams.append(author_gram)

for author_gram in l_most_grams[keys_author_head[1]]:
    author_gram = str(author_gram)
    spookyData[author_gram] = 0
    author_all_grams.append(author_gram)

for author_gram in l_most_grams[keys_author_head[2]]:
    author_gram = str(author_gram)
    spookyData[author_gram] = 0
    author_all_grams.append(author_gram)

print("author_all_grams",author_all_grams)

kl = codecs.open(PATH_TRAIN, 'r','utf-8')
l = kl.readline()
#add new features with n-grams
for index_line, line in enumerate(kl):
    # break into 3-grams
    Id, Sample, Author = line.strip().split('","')
    _,Id = Id.strip().split("\"")
    Id = str(Id)
    #remove all " symbols
    Sample = re.sub('\"', '', Sample)
    # define part of speech

    #create new raw in table

    Sample = Sample.lower()

    tokenized = Sample.split()
    esThreeGrams = list(ngrams(tokenized, NGRAMS))

    for author_gram in author_all_grams:
        s1 = jaccard_distance(esThreeGrams,author_gram)
        if type(author_gram)!=str:
            print("boom")
            author_gram = str(author_gram)
        print(author_gram)
        print(esThreeGrams)
        spookyData.loc[index_line,author_gram] = s1
        print(index_line,spookyData.loc[index_line,author_gram])

# for train data

spookyTest = pd.DataFrame({'Id': ['id26305','id17569','id11008'], 'comma_freq': 0, 'semicolon_freq': 0,'question':0})

for names in list_of_names:
    spookyTest[names] = 0

for grams in author_all_grams:
    spookyTest[grams] = 0


fl = codecs.open(PATH_TEST, 'r','utf-8')
l = fl.readline()

for index_line, line in enumerate(fl):
    # разбиваем строчку на три строковые переменные

    Id, Sample= line.strip().split('","')
    _,Id = Id.strip().split("\"")
    Id = str(Id)
    comma_count, sem_count, quest_count = find_symbols(Sample)
    list_colums = list(spookyTest.columns)
    #remove all " symbols
    Sample = re.sub('\"', '', Sample)
    # define part of speech

    #create new raw in table
    spookyTest.loc[index_line] = 0

    Sample = Sample.lower()

    for names_check in Sample.split():
        if names_check in list_of_names:
            spookyTest.loc[index_line, names_check] +=1

    spookyTest.loc[index_line, 'Id'] = Id
    spookyTest.loc[index_line, 'comma_freq'] = comma_count
    spookyTest.loc[index_line, 'semicolon_freq'] = sem_count
    spookyTest.loc[index_line, 'question'] = quest_count


    tokenized = Sample.split()
    esThreeGrams = list(ngrams(tokenized, NGRAMS))

    for n_grams in author_all_grams:
        s1 = jaccard_distance(esThreeGrams, n_grams)
        spookyTest.loc[index_line, author_gram] = s1

spend_time = -cur_time  + time.time()

print("time all spent ", int(spend_time)/60, spend_time%60)

spookyData.loc[:, (spookyData != 0).any(axis=0)]
spookyTest.loc[:, (spookyTest != 0).any(axis=0)]


file_name = 'spookyData'
spookyData.to_csv(file_name, encoding='utf-8')

file_name = 'spookyTest'
spookyTest.to_csv(file_name, encoding='utf-8')

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
predict = [x for x in spookyData.columns if x not in ['index', 'Id', 'author']]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
target = 'author'

X_train, X_test, y_train, y_test = train_test_split(spookyData[predict], spookyData[target], test_size = 0.2, random_state = 10)

knn.fit(X_train,y_train)
dtrain_predict = knn.predict(X_test)
dtrain_predprob = knn.predict_proba(X_test)


#Print model report:
print("\nModel Report")
#print("AUC Score (Train): %f" % roc_auc_score(y_test, dtrain_predprob))
print("Accuracy : %.4g" % accuracy_score(y_test.values, dtrain_predict))
#print("AUC Score (Train): %f" % roc_auc_score(y_test, dtrain_predprob))

#print(spookyTest)