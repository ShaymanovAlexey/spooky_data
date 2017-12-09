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

NGRAMS = 4

COUNT_GRAMS = 100



spookyData = pd.read_csv('spookyData')

del spookyData['Unnamed: 0']



l_most_grams = {}

keys_author_head = []


for index_line, line in enumerate(fl):
    # разбиваем строчку на три строковые переменные

    Id, Sample, Author = line.strip().split('","')

    Sample = re.sub('\"', '', Sample)
    # define part of speech

    Sample = Sample.lower()

    if most_frequently_pattern.get(Author) == None:
        most_frequently_pattern[Author] = []

    most_frequently_pattern[Author].append(Sample)



for keys_author in most_frequently_pattern.keys():

    text_author  = ' '.join(str(e) for e in most_frequently_pattern[keys_author])
    text_author = re.sub('\.', ' ', text_author)

    # define part of speech
    text = nltk.word_tokenize(text_author)
    speech_data = nltk.pos_tag(text)
    speech_data = [x for _, x in speech_data]
    speech_dat = [tuple(speech_data[i:i + NGRAMS]) for i in range(len(speech_data) - NGRAMS + 1)]
    esThreeFreq = collections.Counter(speech_dat)
    l_most_freq = esThreeFreq.most_common(COUNT_GRAMS)
    l_most_freq = [l1[0] for l1 in l_most_freq]
    l_most_grams[keys_author] = l_most_freq
    keys_author_head.append(keys_author)


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



cur_time = time.time()

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
    Author = re.sub('\"', '', Author)
    Sample = Sample.lower()
    spookyData.loc[index_line, 'author'] = Author
    # define part of speech
    text = nltk.word_tokenize(Sample)
    speech_data = nltk.pos_tag(text)
    speech_data = [x for _, x in speech_data]
    speech_dat = [tuple(speech_data[i:i + NGRAMS]) for i in range(len(speech_data) - NGRAMS + 1)]
    #create new raw in table

    esThreeGrams = speech_dat

    for author_gram in author_all_grams:
        s1 = jaccard_distance(esThreeGrams,author_gram)
        spookyData.loc[index_line,author_gram] = float(s1)
        #print(s1, esThreeGrams,author_gram)


print("last for  ", time.time()-cur_time)

spookyData.loc[:,(spookyData != 0).any(axis=0)]
file_name = 'spookyDataModSpeech'
spookyData.to_csv(file_name, encoding='utf-8')
print("all_merged")
sys.exit()