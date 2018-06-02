#The goal of this file is to convert a CSV file to an usable corpus
#To do so we will use the export.csv file previously cleaned
#Then clean its content and use it as base or the training model doc2vec.model

from sentenceToVector import *
import re
import nltk
from gensim.models import word2vec
import pandas as pd

#Function to get the content of the CSV file
def readCSVinput(filePath,colNames,number_sample=100000, chunksize=500000):
    data = pd.read_csv(filePath, delimiter="~",header=None, names=colNames,
    quoting=0, encoding='utf8', low_memory=False, chunksize=chunksize)
    i = 0
    dict = {'training': None, 'test': None}
    for chunk in data:
        if(i == 0):
            dict['training'] = chunk.sample(chunksize)
            i = i+1
        if(i == 1):
            dict['test'] = chunk.sample(number_sample, random_state=23)
        break

    return dict


STOP_WORDS = nltk.corpus.stopwords.words()


def clean_sentence(val):
    #remove chars that are not letters or numbers, downcase, then remove stop words
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence


def clean_dataframe(data):
    #drop nans, then apply 'clean_sentence' function to summary
    data = data.dropna(how="any")

    for col in ['summary']:
        data[col] = data[col].apply(clean_sentence)

    return data


def build_corpus(data):
    #Creates a list of lists containing words from each sentence"
    corpus = []
    sentences = []
    for col in ['summary']:
        for sentence in data[col].iteritems():
            word_list = re.findall(r"[\w']+", sentence[1])
            sentences.append(word_list)
            for i in word_list:
                corpus.append(i)

    corpus = [" ".join(corpus)]

    return {'corpus':corpus,'sentences':sentences}