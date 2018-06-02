#Import all the dependencies

import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import tokenize
from os import listdir
from os.path import isfile, join


#constants

tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))


def getDocLabels():
    #now create a list that contains the name of all the text file in your data #folder

    docLabels = []
    docLabels = [f for f in listdir("./texts/") if f.endswith(".txt")]
    return docLabels

#create a list data that stores the content of all text files in order of their names in docLabels

def getData(docLabels):
    data = []
    for doc in docLabels:
        data.append(open("./texts/" + doc).read())
    return data


#Assistive class for sentence labeling
class LabeledLineSentence(object):

    def __init__(self, doc_list, labels_list):

        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):

        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.LabeledSentence(doc, [self.labels_list[idx]])



#This function does all cleaning of data using two objects above

def nlp_clean(data):

   new_data = []
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data

#iterator returned over all documents
def model_generator(data, docLabels):
    data = nlp_clean(data)
    it = LabeledLineSentence(data, docLabels)
    model = gensim.models.Doc2Vec(vector_size=2, min_count=0, alpha=0.025, min_alpha=0.025)
    model.build_vocab(it)
    return {'model': model, 'it': it}

#training of model
def model_trainer(model,it):

    for epoch in range(10000):
        print("iteration "+str(epoch+1))
        model.train(it, total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    #saving the created model
    model.save('doc2vec.model')
    print("model saved")

#loading the model

def getTrainedModel(modelName):

    d2v_model = gensim.models.doc2vec.Doc2Vec.load(modelName)
    return d2v_model


#returning the vector of sentence

def findSentenceVector(model,sentence):
    parsed_data = sentence
    infered_vector = model.infer_vector(doc_words=parsed_data, steps=1000, alpha=0.025)
    return infered_vector