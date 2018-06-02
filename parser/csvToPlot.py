from corpusBuilder import *
from sentenceToVector import *
from sentenceVectorPlot import *

#Our goal is to go from a CSV file to a plot of the different sentences on a graph
#To do so we already have a cleaned CSV dataset along with a model trainer
#we also have a plotting tool that will generate the necessary representation



colNames = ["ticketsID",
            "departmentid",
            "requesterid",
            "locationid",
            "ownedbyappuserif",
            "openedbyappuserid",
            "categoryid",
            "priorityid",
            "originid",
            "createdutc",
            "openedutc",
            "dueutc",
            "closedutc",
            "trainingflag",
            "visitrequestedflag",
            "serviceissueflag",
            "ishandled",
            "isexternal",
            "requiredoncallalert",
            "hasbeenescalated",
            "hasbeenreopened",
            "hasbeensurveyresponse",
            "reviewedbycloser",
            "createdbytmail",
            "createdbytouch",
            "createdbyiqmobile",
            "createdbyselfservice",
            "createdbybatchservice",
            "touchedbytouch",
            "touchedbyiqmobile",
            "touchedbyconnector",
            "touchedbyintelliteach",
            "isworklflowmaster",
            "isworkflowsecondary",
            "isbillable",
            "potentialproblemflag",
            "lockedbyappuserid",
            "lastescalatedutc",
            "summary"]

#first we define the file that will be analyzed
csvinput_fetch = readCSVinput("./texts/export.csv", colNames)
csvinput = csvinput_fetch['training']
csv_test = csvinput_fetch['test']

#then we clean the dataframe

cleaned_csv = clean_dataframe(csvinput)
cleaned_test = clean_dataframe(csv_test)

#from the corpus builder we keep the sentences and the corpus

corpus_build = build_corpus(cleaned_csv)
corpus = corpus_build['corpus']
sentences = build_corpus(cleaned_test)['sentences']

#We generate a model for the Doc2Vec alogrithm

generated_model = model_generator(corpus,getDocLabels())
it = generated_model['it']
model = generated_model['model']

#now we train the model and save it

model_trainer(model,it)

#Then we export the trained model for work

export_model = getTrainedModel('doc2vec.model')

#Finally we print the graph representing the different datas

tsne_plot(export_model, sentences)
