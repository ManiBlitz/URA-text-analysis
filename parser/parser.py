from textblob import *


#---Function to remove fillers from text

def removeFillers(text):

    # The goal of this function is to simplify the sentence to its most basic form
    # First we detect the different tags

    tagsRepository = TextBlob(text).tags

    # Then we have a list of tags that need to be removed from the sentence

    tagsBlackList = ["POS","CC","DT","TO"]

    # We then go through our list of words in our sentence and remove those verifying the tag block

    textSimplified = []
    for i in tagsRepository:
        if(i[1] not in tagsBlackList):
            textSimplified.append(i[0])

    # Then we join back our sentence and it is ready to use
    result = " "
    result = result.join(textSimplified)

    return result

#---Function to correct text

def setCorrectedWordSet(text):
    text = text.lower()
    wiki = TextBlob(text)
    words = wiki.split(sep=" ")
    parts = set(words)
    return parts

#---Function to get words lemma

def getLemmatizedList(wordCollection,text):

    # we define a separated lemmatization for words and verbs
    # We first extract the nouns in the word list

    nounsRep = TextBlob(text).noun_phrases

    # Then we lemmatize the sentence
    # while checking if the element belongs to the nouns list

    lemmatizedResult = []
    for i in wordCollection:
        if(i in nounsRep):
            lemmatizedResult.append(Word(i).lemmatize())
        else:
            lemmatizedResult.append(Word(i).lemmatize("v"))

    return lemmatizedResult

#---Function to remove synonyms duplicates


def sentenceRecollection(wordCollection,text):

    #First we need to lemmatize the different words in the collection

    word_base = getLemmatizedList(TextBlob(text.lower()).split(sep=" "),text)

    lemmatizedResult = getLemmatizedList(wordCollection,text)
    result = [item for item in word_base if item in lemmatizedResult]
    return result

#---final function

def ura_parser(text):
    stripedText = removeFillers(text)
    result = sentenceRecollection(setCorrectedWordSet(stripedText), stripedText)
    print(result)
    return result

#---Testing

text = "Scanner is not emailing scanned documents."
ura_parser(text)







