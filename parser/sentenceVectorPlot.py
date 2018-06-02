import matplotlib
import pandas as pd
from docutils.nodes import inline
from sentenceToVector import *
pd.options.mode.chained_assignment = None
import numpy as np
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from kmeansClusters import *
#%matplotlib inline


def tsne_plot(model,sentences):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    i = 0

    for sentence in sentences:
        tokens.append(findSentenceVector(model,sentence))
        labels.append(i)
        i = i+1

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    X = np.array(list(zip(x, y)))
    C_final = get_clusters(X, n=9)
    C = C_final['C']
    clusters = C_final['clusters']

    plt.figure(figsize=(16, 9))
    plt.style.use('ggplot')
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'maroon']
    for i in range(9):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        plt.scatter(points[:, 0], points[:, 1], s=7, c=colors[i%6])
    plt.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')

    plt.plot()
    plt.show()

#tsne_plot(getTrainedModel("doc2vec.model"),["It will work","It will work","IT WILL WORK"])