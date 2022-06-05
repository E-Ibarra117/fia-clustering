
import re
import os
import string
import pprint
import nltk
import spacy
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import cosine_similarity


path='colombia'
documents = []
titles=[]
a=[]
dirs = os.listdir(path)
   
spacy
nlp = spacy.load('es_core_news_md')
for doc in dirs:
       if doc.endswith('.txt'):
           titles.append(doc)
           f=open(os.path.join(path,doc),encoding="utf8")
           words = f.read()
           documents.append(words)
           
           f.close()

regex = r'[' + string.punctuation + ']' 
for i in range(len(documents)): 
    documents[i] = documents[i].lower()
    documents[i] = re.sub('http\S+', ' ', documents[i])
    documents[i] = re.sub("°", ' ', documents[i])
    documents[i] = re.sub(regex , ' ', documents[i])
    # documents[i] = re.sub("á" , 'a', documents[i])
    # documents[i] = re.sub("é" , 'e', documents[i])
    # documents[i] = re.sub("í" , 'i', documents[i])
    # documents[i] = re.sub("ó" , 'o', documents[i])
    # documents[i] = re.sub("ú" , 'u', documents[i])
    documents[i] = re.sub("\d+", ' ', documents[i])
    documents[i] = re.sub("\\s+", ' ', documents[i])
    doc=nlp(documents[i])
    words = [token.text
         for token in doc
         if (not token.is_stop and
             not token.is_punct or
             (token.pos_ == "NOUN" or token.pos_ == "VERB" or token.pos_ == "ADJ"))]
    words=' '.join(words)
    documents[i]=words


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
dist = 1 - cosine_similarity(X)


#kmeans

#agrupacion
true_k = 4
model = KMeans(n_clusters=true_k).fit(X)
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

#codo de jambu
# wcss=[]
# for i in range(1,11):
#     kmeans =KMeans(n_clusters=i).fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1,11),wcss)
# plt.show()

#terminos relevantes
# print("Terminos relevantes de cluster:")
# order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names()
# for i in range(true_k):
#     print("Cluster %d:" % i),
#     for ind in order_centroids[i, :10]:
#         print(' %s' % terms[ind]),
#     print

#visualizacion
from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
kmean_indices = model.fit_predict(X)
scatter_plot_points = mds.fit_transform(dist)
colors = ["r", "b", "c", "y", "m" ,"g"]
x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]
fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])


#documentos y el cluster al que pertenecen
#falta


#jerarquico
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

#agrupacion
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
cluster.fit_predict(dist)


#visualizacion

kmean_indices = np.unique(cluster)
scatter_plot_points = mds.fit_transform(dist)
colors = ["r", "b", "c", "y", "m" ,"g"]
x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]
fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(x_axis, y_axis, c=cluster.labels_)
plt.show()




