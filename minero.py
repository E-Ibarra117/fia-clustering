
import re
import os
import string
import pprint
import nltk
from nltk.tokenize import word_tokenize
import spacy
import pandas as pd
from collections import Counter
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
    documents[i] = re.sub("á" , 'a', documents[i])
    documents[i] = re.sub("é" , 'e', documents[i])
    documents[i] = re.sub("í" , 'i', documents[i])
    documents[i] = re.sub("ó" , 'o', documents[i])
    documents[i] = re.sub("ú" , 'u', documents[i])
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
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)



true_k = 6
model = KMeans(n_clusters=true_k).fit(X)
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

wcss=[]
for i in range(1,11):
    kmeans =KMeans(n_clusters=i).fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()

print("Terminos relevantes de cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

kmean_indices = model.fit_predict(X)
pca = PCA(2)
scatter_plot_points = pca.fit_transform(X.toarray())

colors = ["r", "b", "c", "y", "m" ,"g"]
x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]
fig, ax = plt.subplots(figsize=(20,10))

ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])
print(scatter_plot_points)
plt.show()
