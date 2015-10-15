# TopTermsInLabeldData.py,  Eric Titus,  October 1, 2015
# read in data with labels assigned, and return top phrases per brand

from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
import sys
from time import time

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from nltk.tokenize import TweetTokenizer,word_tokenize, sent_tokenize
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# %matplotlib inline

# CSV read in function to look at 3 data files combined
# Read in data and combine.
def csvReadInLabeled(filestring):
  header_names=['Date','Handle','Source','Content','Brand']
  return(pd.read_csv(filestring,infer_datetime_format=True,
          parse_dates=[0],names=header_names,
          skiprows=[0],encoding='utf-8',
          dayfirst=True,
          dtype={'Content': object},
          na_values={'Content',''}))

data1=csvReadInLabeled('NewData1.csv')
# data1a=data1[:10000].copy()
# data1.head()
overallData=data1

#plot to show what sources data came from
overallData.Handle.groupby(data1.Source).count().plot(kind='bar')
plt.show()

# function to split entries with multiple brand names into 2 entries
def splitBrand(df):
    df['BrandList']=df.Brand.str.split(',')
    df=df.drop('Brand',1)
    return(df)

overallData=splitBrand(overallData) #run split brands on data

#combine all text data from each brand, then put into a dict,
# where the brand is the key, and the text is the value

brandDict={}
for idx,ser in overallData[['BrandList','Content']].iterrows():
    for brand in ser['BrandList']:
        brandStripped=brand.strip()
        if brandStripped in brandDict:
            brandDict[brandStripped]=brandDict[brandStripped]+' '+ser['Content']
        else:
            brandDict[brandStripped]=ser['Content']
type(brandDict.keys())

#pull stopwords
import codecs
f = codecs.open('stopwordList.txt',
   encoding='utf-8')

stoplist=[];
for line in f:
	stoplist.append(unicode(line)[:-1])
f.close()

# make overall document to tokenize/vectorize
brands=[]
documents=[]
for k, v in brandDict.items():
    documents.append(v.lower())
    brands.append(k)

def tknzrTweet(txt):
    tknzr=TweetTokenizer(strip_handles=True, reduce_len=True)
    return(tknzr.tokenize(txt))

def tknzrStd(txt):
# '''Tokenizes into sentences, then strips punctuation/abbr, converts to lowercase and tokenizes words'''
    return [word_tokenize(" ".join(re.findall(u'\w+', t,flags = re.UNICODE | re.LOCALE)).lower()) 
            for t in sent_tokenize(txt.replace("'", ""))]

# Main tokenization function
def fullTokDict(documents,stopwords):
    docOut=[];
    for idx in np.arange(len(documents)):
        if type(documents[idx])!=unicode:
            tempDoc=[u'']
        else:
            tempDoc=tknzrTweet(documents[idx])
            tempDoc2=[]
        for word in tempDoc:
            if word not in stopwords:
                print
                tempDoc2.append(
                    re.sub(u'^https?:\/\/.*[\r\n]*|^\w\s|\d|^\W\s|\W{2,10}|:.+|^\w\s','',word))
        tempDoc2=filter(None,tempDoc2)
        docOut.append(tempDoc2)
    return(docOut)

# tokenize data
procDoc=fullTokDict(documents,stoplist)

# define vectorizer paramaters. This turns the data into
# a TFIDF vectorized array
vectorizer = TfidfVectorizer(max_df=0.7, #max_features=300,
                                 min_df=2,
                                 use_idf=True,
                                 encoding='utf-8',
                                 tokenizer=lambda i:i,
                                 lowercase=False,
                                 ngram_range=(1,5),
                                 sublinear_tf=False
                                 )
X = vectorizer.fit_transform(procDoc) #vectorize data

n_docs = X.shape[0]
tfidftables = [{} for _ in xrange(n_docs)]
terms = vectorizer.get_feature_names()

for i, j in zip(*X.nonzero()):
    tfidftables[i][terms[j]] = X[i, j]

# get feature names, then weight ngrams and hashtags,
# while removing single, non # terms.
ftNames=vectorizer.get_feature_names()
for idx in np.arange(len(ftNames)):
    if (len(ftNames[idx].split())==1) & (ftNames[idx][0]!=u'#'):
        for row in X[:,idx].nonzero()[0]:
            X[row,idx]=0
    else:
        for row in X[:,idx].nonzero()[0]:
            X[row,idx]*=(1+(len(ftNames[idx].split())*0.1))

# make plot showing how many brands are represented
df=pd.DataFrame(tfidftables, index=brands)
data=df.T
data.count().plot(kind='bar')

# This loops picks out the top terms for each brand and
# saves them to a file.

brandList=[]
f=open('TopPostsInEachBrand.txt','w')
for brand in brands:
    testSort=data.sort(columns=[brand],ascending=False)
    topWords=testSort[brand][0:20].index

    string='top terms in brand: '+brand +'\n'
    f.write(string.encode('utf-8'))
    for idx in np.arange(20):
      string=unicode(testSort[brand])+'\n'
      f.write(string.encode('utf-8'))

    brandlistStr=''
    for word in topWords:
        brandlistStr=brandlistStr+word
    brandList.append(brandlistStr)
f.close()

# Take the top phrases for each brand, and display a wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud
for brand in brandList:
    wordcloud = WordCloud().generate(brand)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()