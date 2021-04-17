# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:34:12 2020

@author: hp 620 tx
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('data.tsv',delimiter='\t',quoting=3)
#quoting = 0 for quoting minimal values
#quoting = 1 for quoting all the values
#quoting = 2 for non-numerical values
#quoting = 3 for quoting none(considering no quotations in dataset)
# re- Regular Expression library (to remove unwanted dots and extra characters which wont be used for prediction)
# sub method to remove dots and consider only alphabets
# nltk library - Natural Language Tool Kit  
#download stopwords method from nltk (for present dataset only english stopwords)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords  #corpus - grouping the texts
from nltk.stem.porter import PorterStemmer
dataset['reviewText']=dataset['reviewText'].apply(str)
#tokenizer = nltk.RegexpTokenizer(r"\w+")
#new_words = tokenizer.tokenize(dataset['reviewText'])

cps=[] #empty list to append all the converted words
for i in range(0,982619):  #since there are 1000 reviews
    review=re.sub('[^a-zA-Z]',' ',dataset['reviewText'][i])
#Line 19 - importing regular expresion library
#line 26 - Taking Review variable and seperating values with giving spaces by considering only alphabets in Zeroth Row using sub method(to remove dots)
#line 30 - to convert into small letters
    review=review.lower()
    review=review.split() # convert string into list
#to remove stop words in one single sentence and to get the words which are not there in stopwords
    review=[word for word in review if not word in set(stopwords.words('english'))]
#stemming concepts - removes ed,ly...extensions of words 
    ps=PorterStemmer() #both stemming and lemmatisation takesplace at a time
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review) #to join all words which are in review with ' 'space in list
    cps.append(review)
from sklearn.feature_extraction.text import CountVectorizer #for creating bag of words - for counting different vectors which are there in particular list and making them as different columns and we will be counting all the values if the repetition values are there and noting them under particular column 
cv=CountVectorizer(max_features=982550) #we are having 1565 words in total i.e omitting 65 elements or attributes which are created and making it more precise about it
#x,y two arrays as input and output and then giving to ann model
#o's and 1's are just number of repetitions- - t can be any number that word is repeated
# Most of them are zeros here because they are not repeated in those sentences - its called as sparsity or sparse matrix - to remove such kind of dependencies parameter called max_features is used.
x=cv.fit_transform(cps).toarray()
y=dataset.iloc[:,-2].values
# Splitting data into training and testing parts
from sklearn.model_selection import train_test_split
x.shape
y.shape
y.reshape(-1,1)
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
#now using ANN model to predict
import keras
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(input_dim=1500,init="random_uniform",activation="sigmoid",output_dim=1000))#input layer
model.add(Dense(input_dim=100,init="random_uniform",activation="sigmoid",output_dim=100))#hidden layer
model.add(Dense(input_dim=1,init="random_uniform",activation="sigmoid",output_dim=1))#output layer
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])#adam-batch descent
model.fit(x_train,y_train,epochs=50000,batch_size=10)#epochs-number of iterations
y_pred=model.predict(x_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cn=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

model.save("nlpmodel.h5")
from keras.models import load_model
import numpy as np
import pickle
model=load_model("nlpmodel.h5")
with open('CountVectorizer','rb')as file:
    cv=pickle.load(file)
    entered_input="the book is very nice"
    inp=cv.transform([entered_input])
    y_pred=model.predict(inp)
    if(y_pred>0.5):
        print("it is a positive review")
    else:
        print("it is a negative review")
