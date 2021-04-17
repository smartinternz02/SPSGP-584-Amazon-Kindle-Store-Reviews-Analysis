# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:17:32 2020

@author: hp 620 tx
"""
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph #global graph
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


from flask import Flask,request,render_template # request is post in html page
from keras.models import load_model
import numpy as np
global model,graph #global variables
import tensorflow as tf
graph = tf.get_default_graph()
model = load_model('nlpmodel.h5')
app = Flask(__name__)
@app.route('/')#whenever browser finds localhost:5000 then execute below function 
def home(): 
    return render_template('index.html') # this function is returning index.html file
@app.route('/login',methods=['POST']) # when you click submit on html page it is redirecting to this url.
def login():  # as soon as this url is directed then call this function
    A=request.form['a'] # from html page whatever the text is typed that is requested from the 'form' functionality and is stored in a main variable 
    B=request.form['b']
    total = [[a,b]]
    with graph.as_default():
        ypred=model.predict(np.array(total))
        print(ypred)
        y=ypred[0][0]
    E=str(A)+str(B)+str(C)+str(D)
    return render_template('index.html',abc = y) # after typing the name, show this name on index.html file where we have created variable abc
if __name__ == '__main__':
    app.run(debug = True)