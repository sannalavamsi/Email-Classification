#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:55:02 2020

@author: vamsi
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle



app = Flask(__name__)
model = pickle.load(open('MNB_sm_model.pkl', 'rb'))
cv=pickle.load(open('transform.pkl','rb'))

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data)
		my_prediction = model.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)