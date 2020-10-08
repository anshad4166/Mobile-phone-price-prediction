
import flask
import pickle
import numpy as np
from flask import Flask, render_template, request

app=Flask(__name__)
loaded_model = pickle.load(open("model.pkl","rb"))

@app.route('/')
@app.route('/index')

def index():
    return flask.render_template('index.html')

@app.route('/result',methods = ['POST'])
def result():     
   int_features= [int(x) for x in request.form.values()] 
   final_features=[np.array(int_features)]
   #print(final_features)
   loaded_model = pickle.load(open("model.pkl","rb"))
   prediction=loaded_model.predict(final_features)
  
   return render_template('result.html',prediction=prediction)

if __name__ == '__main__':
	app.run()
