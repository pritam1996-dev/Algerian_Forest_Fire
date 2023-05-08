import pickle
from flask import Flask , jsonify , request , render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler




app = Flask(__name__)

# import lasso regression and import standardscaler pickle file

lasso_model= pickle.load(open('models/lasssso.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler_algerian_Forest.pkl','rb'))

# Route for home page

@app.route('/')

def index():
    
    return render_template('index.html')



@app.route('/predictdata' , methods = ['GET','POST'])

def predict_datapoint():

    if request.method == 'POST':

        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region  = float(request.form.get('Region'))

        # Transform the test data

        new_scaled_data = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

       
        # prediction on test data

        result = lasso_model.predict(new_scaled_data)


        return render_template('home.html' , result = result[0])
    
    
    else:

        return render_template('home.html')
    



if __name__ == "__main__":
    
    app.run(host = "0.0.0.0")