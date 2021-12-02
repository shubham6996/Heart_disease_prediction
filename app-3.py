from flask import Flask, render_template, url_for, request
import joblib
import os
import sklearn
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('hooomee.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = int(request.form["oldpeak"])
    st_slope = int(request.form["ST_slope"])
    fbs = float(request.form['fbs'])

    if sex == 1:
        sex_male = pd.DataFrame([[1]],columns = ['sex_male'])
    else:
        sex_male = pd.DataFrame([[0]],columns = ['sex_male'])
        
    if cp == 1:
        chest_pain = pd.DataFrame([[0,0,0]],columns = ['chest pain type_atypical chest pain',
       'chest pain type_non cardiac chest pain',
       'chest pain type_typical chest pain'])
    elif cp == 2:
        chest_pain = pd.DataFrame([[1,0,0]],columns = ['chest pain type_atypical chest pain',
       'chest pain type_non cardiac chest pain',
       'chest pain type_typical chest pain'])
    elif cp == 3:
        chest_pain = pd.DataFrame([[0,1,0]],columns = ['chest pain type_atypical chest pain',
       'chest pain type_non cardiac chest pain',
       'chest pain type_typical chest pain'])
    else:
        chest_pain = pd.DataFrame([[0,0,1]],columns = ['chest pain type_atypical chest pain',
       'chest pain type_non cardiac chest pain',
       'chest pain type_typical chest pain'])

    if restecg == 0:
        rest_ecg = pd.DataFrame([[0,0]],columns = ['resting ecg_normal',
       'resting ecg_st-t wave abnormality'])
    elif restecg == 1:
        rest_ecg = pd.DataFrame([[1,0]],columns = ['resting ecg_normal',
       'resting ecg_st-t wave abnormality'])
    else:
        rest_ecg = pd.DataFrame([[0,1]],columns = ['resting ecg_normal',
       'resting ecg_st-t wave abnormality'])

    if st_slope == 0:
        st_df = pd.DataFrame([[0,0]],columns = ['ST slope_flat',
       'ST slope_upsloping'])
    elif st_slope == 2:
        st_df = pd.DataFrame([[0,1]],columns = ['ST slope_flat',
       'ST slope_upsloping'])
    else:
        st_df = pd.DataFrame([[1,0]],columns = ['ST slope_flat',
       'ST slope_upsloping'])

    df1 = pd.DataFrame([[age,trestbps,chol,fbs,thalach,exang,oldpeak]],columns=['age', 'resting bp s', 'cholesterol', 'fasting blood sugar',
       'max heart rate', 'exercise angina', 'oldpeak'])
    x = pd.concat([df1, sex_male,chest_pain,rest_ecg,st_df], axis=1)
     
    print(age,sex,trestbps,chol,restecg,thalach,exang,cp,fbs)
    scaler_path = os.path.join(os.path.dirname(__file__), 'models_scaler.pkl')
    scaler = None
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    x[['age','resting bp s','cholesterol','max heart rate','oldpeak']] = scaler.fit_transform(x[['age','resting bp s','cholesterol','max heart rate','oldpeak']])    

    model_path = os.path.join(os.path.dirname(__file__), 'model_pkl')
    clf = joblib.load(model_path)

    with open(model_path, 'rb') as file:
        clf = pickle.load(file)
    
    y = clf.predict(x[['age', 'resting bp s', 'cholesterol', 'fasting blood sugar',
       'max heart rate', 'exercise angina', 'oldpeak', 'sex_male',
       'chest pain type_atypical chest pain',
       'chest pain type_non cardiac chest pain',
       'chest pain type_typical chest pain', 'resting ecg_normal',
       'resting ecg_st-t wave abnormality', 'ST slope_flat',
       'ST slope_upsloping']])
    print(y)

    # No heart disease
    if y == 0:
        return render_template('nodisease.html')

    # y=1,2,4,4 are stages of heart disease
    else:
        return render_template('heartdisease.html', stage=int(y))


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True, port="5000")
