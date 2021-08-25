import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

st.write("""
# Mortality Prediction Model for Patients with Liver Disease
""")

st.sidebar.header('Input Patient Status')

def pat_status():
    temp = st.sidebar.slider('Temperature', 95.0, 102.0, 110.0)
    heart = st.sidebar.slider('Heart-Rate', 85.0, 150.0, 200.0)
    resp = st.sidebar.slider('Respiratory-Rate', 0.0, 50.0, 100.0)
    sato2 = st.sidebar.slider('O2-Saturation', 90.0, 95.0, 100.0)
    sbp = st.sidebar.slider('SBP', 50.0, 100.0, 200.0)
    dbp = st.sidebar.slider('DBP', 50.0, 100.0, 200.0)
    alt = st.sidebar.slider('ALT', 5.0, 30.0, 100.0)
    alp = st.sidebar.slider('ALP', 5.0, 50.0, 200.0)
    ast = st.sidebar.slider('AST', 5.0, 50.0, 100.0)
    alb = st.sidebar.slider('Albumin', 1.0, 5.0, 10.0)
    ggt = st.sidebar.slider('GGT', 1.0, 110.0, 300.0)

    data = {'Temperature': temp,
            'Heart Rate': heart,
            'Respiratory Rate': resp,
            'O2 Saturation': sato2,
            'SBP': sbp,
            'DBP': dbp,
            'ALT': alt,
            'ALP': alp,
            'AST': ast,
            'Albumin': alb,
            'GGT': ggt}
    features = pd.DataFrame(data, index=[0])
    return features

df = pat_status()

st.subheader('Patient Status')
st.write(df)

filename = 'model/tune_xgb_model.model'
# clf = pickle.load(open(filename, 'rb'))
clf = XGBClassifier()
clf.load_model(filename)

def preprocessing_input(df):
    df = df.copy()

    sc = StandardScaler()
    sc.fit(df)

    test = pd.DataFrame(sc.transform(df), index=df.index, columns=df.columns)

    return test

test = preprocessing_input(df)

pred = clf.predict(test)
pred_prob = clf.predict_proba(test)

st.subheader('Prediction')
st.write(pred)

st.subheader('Prediction Probability')
st.write(pred_prob)
