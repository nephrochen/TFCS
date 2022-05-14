import pickle
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import shap
import hashlib
import plotly.express as px
import plotly
import copy
import matplotlib.pyplot as plt
import joblib
#import xgboost as xgb

feature_mapping = {
    'smoker': "Smoking status",
    'cognitiveStatus2': "Cognitive status 2",    
    'elEscorialAtDx': "El Escorial category at diagnosis",
    'anatomicalLevel_at_onset': "Anatomical level at onset",
    'site_of_onset': "Site of symptom onset",
    'onset_side': "Onset side",
    'ALSFRS1': "ALSFRS-R part 1 score",
    'FVCPercentAtDx': "FVC% at diagnosis",
    'weightAtDx_kg': "Weight at diagnosis (kg)",
    'rateOfDeclineBMI_per_month': "Rate of BMI decline (per month)",
    'age_at_onset': "Age at symptom onset",
    'firstALSFRS_daysIntoIllness': "Time of first ALSFRS-R measurement (days from symptom onset)"
}

def prediction_page():
    st.markdown("""<style>.big-font {font-size:100px !important;}</style>""", unsafe_allow_html=True) 
    st.markdown(
        """<style>
        .boxBorder {
            border: 2px solid #990066;
            padding: 10px;
            outline: #990066 solid 5px;
            outline-offset: 5px;
            font-size:25px;
        }</style>
        """, unsafe_allow_html=True) 
    st.markdown('<div class="boxBorder"><font color="RED">Disclaimer: This predictive tool is only for research purposes</font></div>', unsafe_allow_html=True)
    st.write("## Model Perturbation Analysis")

    
    col1, col2, col3, col4 = st.columns(4)
    for i in range(0, len(categorical_columns), 4):
        with col1:
            if (i+0) >= len(categorical_columns):
                continue
            c1 = categorical_columns[i+0] 
            f1 = st.selectbox()

        with col2:
            if (i+1) >= len(categorical_columns):
                continue
            c2 = categorical_columns[i+1] 
            f2 = st.selectbox()
        with col3:
            if (i+2) >= len(categorical_columns):
                continue
            c3 = categorical_columns[i+2] 
            f3 = st.selectbox()
        with col4:
            if (i+3) >= len(categorical_columns):
                continue
            c4 = categorical_columns[i+3] 
            f4 = st.selectbox()
    