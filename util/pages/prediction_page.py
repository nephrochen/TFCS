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
categorical_columns =['gender',
 'height',
 'weight',
 'sk',
 'age',
 'DM',
 'HT',
 'HL',
 'COPD',
 'PVD',
 'CVA',
 'NYHA',
 'ART',
#  'AF',
#  'MI',
#  'PCI_his',
#  'cs_SG',
 'Crea',
 'TC',
 #'LDL_C',
 'Glu',
 'ef',
 #'lvidd',
 #'la',
 'CBP_t',
 #'ACC_t',
 'IABP',
]


numerical_columns = []
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
            f1 = st.selectbox(categorical_columns[1],categorical_columns)

        with col2:
            if (i+1) >= len(categorical_columns):
                continue
            c2 = categorical_columns[i+1] 
            f2 = st.selectbox(categorical_columns[2],categorical_columns)
        with col3:
            if (i+2) >= len(categorical_columns):
                continue
            c3 = categorical_columns[i+2] 
            f3 = st.selectbox(categorical_columns[3],categorical_columns)
        with col4:
            if (i+3) >= len(categorical_columns):
                continue
            c4 = categorical_columns[i+3] 
            f4 = st.selectbox(categorical_columns[4],categorical_columns)
    
    for col in numerical_columns:
        X_new[col] = X_new[col].map(lambda x: float(x) if not x=='Not available' else np.nan)
    for i in range(0, len(numerical_columns), 4):
        with col1:
            if (i+0) >= len(numerical_columns):
                continue
            c1 = numerical_columns[i+0] 
            idx = X_new.loc[select_patient, c1]
            f1 = st.number_input("{}".format(feature_mapping[c1]), min_value=X_new[c1].min(),  max_value=X_new[c1].max(), value=idx)
            new_feature_input[c1].append(f1)
        with col2:
            if (i+1) >= len(numerical_columns):
                continue
            c2 = numerical_columns[i+1] 
            idx = X_new.loc[select_patient, c2]
            f2 = st.number_input("{}".format(feature_mapping[c2]), min_value=X_new[c2].min(),  max_value=X_new[c2].max(), value=idx)
            new_feature_input[c2].append(f2)
        with col3:
            if (i+2) >= len(numerical_columns):
                continue
            c3 = numerical_columns[i+2] 
            idx = X_new.loc[select_patient, c3]
            f3 = st.number_input("{}".format(feature_mapping[c3]), min_value=X_new[c3].min(),  max_value=X_new[c3].max(), value=idx)
            new_feature_input[c3].append(f3)
        with col4:
            if (i+3) >= len(numerical_columns):
                continue
            c4 = numerical_columns[i+3] 
            idx = X_new.loc[select_patient, c4]
            f4 = st.number_input("{}".format(feature_mapping[c4]), min_value=X_new[c4].min(),  max_value=X_new[c4].max(), value=idx)
            new_feature_input[c4].append(f4)
    
    st.write('--'*10)
    st.write("### Do you want to see the effect of changing a factor on this patient?")
    color_discrete_map = {}
    color_discrete_map_list = ["red", "green", "blue", "goldenred", "magenta", "yellow", "pink", "grey"]
    for e, classname in enumerate(class_names):
        color_discrete_map[classname] = color_discrete_map_list[e] 
    
    show_whatif = st.checkbox("Enable what-if analysis")
    col01, col02 = st.columns(2)
    with col01:
        st.write('### Prediction on actual feature values')

        import altair as alt
        K = K.rename(columns={"classname": "Class Labels", "predicted_probability": "Predicted Probability"})
        f = alt.Chart(K).mark_bar().encode(
                    y=alt.Y('Class Labels:N',sort=alt.EncodingSortField(field="Predicted Probability", order='descending')),
                    x=alt.X('Predicted Probability:Q'),
                    color=alt.Color('color', legend=None),
                ).properties(width=500, height=300)
        st.write(f)
        st.write('#### Model Output Trajectory for {} Class using SHAP values'.format(predicted_class))

        @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
        def load_model5():
            with open('saved_models/trainXGB_gpu_{}.data'.format(predicted_class), 'rb') as f:
                new_train = pickle.load(f)
            return new_train
        new_train = load_model5()
        exval = new_train[2]['explainer_train'] 
        explainer_train = shap.TreeExplainer(M_dict[predicted_class])

        shap_values_train = explainer_train.shap_values(t1)
        t1 = t2.copy() #
        t1.columns = t1.columns.map(lambda x: feature_mapping.get(x, x).split(' (')[0])
        shap.force_plot(exval, shap_values_train, t1.round(2), show=False, matplotlib=True, link='logit', contribution_threshold=0.10)
        plt.savefig("/app/mar4_force_plot.pdf", bbox_inches='tight')
        plt.savefig("/app/mar4_force_plot.eps", bbox_inches='tight')
        st.pyplot()
        fig, ax = plt.subplots()
        t2.columns = t2.columns.map(lambda x: feature_mapping.get(x, x))
        r = shap.decision_plot(exval, shap_values_train, t2.round(2), link='logit', return_objects=True, new_base_value=0, highlight=0)
        st.pyplot(fig)
        fig.savefig('/app/mar4_decisionplot.pdf', bbox_inches='tight')
        fig.savefig('/app/mar4_decisionplot.eps', bbox_inches='tight')
