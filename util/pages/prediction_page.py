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
import ast
from streamlit_shap import st_shap
##


from ..functions.table import mask_equal
from ..functions.col import pdb_code_col
from ..functions.path import pages_str, data_str, get_file_path
from ..functions.gui import create_st_button#, show_st_structure
from PIL import Image

def format_func(option):
    return CHOICES[option]


# def pdt_feature(f_info,f_i,):
#         CHOICES =dict(zip(ast.literal_eval(f_info.loc[f_i,"value"]), 
#                         ast.literal_eval(f_info.loc[f_i,"u_name"])))
#         fs = st.selectbox(f_info.loc[f_i,"dis_name"]+"dasdsad",
#                         options=list(CHOICES.keys()),
#                         index=int( f_info.loc[f_i,"index"]), 
#                         format_func=lambda x: display[x])

def pdt_feature(f_info,f_i,):
    if f_info.loc[f_i,"cat"]:
        fs = st.selectbox(f_info.loc[f_i,"dis_name"],
                          tuple(ast.literal_eval(f_info.loc[f_i,"value"])), 
                           index=int( f_info.loc[f_i,"index"]),
                         format_func=lambda x: tuple(ast.literal_eval(f_info.loc[f_i,"u_name"]))[int(x)])
                            

# def pdt_feature(f_info,f_i,):
#     if f_info.loc[f_i,"cat"]:
#         fs = st.selectbox(f_info.loc[f_i,"dis_name"],
#                           tuple(ast.literal_eval(f_info.loc[f_i,"value"])), 
#                            index=int( f_info.loc[f_i,"index"]))
                        
    else:
        min_v=int(ast.literal_eval(f_info.loc[f_i,"value"])[0])
        max_v=int(ast.literal_eval(f_info.loc[f_i,"value"])[1])
        fs = st.number_input(f_info.loc[f_i,"dis_name"], 
        min_value=min_v, max_value=max_v, value=int(f_info.loc[f_i,"index"]),
        step= int((max_v-min_v)/3))
    return fs

def shap_values_waterfall(k_explainer,ip,f_n):
    shap_values = k_explainer.shap_values(pd.DataFrame([ip,ip]))
    tmp = shap.Explanation(values = np.array(shap_values[1], dtype=np.float32),
                        base_values = np.array([k_explainer.expected_value[1]]*len(shap_values[1]), 
                        dtype=np.float32),data=pd.DataFrame([ip,ip]),feature_names=f_n)
    return tmp[1]



def prediction_page():
    explainer = joblib.load('util/models/ts_k_explainer_new.pkl') 
    #shap_values = joblib.load('util/models/ts_shap_values.pkl') 
    f_info = pd.read_csv('util/data/f_info.csv',index_col=0)
    model = joblib.load('util/models/ts_new.pkl') 
    f=[ 'gender', 'height', 'sk','weight','ART','CVA','TC','LDL_C','aspirin',
 'Catecholamine', 'la',
'statins',
'NYHA','cs_SG','Crea','Glu','lvidd','ef','IABP', 'TR',
 'AR',
 'AF',
 'CBP_t']




    
    f_input=[]
    
    left_col, right_col = st.columns(2)
    with left_col: st.image(Image.open('/app/tfcs/util/data/umap.png'),width=400, caption='')
    right_col.markdown("# TFML-CV-Score")
    right_col.markdown("### A tool for predicting perioperative mortality of combined valve surgery and CABG")
    right_col.markdown("**Created by Dr.Zhihui Zhu, Dr.Chenyu Li, Prof.Haibo Zhang**")
    #right_col.markdown("**Beijing Anzhen Hospital**")


    database_link_dict = {
        "bioRxiv Paper": "#############",
        "GitHub Page": "https://github.com/nephrochen",
        "Chinese Cardiac Surgery Registry": "http://ccsr.cvs-china.com/",
    }

    st.sidebar.markdown("## Database-Related Links")
    for link_text, link_url in database_link_dict.items():
        create_st_button(link_text, link_url, st_col=st.sidebar)

    community_link_dict = {
        "National Clinical Research Center for Cardiovascular Diseases": "https://www.nccmrc.com/",
        "National Center for Cardiovascular Diseases": "https://www.nccd.org.cn/Home/English",
        "Beijing Anzhen Hospital": "https://www.anzhen.org/",
        "Fuwai Hospital": "https://www.fuwai.com/Hospitals/Main?type=4",
    }

    st.sidebar.markdown("## Community-Related Links")
    for link_text, link_url in community_link_dict.items():
        create_st_button(link_text, link_url, st_col=st.sidebar)

    software_link_dict = {

        "Pandas": "https://pandas.pydata.org",
        "NumPy": "https://numpy.org",
        "Sklearn": "https://scikit-learn.org/stable/",
        "Matplotlib": "https://matplotlib.org",
        "Streamlit": "https://streamlit.io",
    }

    st.sidebar.markdown("## Software-Related Links")
    link_1_col, link_2_col, link_3_col = st.sidebar.columns(3)

    i = 0
    link_col_dict = {0: link_1_col, 1: link_2_col, 2: link_3_col}
    for link_text, link_url in software_link_dict.items():

        st_col = link_col_dict[i]
        i += 1
        if i == len(link_col_dict.keys()):
            i = 0

        create_st_button(link_text, link_url, st_col=st_col)

    st.sidebar.markdown("---")
    cols= st.sidebar.columns(4)
    ps=['az','az2','fw','fw2']
    
    for i in range(len(ps)):
        with cols[i]: st.image(Image.open('/app/tfcs/util/data/'+ps[i]+'.png'), caption='')



    st.markdown("---")
    st.markdown("<style>.boxBorder {border: 10px solid #f5e893;font-size: 25px;background-color: #f5e893;text-align:center;}</style>", unsafe_allow_html=True) 
    st.markdown('<div class="boxBorder"><font color="BLACK"> <strong>Disclaimer: This predictive tool is only for research purposes <strong></font></div>', unsafe_allow_html=True)
   
   
    left_p, right_p = st.columns(2)
    with left_p: st.write("## Model Perturbation Analysis")

    for j in range(5):#x5columns
        cols= st.columns(5)
        for i in range(len(cols)):
            with cols[i]:f_input.append(pdt_feature(f_info,f[i+4*j]))

    st.markdown("---")
    st.write('## Waterfall plot for predict mortality rate')
    st_shap(shap.plots.waterfall(shap_values_waterfall(explainer,f_input,f)), height=500, width=1000)
    with right_p: st.write('## Predict mortality rate: '+str(round( model.predict_proba([f_input])[:, 1][0]*100,2) )+"%")