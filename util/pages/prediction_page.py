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
 
##
def pdt_feature(f_info,f_i,):
    if f_info.loc[f_i,"cat"]:
        fs = st.selectbox(f_i,ast.literal_eval(f_info.loc[f_i,"value"]), 
                        index=int( f_info.loc[f_i,"index"]))
    else:
        fs = st.number_input(f_i, 
        min_value=float(ast.literal_eval(f_info.loc[f_i,"value"])[0]), 
        max_value=float(ast.literal_eval(f_info.loc[f_i,"value"])[1]), 
        value=float(f_info.loc[f_i,"index"]))
    return fs



def prediction_page():
    f_info = pd.read_csv('util/data/f_info.csv',index_col=0)
    model = joblib.load('util/models/ts.pkl') 
    f=['gender','height','weight','sk','age','DM','HT','HL','COPD','PVD',
    'CVA','NYHA','ART','Crea','TC','Glu','ef','CBP_t','IABP']
    f_input=[]

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

    
    cols= st.columns(4)
    for i in range(len(cols)):
        with cols[i]:
            f_input.append(pdt_feature(f_info,f[i]))
    
    cols= st.columns(4)
    for i in range(len(cols)):
        with cols[i]:
            f_input.append(pdt_feature(f_info,f[i+4]))
    
    cols= st.columns(4)
    for i in range(len(cols)):
        with cols[i]:
            f_input.append(pdt_feature(f_info,f[i+8]))
    
    cols= st.columns(4)
    for i in range(len(cols)):
        with cols[i]:
            f_input.append(pdt_feature(f_info,f[i+12]))
    
    cols= st.columns(4)
    for i in range(len(cols)-1):
        with cols[i]:
            f_input.append(pdt_feature(f_info,f[i+16]))
    









    # st.write('--'*10)
    # st.write("### Do you want to see the effect of changing a factor on this patient?")
    # color_discrete_map = {}
    # color_discrete_map_list = ["red", "green", "blue", "goldenred", "magenta", "yellow", "pink", "grey"]
    # for e, classname in enumerate(class_names):
    #     color_discrete_map[classname] = color_discrete_map_list[e] 
    
    # show_whatif = st.checkbox("Enable what-if analysis")
    # col01, col02 = st.columns(2)
    # with col01:
    #     st.write('### Prediction on actual feature values')

    #     import altair as alt
    #     K = K.rename(columns={"classname": "Class Labels", "predicted_probability": "Predicted Probability"})
    #     f = alt.Chart(K).mark_bar().encode(
    #                 y=alt.Y('Class Labels:N',sort=alt.EncodingSortField(field="Predicted Probability", order='descending')),
    #                 x=alt.X('Predicted Probability:Q'),
    #                 color=alt.Color('color', legend=None),
    #             ).properties(width=500, height=300)
    #     st.write(f)
    #     st.write('#### Model Output Trajectory for {} Class using SHAP values'.format(predicted_class))

    #     @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    #     def load_model5():
    #         with open('saved_models/trainXGB_gpu_{}.data'.format(predicted_class), 'rb') as f:
    #             new_train = pickle.load(f)
    #         return new_train
    #     new_train = load_model5()
    #     exval = new_train[2]['explainer_train'] 
    #     explainer_train = shap.TreeExplainer(M_dict[predicted_class])

    #     shap_values_train = explainer_train.shap_values(t1)
    #     t1 = t2.copy() #
    #     t1.columns = t1.columns.map(lambda x: feature_mapping.get(x, x).split(' (')[0])
    #     shap.force_plot(exval, shap_values_train, t1.round(2), show=False, matplotlib=True, link='logit', contribution_threshold=0.10)
    #     plt.savefig("/app/mar4_force_plot.pdf", bbox_inches='tight')
    #     plt.savefig("/app/mar4_force_plot.eps", bbox_inches='tight')
    #     st.pyplot()
    #     fig, ax = plt.subplots()
    #     t2.columns = t2.columns.map(lambda x: feature_mapping.get(x, x))
    #     r = shap.decision_plot(exval, shap_values_train, t2.round(2), link='logit', return_objects=True, new_base_value=0, highlight=0)
    #     st.pyplot(fig)
    #     fig.savefig('/app/mar4_decisionplot.pdf', bbox_inches='tight')
    #     fig.savefig('/app/mar4_decisionplot.eps', bbox_inches='tight')
