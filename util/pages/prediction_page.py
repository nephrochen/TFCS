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
def pdt_feature(f_info,f_i,):
    if f_info.loc[f_i,"cat"]:
        fs = st.selectbox(f_info.loc[f_i,"dis_name"],tuple(ast.literal_eval(f_info.loc[f_i,"value"])), 
                        index=int( f_info.loc[f_i,"index"]))
                        
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
    explainer = joblib.load('util/models/ts_k_explainer_fs.pkl') 
    shap_values = joblib.load('util/models/ts_shap_values.pkl') 
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

    for j in range(4):
        cols= st.columns(4)
        for i in range(len(cols)):
            with cols[i]:f_input.append(pdt_feature(f_info,f[i+4*j]))
        
    cols= st.columns(4)
    for i in range(len(cols)-1):
        with cols[i]: f_input.append(pdt_feature(f_info,f[i+16]))
    
    print(f_input)

    st.write('## Predict mortality rate: '+str(round(
        model.predict_proba([f_input])[:, 1][0]*100,2
    )
        )+"%")

   
    st.write('## Details:')


    st_shap(shap.plots.waterfall(shap_values_waterfall(explainer,f_input,f)))
    # st_shap(shap.plots.beeswarm(shap_values), height=300)

    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X)

    # st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_display.iloc[0,:]), height=200, width=1000)
    # st_shap(shap.force_plot(explainer.expected_value, shap_values[:1000,:], X_display.iloc[:1000,:]), height=400, width=1000)








    # #st.set_option('deprecation.showPyplotGlobalUse', False)
    # explainer = joblib.load('util/models/ts_k_explainer.pkl') 
    # shap_values = explainer.shap_values(pd.Series(f_input))
    # shap.force_plot(explainer.expected_value[1],shap_values[1], f_input,matplotlib=True,show=False
    #                   ,figsize=(16,5))
    # st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)


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
