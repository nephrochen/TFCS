import streamlit as st
import pandas as pd
import shap
import plotly.graph_objects as go
import numpy as np
from ..functions.path import get_file_path, pages_str, data_str, get_dir_name
from util.functions.path import get_file_path, get_dir_name, util_str, data_str
#from ..functions.colors import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast


#####
def sc_dt_num(u,i_select):
    data=[go.Scatter3d(x=u['UMAP1'].values, 
                    y=u['UMAP2'].values,
                    z=u['UMAP3'].values,
                    mode='markers',
                    marker=dict(size=3,
                    color=u[i_select].values,
                    colorscale=['#9ac9db','#c82423'],
                    showscale=True,  
                    opacity=0.5,
                    colorbar=dict(title=str(i_select))))]
    return data     
 
##################################


    fs = st.selectbox(f_info.loc[f_i,"dis_name"],tuple(ast.literal_eval(f_info.loc[f_i,"value"])), 
                    index=int( f_info.loc[f_i,"index"]))





c_l=['#8ebbd9','#f08c8d','#b49dcc','#d7ab93','#8ebbd9','#f08c8d','#b49dcc','#d7ab93']
def sc_dt_ct(u,i_select,g_l,u_name):
    print(u_name.index)
    grades=sorted(g_l)
    data=[]
    for i in range(len(grades)):
        df_grade=u[u[i_select]==grades[i]]
        data.append(
        go.Scatter3d(
                x=df_grade['UMAP1'].values,
                y=df_grade['UMAP2'].values,
                z=df_grade['UMAP3'].values,
                mode='markers',
                marker=dict(size=3,opacity=0.75, color=c_l[i],),
                name='ast.literal_eval(u_name.loc[i_select,"u_name"])[i]'
            ))
    return data



##################################
def umap_page():
    st.write("## Topological Space for ALS Subtypes using Semi-supervised Approach")
    u_name = pd.read_csv("/app/tfcs/util/data/f_info2.csv",index_col=0)
    umap = pd.read_csv("/app/tfcs/util/data/umap.csv", sep=',')
    umap_usp =pd.read_csv("/app/tfcs/util/data/umap_usp.csv", sep=',')
    colorable_columns_maps ={
        'SHAP': "SHAP", 
        'dead': "dead",
        'surgery': "surgery",
        "CBP_t":"CBP_t",
        "age":"age",
        "NYHA":"NYHA",
        "weight":"weight",
        "ef":"ef",
        "Crea":"Crea",
        "IABP":"IABP"
    }
    colorable_columns = list(colorable_columns_maps) 
    colorable_columns = list(set(colorable_columns).intersection(set(list(umap.columns))))

    st.sidebar.markdown("### Indicator Selection")
    i_select = st.sidebar.selectbox('', [colorable_columns_maps[i] for i in colorable_columns], index=0)

    umap = umap.rename(columns=colorable_columns_maps) 
    umap_usp = umap_usp.rename(columns=colorable_columns_maps) 
    umap_org = umap[[i_select] + ['UMAP1', 'UMAP2', 'UMAP3']].dropna()
    umap_rep = umap_usp[[i_select] + ['UMAP1', 'UMAP2', 'UMAP3']].dropna()
    
    
    col1, col2 = st.columns(2)

    g=umap_org[i_select].unique()
    with col1:
        st.write('### Topological Space for indicator')
        if len(g) < 5: fig = go.FigureWidget(sc_dt_ct(umap_org,i_select,g,u_name))
        else: fig = go.FigureWidget(sc_dt_num(umap_org,i_select))
        fig.update_layout(template='plotly_white',margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)


    with col2:
        st.write('### Topological Space for Mortality Risk')
        fig = go.FigureWidget(sc_dt_num(umap,"SHAP"))
        fig.update_layout(template='plotly_white',margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)


    col1, col2 = st.columns(2)


    with col1:
        st.write('### Topological Space for types of surgery')
        fig = go.FigureWidget(sc_dt_ct(umap,"surgery",umap["surgery"].unique(),u_name))
        fig.update_layout(template='plotly_white',margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write('### Topological Space for Mortality Risk')
        fig = go.FigureWidget(sc_dt_ct(umap,"dead",umap["dead"].unique(),u_name))
        fig.update_layout(template='plotly_white',margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)












