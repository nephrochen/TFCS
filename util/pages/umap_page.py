import streamlit as st


import pandas as pd
import shap
shap.__version__


import plotly.graph_objects as go
import numpy as np
from ..functions.path import get_file_path, pages_str, data_str, get_dir_name
from util.functions.path import get_file_path, get_dir_name, util_str, data_str



def umap_page():
    st.write("## Topological Space for ALS Subtypes using Semi-supervised Approach")



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

    st.sidebar.markdown("### PDB Selection")
    select_color = st.sidebar.selectbox('', [colorable_columns_maps[i] for i in colorable_columns], index=0)

    umap = umap.rename(columns=colorable_columns_maps) 
    umap_usp = umap_usp.rename(columns=colorable_columns_maps) 
    umap_org = umap[[select_color] + ['UMAP1', 'UMAP2', 'UMAP3']].dropna()
    umap_rep = umap_usp[[select_color] + ['UMAP1', 'UMAP2', 'UMAP3']].dropna()
    col1, col2 = st.columns(2)
    with col1:
        st.write('### Discovery Cohort')
        u=umap_org
        fig = go.FigureWidget(data=[go.Scatter3d(x=u['UMAP1'].values, 
                                                 y=u['UMAP2'].values, 
                                                 z=u['UMAP3'].values, 
                                                mode='markers',
            marker=dict(size=3,color=u[select_color].values,colorscale='Viridis',  opacity=0.5))])
        fig.update_layout(template='plotly_white',margin=dict(l=0, r=0, b=0, t=0),)
        fig.update_layout(showlegend=True,legend=dict(orientation="h",yanchor="bottom",xanchor="right",) )
        st.plotly_chart(fig, use_container_width=True,template="plotly_dark")
    with col2:
        st.write('### Replication Cohort')
        u=umap_rep
        fig = go.FigureWidget(data=[go.Scatter3d(x=u['UMAP1'].values, y=u['UMAP2'].values, z=u['UMAP3'].values, mode='markers',
            marker=dict(size=3,color=u[select_color].values,colorscale='Viridis',  opacity=0.5))])
        fig.update_layout(template='plotly_white',margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
