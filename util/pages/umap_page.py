import streamlit as st


import pandas as pd
import shap
shap.__version__


import plotly.graph_objects as go
import numpy as np
from ..functions.path import get_file_path, pages_str, data_str, get_dir_name
from util.functions.path import get_file_path, get_dir_name, util_str, data_str
#from ..functions.colors import *


def umap_page():
    st.write("## Topological Space for ALS Subtypes using Semi-supervised Approach")
    umap = pd.read_csv("/app/tfcs/util/data/umap.csv", sep=',')
    umap_usp =pd.read_csv("/app/tfcs/util/data/umap_usp.csv", sep=',')
    colorable_columns_maps ={
        #'SHAP': "SHAP", 
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


    def sc_dt_num(u,select_color):
        data=[go.Scatter3d(x=u['UMAP1'].values, 
                        y=u['UMAP2'].values,
                        z=u['UMAP3'].values,
                        mode='markers',
                        marker=dict(size=3,
                        color=u[select_color].values,
                        #colorscale='Viridis',
                        colorscale=['#9ac9db','#c82423'],
                        showscale=True,  
                        opacity=0.5,
                        colorbar=dict(title=str(select_color))))]
        return data     

    def sc_dt_ct(u,select_color):
        data=[go.Scatter3d(x=u['UMAP1'].values, 
                        y=u['UMAP2'].values,
                        z=u['UMAP3'].values,
                        mode='markers',
                        marker=dict(size=3,
                        color=u[select_color].values,
                        #colorscale='Viridis',
                        colorscale=[[0,'#9ac9db'],[0.5,'#9ac9db'],[0.5,'#c82423'],[1,'#c82423']],
                        colorbar = dict(thickness=25, 
               # tickvals=[.9,1.9,2.9], 
                #ticktext=["CABG","V","CC","CCa"],
                title=str(select_color)),
                        showscale=True,  
                        opacity=0.5))]
        return data     

    g=umap_org[select_color].unique()
    with col1:
        st.write('### Topological Space for indicator')
        if len(g) < 3: fig = go.FigureWidget(sc_dt_ct(umap_org,select_color))
        else: fig = go.FigureWidget(sc_dt_num(umap_org,select_color))
        fig.update_layout(template='plotly_white',margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)


    with col2:
        st.write('### Topological Space for Mortality Risk')
        fig = go.FigureWidget(sc_dt_num(umap,"SHAP"))
        fig.update_layout(template='plotly_white',margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)



        # u=umap_org
        # fig = go.FigureWidget(data=[go.Scatter3d(x=u['UMAP1'].values, 
        #                                          y=u['UMAP2'].values, z=u['UMAP3'].values, mode='markers',
        #     marker=dict(size=3,color=u[select_color].values,colorscale='Viridis' ,  opacity=0.5), showlegend = True)])
        # fig.update_layout(template='plotly_white',margin=dict(l=0, r=0, b=0, t=0),)
        # fig.update_layout(showlegend=True,legend=dict(orientation="h",yanchor="bottom",xanchor="right",) )
        # #fig.update_layout(legend_itemsizing ='select_color')
               
        # st.plotly_chart(fig, use_container_width=True,template="plotly_dark")


    # def sc_dt_ct(u,select_color):
    #     grades=sorted(u[select_color].unique())
    #     data=[]
    #     for g in grades:
    #         df_grade=u[u[select_color]==g]
    #         data.append(
    #             go.Scatter3d(x=df_grade['UMAP1'].values, 
    #                     y=df_grade['UMAP2'].values,
    #                     z=df_grade['UMAP3'].values,
    #                 mode='markers',
    #                 colorscale=['#9ac9db','#c82423'],
    #                 marker=dict(opacity=0.5,),
    #                 name='Gradeafsafasfd:'+str(g)
    #             )
    #         )
    #     return data
        