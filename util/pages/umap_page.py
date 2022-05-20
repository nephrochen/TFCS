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
################
def RGB_to_Hex(rgb):
    RGB = rgb.split(',') 
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def RGB_list_to_Hex(RGB):
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def Hex_to_RGB(hex):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    rgb = str(r) + ',' + str(g) + ',' + str(b)
    return rgb, [r, g, b]

def gradient_color(color_list, color_sum=700):
    color_center_count = len(color_list)
    color_sub_count = int(color_sum / (color_center_count - 1))
    color_index_start = 0
    color_map = []
    for color_index_end in range(1, color_center_count):
        color_rgb_start = Hex_to_RGB(color_list[color_index_start])[1]
        color_rgb_end = Hex_to_RGB(color_list[color_index_end])[1]
        r_step = (color_rgb_end[0] - color_rgb_start[0]) / color_sub_count
        g_step = (color_rgb_end[1] - color_rgb_start[1]) / color_sub_count
        b_step = (color_rgb_end[2] - color_rgb_start[2]) / color_sub_count
        now_color = color_rgb_start
        color_map.append(RGB_list_to_Hex(now_color))
        for color_index in range(1, color_sub_count):
            now_color = [now_color[0] + r_step, now_color[1] + g_step, now_color[2] + b_step]
            color_map.append(RGB_list_to_Hex(now_color))
        color_index_start = color_index_end
    return color_map

def col_sca(g,input_colors):
    input_colors = input_colors
    colors = gradient_color(input_colors)
    c=[colors[int((len(colors)-1)*(i*1/(len(g)-1)))] for i in range(len(g))]
    p=[i*1/len(g) for i in range(len(g)+1)]
    c_l = [it for su in [[c[i]]*2 for i in range(len(g))] for it in su]
    p_l = [it for su in [[p[i]]*2 for i in range(len(g)+1)] for it in su]
    return [[x,y] for x,y in zip(p_l[1:len(p_l)-1],c_l )]

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

def sc_dt_ct(u,i_select,g):
    data=[go.Scatter3d(x=u['UMAP1'].values, 
                    y=u['UMAP2'].values,
                    z=u['UMAP3'].values,
                    mode='markers',
                    marker=dict(size=3,
                    color=u[i_select].values,
                    #colorscale=col_sca(g,['#9ac9db','#c82423']),
                    colorbar = dict(thickness=25, title=str(i_select)),
                    showscale=True,  
                    opacity=0.5))]
    return data     
##################################


def sc_dt_ct(u,i_select,g_l):
    grades=sorted(g_l)
    data=[]
    for g in grades:
        df_grade=u[u[i_select]==g]
        data.append(
        go.Scatter3d(
                x=df_grade['UMAP1'].values,
                y=df_grade['UMAP2'].values,
                z=df_grade['UMAP3'].values,
                mode='markers',
                marker=dict(size=3,
                    opacity=0.5,
                ),
                name='Grade:'+str(g)
            )
        )
    return data





##################################
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
        if len(g) < 3: fig = go.FigureWidget(sc_dt_ct(umap_org,i_select,g))
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
        fig = go.FigureWidget(sc_dt_num(umap,"surgery"))
        fig.update_layout(template='plotly_white',margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write('### Topological Space for Mortality Risk')
        fig = go.FigureWidget(sc_dt_num(umap,"dead"))
        fig.update_layout(template='plotly_white',margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)












