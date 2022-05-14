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
    color_discrete_map = {}
    color_discrete_map_list = ["red", "green", "blue", "magenta", "yellow", "pink", "grey", "black", "brown", "purple"]
    for e, classname in enumerate(sorted( list(set(umap_org[select_color]).union(set(umap_rep[select_color]))) ) ) :
        color_discrete_map[classname] = color_discrete_map_list[e%10] 



    col1, col2 = st.columns(2)
    with col1:
        st.write('### Discovery Cohort')
        u=umap_org
        fig = go.FigureWidget(data=[go.Scatter3d(x=u['UMAP1'].values, y=u['UMAP2'].values, z=u['UMAP3'].values, mode='markers',
            marker=dict(size=3,color=u[select_color].values,colorscale='Viridis',  opacity=0.5))])
        fig.update_layout(template='plotly_white',margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True,template="plotly_dark")
    with col2:
        # st.write('### Replication Cohort')
        # u=umap_rep
        # fig = go.FigureWidget(data=[go.Scatter3d(x=u['UMAP1'].values, y=u['UMAP2'].values, z=u['UMAP3'].values, mode='markers',
        #     marker=dict(size=3,color=u[select_color].values,colorscale='Viridis',  opacity=0.5))])
        # fig.update_layout(template='plotly_white',margin=dict(l=0, r=0, b=0, t=0))
        # st.plotly_chart(fig, use_container_width=True)

        import plotly.graph_objects as go
        import numpy as np
        t = np.linspace(0, 10, 50)
        x, y, z = np.cos(t), np.sin(t), t

        fig= go.Figure(go.Scatter3d(x=x, y=y, z=z, mode='markers'))
        x_eye = -1.25
        y_eye = 2
        z_eye = 0.5
        fig.update_layout(
                title='Animation Test',
                width=600,
                height=600,
                scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
                updatemenus=[dict(type='buttons',
                        showactive=False,
                        y=1,
                        x=0.8,
                        xanchor='left',
                        yanchor='bottom',
                        pad=dict(t=45, r=10),
                        buttons=[dict(label='Play',
                                        method='animate',
                                        args=[None, dict(frame=dict(duration=5, redraw=True), 
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
        )

        def rotate_z(x, y, z, theta):
            w = x+1j*y
            return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z
        frames=[]
        for t in np.arange(0, 6.26, 0.1):
            xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
            frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
        fig.frames=frames

        st.plotly_chart(fig, use_container_width=True)

