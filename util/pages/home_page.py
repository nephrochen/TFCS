# -*- coding: utf-8 -*-
"""
  Copyright 2022 Mitchell Isaac Parker

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import streamlit as st
from PIL import Image

from ..functions.table import mask_equal
from ..functions.col import pdb_code_col
from ..functions.path import pages_str, data_str, get_file_path
from ..functions.gui import create_st_button#, show_st_structure

def home_page():

    left_col, right_col = st.columns(2)
    right_col.markdown("# TFML-CV-Score")
    right_col.markdown("### A tool for predicting perioperative mortality of combined valve surgery and CABG")
    right_col.markdown("**Created by Dr.Zhihui Zhu, Dr.Chenyu Li, Prof.Haibo Zhang**")
    right_col.markdown("**Beijing Anzhen Hospital, Capital Medical University and Ludwig-Maximilians-Universität München**")

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

    st.markdown("---")

    st.markdown(
        """
        ### Summary
        *TFML-score*is a tool for predicting perioperative mortality of 
        combined valve surgery and CABG. 
        TFML-score database presents an analysis of 19 indicators
        map structures Individualized mortality prediction for patients. 

        In this study, we sought to develop a ML-based transfer learning risk prediction model integrating past history, 
        preoperative and intraoperative indicators to predict perioperative death after combined heart valve surgery and CABG, 
        by pooling several large cohorts of patients to inform model development and validation. 
        Here, we reported ML models with high predictive power for death prediction, 
        along with patient-level and cohort-level interpretations and discuss the use of ML as a tool to aid understanding. 

        """
    )

    left_col, right_col = st.columns(2)

    # img = Image.open(
    #     get_file_path(
    #         "rascore_abstract.png",
    #         dir_path=get_neighbor_path(__file__, pages_str, data_str),
    #     )
    # )

    # right_col.image(img, output_format="PNG")


    # left_col.markdown(
    #     """
    #     ### Usage

    #     To the left, is a dropdown main menu for navigating to 
    #     each page in the *Rascore* database:

    #     - **Home Page:** We are here!
    #     - **Database Overview:** Overview of the *Rascore* database, molecular annotations, 
    #     and RAS conformational classification.
    #     - **Search PDB:** Search for individual PDB entries containing RAS structures.
    #     - **Explore Conformations:** Explore RAS SW1 and SW2 conformations found in the PDB by nucleotide state.
    #     - **Analyze Mutations:** Analyze the structural impact of RAS mutations by comparing WT and mutated structures.
    #     - **Compare Inhibitors:** Compare inhibitor-bound RAS structures by compound binding site and chemical substructure.
    #     - **Query Database:** Query the *Rascore* database by conformations and molecular annotations.
    #     - **Classify Structures:** Conformationally classify and annotate the molecular contents of uploaded RAS structures.
    #     """
    # )
    st.markdown("---")

    left_info_col, right_info_col = st.columns(2)

    left_info_col.markdown(
        f"""
        ### Authors
        Please feel free to contact us with any issues, comments, or questions.

        ##### Dr.Zhihui Zhu

        - Email:  <Zhihui.Zhu@med.uni-muenchen.de>

        ##### Dr.Chenyu Li

        - Email: <Chenyu.Li@med.uni-muenchen.de or Chenyu.Li@qdu.edu.cn>

        ##### Prof.Haibo Zhang

        - Email: <zhanghb2318@163.com>
        """,
        unsafe_allow_html=True,
    )

    right_info_col.markdown(
        """
        ### Funding

        - Beijing Science and Technology Commission of China D171100002917001 (to Z.Z.)
        - Beijing Science and Technology Commission of China D171100002917003 (to H.Z.)
         """
    )

    right_info_col.markdown(
        """
        ### License
        Apache License 2.0
        """
    )

