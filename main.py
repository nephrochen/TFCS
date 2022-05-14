

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import streamlit as st
from PIL import Image

from util.functions.path import get_file_path, get_dir_name, util_str, data_str
from util.pages.home_page import home_page
from util.pages.umap_page import umap_page
from util.pages.prediction_page import prediction_page


class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({"title": title, "function": func})

    def run(self):
        img = Image.open(
            get_file_path(
                "rascore_logo.png",
                dir_path=f"{get_dir_name(__file__)}/{util_str}/{data_str}",
            ),
        )

        st.set_page_config(page_title="rascore", page_icon=img, layout="wide")

        st.sidebar.markdown("## Main Menu")
        app = st.sidebar.selectbox(
            "Select Page", self.apps, format_func=lambda app: app["title"]
        )
        st.sidebar.markdown("---")
        app["function"]()


app = MultiApp()

app.add_app("Home Page", home_page)
app.add_app("prediction", prediction_page)
app.add_app("umap", umap_page)
app.run()
