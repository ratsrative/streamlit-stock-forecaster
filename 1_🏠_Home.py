import json

import requests  # pip install requests
import streamlit as st  # pip install streamlit
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
from utilities import load_css


# GitHub: https://github.com/andfanilo/streamlit-lottie
# Lottie Files: https://lottiefiles.com/
st.set_page_config(
    page_title="Hello",
    page_icon="🏠",
)
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()


lottie_coding_1 = load_lottiefile("Animation-Bull.json")
lottie_coding_2 = load_lottiefile("Animation-Loading.json")# replace link to local lottie file
st.title(":green[Market Overview] : Your Gateway to Stock Insights :white_check_mark:")
st.write("""
* Welcome to **:green[Market Overview]**, your premier stock analysis app. 
* Gain insights, make informed decisions, and stay ahead of the market with our comprehensive dashboard. 
* Real-time data, customizable charts, and expert analytic methods backed by artificial intelligence empower you to navigate the complexities of the stock market confidently. 
* Start optimizing your investments **today!**""")
st_lottie(
    lottie_coding_1,
    speed=1,
    reverse=False,
    loop=True,
    quality="low",  # medium ; high
    #renderer="svg",  # canvas
    height=None,
    width=600,
    key=None,
)
st_lottie(
    lottie_coding_2,
    speed=1,
    reverse=False,
    loop=True,
    quality="low",  # medium ; high
    height=None,
    width=None,
    key=None,
)

load_css()