import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime
from st_aggrid import AgGrid
@st.cache_data
def get_nifty500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_500",match='Nifty 500 List')
    df = df[0]
    df.columns = df.iloc[0]
    df["Symbol"] = df["Symbol"] + '.NS'
    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(
        zip(df["Symbol"], df["Company Name"])
    )
    return tickers, tickers_companies_dict
@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start, end)

image1 = Image.open('./Pages/Stock Market Analysis Header.png')
st.image(image1)
st.sidebar.header("Stock Parameters")

available_tickers, tickers_companies_dict = get_nifty500_components()

tickers = st.sidebar.selectbox(
    "Ticker",
    available_tickers,
    format_func=tickers_companies_dict.get
)
ticker = yf.Ticker(tickers)

## inputs for technical analysis
st.sidebar.header("Financial Reports")
financials = st.sidebar.selectbox("Add Financials", ["Income Statement",
                                  "Quarterly Income Statement",
                                  "Balance Sheet",
                                  "Quarterly Balance Sheet",
                                  "Cash Flow Statement",
                                  "Quarterly Cash Flow Statement"])
# df = load_data(ticker, start_date, end_date)
title_str = st.header(f"{tickers_companies_dict[tickers]}'s {financials}")
st.download_button("Download file", f"{financials}")
st.write("""
### User manual
* you can select any of the companies that is a component of the **:green[NIFTY 500]** index
* you can download the selected  Financial documents as a CSV files For Fundamental Analysis
""")
#############################

if financials =='Income Statement':
    fin_df = ticker.income_stmt.reset_index()
    fin_df.columns = fin_df.columns.astype(str)
    fin_df.rename(columns={'index':'Income/Expenditure'},inplace=True)
    AgGrid(fin_df)

elif financials =='Quarterly Income Statement':
    st.dataframe(ticker.quarterly_income_stmt, use_container_width=True)
elif financials == 'Balance Sheet':
    balance_sheet_df = ticker.balance_sheet
    balance_sheet_df.columns = balance_sheet_df.columns.astype(str)
    AgGrid(balance_sheet_df)
elif financials == 'Quarterly Balance Sheet':
    quarterly_balance_sheet_df = ticker.quarterly_balance_sheet.reset_index()
    AgGrid(ticker.quarterly_balance_sheet,use_container_width=True)
elif financials == 'Cash Flow Statement':
    AgGrid(ticker.cash_flow,use_container_width=True)
elif financials == 'Quarterly Cash Flow Statement':
    AgGrid(ticker.quarterly_cashflow,use_container_width=True)
