import streamlit as st
import yfinance as yf
import pandas as pd

# Initialize an empty dictionary to store watchlists
watchlists = {}

# Define a function to create a new watchlist page
def create_watchlist_page(watchlist_name):
  """Creates a new page for a specific watchlist.

  Args:
      watchlist_name: The name of the watchlist.
  """
  # Add a new page for the watchlist
  watchlist_page = st.sidebar.checkbox(watchlist_name)
  if watchlist_page:
    st.subheader(watchlist_name)

    # Check if the watchlist exists
    if watchlist_name not in watchlists:
      watchlists[watchlist_name] = []

    # Add a button to add new stocks to the watchlist
    if st.button(f"Add Stock to {watchlist_name}"):
      stock_symbol = st.text_input("Enter Stock Symbol")
      if stock_symbol:
        try:
          # Download stock data using yfinance
          stock_data = yf.download(stock_symbol, period="1d", interval="1m")
          # Add the stock data to the watchlist
          watchlists[watchlist_name].append(stock_data)
          st.success(f"Stock {stock_symbol} added to {watchlist_name} watchlist")
        except:
          st.error(f"Error downloading data for {stock_symbol}")

    # Display the watchlist data
    if watchlists[watchlist_name]:
      watchlist_df = pd.concat(watchlists[watchlist_name])
      st.dataframe(watchlist_df)

# Create the main page
st.title("Stock Market Watchlist")

# Create pages for different watchlists
create_watchlist_page("My Watchlist")
create_watchlist_page("Technology Stocks")
create_watchlist_page("Financial Stocks")

# Add more pages as needed!

