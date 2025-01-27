# Stock Market Analysis and Forecasting Dashboard

## Introduction

The Stock Market Analysis and Forecasting Dashboard is a comprehensive Streamlit application that provides in-depth analysis and forecasting capabilities for NIFTY 500 companies. The application is divided into four main pages, each focusing on different aspects of stock market analysis and prediction.

## Features

1. **Fundamental Analysis**:
   - Displays various financial ratios including Profitability Ratios, Leverage Ratios, Valuation Ratios, and Operating Ratios.
   - Calculates the Piotroski F-Score, which is a tool used to evaluate the financial strength of a company.
   - Provides the Intrinsic Value of the stock using the Graham's model.
   - Allows users to download the financial statements for the selected company.

2. **Technical Analysis**:
   - Generates various technical indicators such as Simple Moving Average (SMA), Exponential Moving Average (EMA), Trend Line, Bollinger Bands, Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD).
   - Plots the historical stock price data along with the technical indicators to provide a comprehensive visual analysis.

3. **Forecasting**:
   - Implements custom forecasting models using the Prophet library and AutoML (PyCaret) to predict future stock prices.
   - Displays the forecasted stock prices and various performance metrics to evaluate the accuracy of the models.

4. **Sentiment Analysis**:
   - Fetches the latest 10 news articles related to the selected company from the NewsAPI.
   - Performs sentiment analysis on the news articles using the VADER (Valence Aware Dictionary and sEntiment Reasoner) algorithm.
   - Classifies the sentiment as positive, negative, or neutral, and provides the overall sentiment score.

## Installation and Usage

1. Clone the repository:
   ```
   git clone https://github.com/nithinmanayilgithub/streamlit-stock-forecaster.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root directory and add your API keys:
   ```
   news_api_key=your_news_api_key_here
   ```

4. Run the Streamlit application:
   ```
   streamlit run 1_🏠_Home.py
   ```

5. The application will open in your default web browser. You can navigate through the different pages using the sidebar menu.

## Technology Stack

- Python
- Streamlit
- Pandas
- Numpy
- Yfinance
- Plotly
- VADER Sentiment Analyzer
- Prophet
- PyCaret

## Future Improvements

- Implement a more comprehensive portfolio management system.
- Add support for more stock exchanges and international markets.
- Enhance the user interface and provide more customization options.
- Integrate additional data sources and analysis techniques.

## Conclusion

The Stock Market Analysis and Forecasting Dashboard provides a powerful and user-friendly tool for investors and analysts to explore the stock market. By leveraging various analytical techniques and forecasting models, the application aims to help users make informed investment decisions.

![https://youtu.be/azvRbWnJl1A]
