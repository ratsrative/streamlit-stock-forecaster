# import Libraries
import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import os
load_dotenv()  # take environment variables from .env
# Create class for Raising Exception
class InvalidCompanyNameException(Exception):
    pass

# Header Image
image1 = Image.open('./pages/Stock Market Analysis Header.png')
st.image(image1)

# get nifty500 data
@st.cache_data
def get_nifty500_list():
    df = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_500", match='Nifty 500 List')
    df = df[0]
    df.columns = df.iloc[0]
    companies = df["Company Name"].to_list()
    return companies


def calculate_sentiment(text):
    # Run VADER on text
    score = sentimentAnalyser.polarity_scores(text)
    # Extract compound score
    compound_score = score['compound']
    return compound_score

def sentiment_category(text):
    # Run VADER on text
    score = sentimentAnalyser.polarity_scores(text)
    # Extract compound score
    compound_score = score['compound']
    if compound_score > 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

available_companies = get_nifty500_list()

# Initialize VADER so we can use it later
sentimentAnalyser = SentimentIntensityAnalyzer()

company = st.selectbox(
    "Select **NIFTY 500** Company",
    available_companies)
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"
NEWS_API_KEY = os.getenv('news_api_key')

news_params = {
    "apiKey": NEWS_API_KEY,
    "qInTitle":company,
}
news_response = requests.get(NEWS_ENDPOINT, params=news_params)
articles = news_response.json()["articles"]
sixteen_articles = articles[:10]
formatted_articles = [
    (f"{company}", f"{article['title']}", f"{article['description']}", f"{article['url']}", f"{article['publishedAt']}")
    for article in sixteen_articles]

df_news = pd.DataFrame(formatted_articles, columns=['Company', 'Headline', 'Brief', 'URL', 'Timestamp'])
df_news['Timestamp'] = pd.to_datetime(df_news['Timestamp'])
df_news['Sentiment Score'] = df_news['Brief'].apply(calculate_sentiment)
df_news['sentiment Category'] = df_news['Brief'].apply(sentiment_category)
df_news.sort_values(by=['Timestamp'], ascending=True, inplace=True)

try:
    if company == "Company Name":
        raise InvalidCompanyNameException("Invalid company name: 'Company Name'")
    if df_news.shape[0] == 0:
        st.write(f'The current news cycle has seen less focus on **{company}**')
    else:
        st.header(f'Top News : {company}')
        for i in range(len(df_news)):
            st.subheader(f'News {i + 1}')
            st.write(str(df_news['Headline'][i]))
            st.write(str(df_news['Brief'][i]))
            st.write(str(df_news['URL'][i]))
            st.write(str(df_news['Timestamp'][i]))
            sentiment_score_str = str(df_news['Sentiment Score'][i])
            st.write(f'Sentiment Score : {sentiment_score_str}')
            news_sentiment_str = str(df_news['sentiment Category'][i])
            st.write(f'News Sentiment: {news_sentiment_str}')

        fig_px_line = px.line(df_news, x="Timestamp", y="Sentiment Score", color='sentiment Category',
                          markers=True, color_discrete_sequence=["red","goldenrod","green"])

        fig_px_line.update_traces(textposition="bottom right")
        st.plotly_chart(fig_px_line)
except KeyError:
    st.write(f'The current news cycle has seen less focus on {company}')
