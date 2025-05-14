import os
from datetime import datetime, timedelta

import nltk
import pandas as pd
import praw
import yfinance as yf
from dotenv import load_dotenv
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure required NLTK data is available
nltk.download('vader_lexicon')

# Load API credentials from .env file
load_dotenv()

class StockSentimentAnalyzer:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        self.sia = SentimentIntensityAnalyzer()

    def get_reddit_posts(self, stock_symbol: str, limit: int = 100) -> pd.DataFrame:
        subreddit_query = 'stocks+investing+wallstreetbets'
        search_query = f"{stock_symbol} stock"
        posts = []

        for post in self.reddit.subreddit(subreddit_query).search(search_query, limit=limit):
            posts.append({
                "title": post.title,
                "text": post.selftext,
                "score": post.score,
                "created_utc": datetime.fromtimestamp(post.created_utc)
            })

        return pd.DataFrame(posts)

    def analyze_sentiment(self, text: str) -> float:
        return self.sia.polarity_scores(text)['compound']

    def get_stock_data(self, stock_symbol: str, days: int = 30) -> pd.DataFrame:
        end = datetime.now()
        start = end - timedelta(days=days)
        return yf.Ticker(stock_symbol).history(start=start, end=end)

    def predict_trend(self, stock_symbol: str) -> str:
        posts = self.get_reddit_posts(stock_symbol)
        if posts.empty:
            return "No data available to make prediction."

        posts['sentiment'] = posts['text'].apply(self.analyze_sentiment)
        avg_sentiment = posts['sentiment'].mean()

        if avg_sentiment > 0.2:
            return "Bullish (Positive sentiment detected)"
        elif avg_sentiment < -0.2:
            return "Bearish (Negative sentiment detected)"
        else:
            return "Neutral (Mixed sentiment)"

    def sentiment_summary(self, stock_symbol: str) -> dict:
        posts = self.get_reddit_posts(stock_symbol)
        posts['sentiment'] = posts['text'].apply(self.analyze_sentiment)
        return {
            "positive": (posts['sentiment'] > 0.2).sum(),
            "neutral": ((posts['sentiment'] >= -0.2) & (posts['sentiment'] <= 0.2)).sum(),
            "negative": (posts['sentiment'] < -0.2).sum()
        }

    def recent_price_change(self, stock_symbol: str) -> float:
        data = self.get_stock_data(stock_symbol, days=7)
        if data.empty:
            return None
        start, end = data['Close'].iloc[0], data['Close'].iloc[-1]
        return ((end - start) / start) * 100

    def average_post_score(self, stock_symbol: str) -> float:
        posts = self.get_reddit_posts(stock_symbol)
        return posts['score'].mean() if not posts.empty else None

    def sentiment_over_time(self, stock_symbol: str) -> pd.Series:
        posts = self.get_reddit_posts(stock_symbol)
        if posts.empty:
            return pd.Series()

        posts['created_utc'] = pd.to_datetime(posts['created_utc'])
        posts['sentiment'] = posts['text'].apply(self.analyze_sentiment)
        posts.set_index('created_utc', inplace=True)
        return posts['sentiment'].resample('D').mean().dropna()

    def get_most_positive_post(self, stock_symbol: str) -> pd.Series:
        posts = self.get_reddit_posts(stock_symbol)
        if posts.empty:
            return None

        posts['sentiment'] = posts['text'].apply(self.analyze_sentiment)
        return posts.loc[posts['sentiment'].idxmax()]

    def get_most_negative_post(self, stock_symbol: str) -> pd.Series:
        posts = self.get_reddit_posts(stock_symbol)
        if posts.empty:
            return None

        posts['sentiment'] = posts['text'].apply(self.analyze_sentiment)
        return posts.loc[posts['sentiment'].idxmin()]

def main():
    analyzer = StockSentimentAnalyzer()
    stock = input("Enter stock symbol (e.g., AAPL): ").upper()

    try:
        print(f"\nAnalysis for {stock}:")
        print(analyzer.predict_trend(stock))

        summary = analyzer.sentiment_summary(stock)
        print("\nSentiment distribution:")
        print(f"Positive: {summary['positive']}, Neutral: {summary['neutral']}, Negative: {summary['negative']}")

        price_change = analyzer.recent_price_change(stock)
        if price_change is not None:
            print(f"\n7-day price change: {price_change:.2f}%")

        avg_score = analyzer.average_post_score(stock)
        if avg_score is not None:
            print(f"Average Reddit post score: {avg_score:.2f}")

        trend = analyzer.sentiment_over_time(stock)
        if not trend.empty:
            print("\nSentiment trend (last 5 days):")
            print(trend.tail())

        most_positive = analyzer.get_most_positive_post(stock)
        if most_positive is not None:
            print("\nMost Positive Post:")
            print(f"Title: {most_positive['title']}\nSentiment: {most_positive['sentiment']:.2f}")

        most_negative = analyzer.get_most_negative_post(stock)
        if most_negative is not None:
            print("\nMost Negative Post:")
            print(f"Title: {most_negative['title']}\nSentiment: {most_negative['sentiment']:.2f}")

        print("\nRecent Reddit posts:")
        for _, post in analyzer.get_reddit_posts(stock, limit=5).iterrows():
            print(f"\nTitle: {post['title']}\nSentiment: {analyzer.analyze_sentiment(post['text']):.2f}")

    except Exception as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    main()
 
