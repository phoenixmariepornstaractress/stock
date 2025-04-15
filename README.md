# Reddit Stock Sentiment Analyzer

This project analyzes Reddit sentiment to predict stock market trends. It uses natural language processing to analyze Reddit posts and comments about stocks, and correlates this sentiment with actual stock price movements.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Reddit API credentials:
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```

3. Run the main script:
```bash
python stock_sentiment.py
```

## Features

- Fetches Reddit posts and comments about specific stocks
- Analyzes sentiment using natural language processing
- Correlates sentiment with stock price movements
- Provides basic trend predictions

## Note

This is a beginner-friendly project for educational purposes. Always do your own research before making investment decisions. 