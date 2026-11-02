# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

Prerequisite: install `uv` if you don't already have it.

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Sync the dependencies

   ```
   $ uv sync
   ```

2. Run the app

   ```
   $ uv run streamlit run streamlit_app.py
   ```

### Code organization

- `streamlit_app.py` handles Streamlit UI, authentication, and AI conversation flow.
- `stock_tools/chat_history.py` provides chat-session and global-memory database operations.
- `stock_tools/charts.py` queues chart requests, renders charts, and restores saved charts.
- `stock_tools/youtube_tools.py` retrieves YouTube transcripts using subtitles or Whisper.
- `stock_tools/market_data.py` provides stock price, Fubon eBroker institutional-holdings snapshots, FinMind daily institutional buy/sell data, and news. Holdings queries use the dedicated `fetch_fubon_holdings` tool; the Fubon page's update frequency is unverified, and FinMind reports daily trading flow, not holdings.
- `stock_tools/portfolio.py` records trades and summarizes holdings.
- `stock_tools/strategies.py` scans markets and runs strategy backtests.
- `stock_tools/common.py` contains shared database and symbol helpers.
