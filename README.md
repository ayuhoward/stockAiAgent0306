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
- `stock_tools/youtube_tools.py` retrieves captions with the current YouTube Transcript API and falls back to yt-dlp audio plus Whisper; it preserves both subtitle and audio errors when retrieval fails.
- `stock_tools/market_data.py` provides stock price, Fubon eBroker institutional-holdings snapshots, FinMind daily institutional buy/sell data, and news. Holdings queries use the dedicated `fetch_fubon_holdings` tool; the Fubon page's update frequency is unverified, and FinMind reports daily trading flow, not holdings.
- `stock_tools/institutional_screening.py` scans TWSE/TPEx daily investment-trust flows and uses current Fubon holdings to estimate historical accumulation from a near-zero position. It retries transient report failures and identifies any remaining gaps in partial results. Reconstructed holdings are estimates, not historical snapshots.
- `stock_tools/portfolio.py` records trades and summarizes holdings; the Streamlit sidebar shows positions, Chinese company names, share counts, average costs, and total cost basis from the database. `stock_tools/stock_names.py` loads and caches names from TWSE and TPEx.
- `stock_tools/strategies.py` scans markets and runs strategy backtests.
- `stock_tools/common.py` contains shared database and symbol helpers.
