import datetime
import urllib.parse

import feedparser
import pandas as pd
import pandas_ta as ta
import requests
import streamlit as st
import yfinance as yf

from stock_tools.common import normalize_symbol


def show_stock_chart(symbol: str, period: str = "6mo") -> dict:
    """
    當使用者要求畫 K 線圖、走勢圖時呼叫此工具。
    Args:
        symbol: 股票代碼 (例如 '2330', 'AAPL')
        period: 資料長度。有效值必須為: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'。預設為 '6mo'。
    """
    clean_symbol = normalize_symbol(symbol)

    if "render_chart_queue" not in st.session_state:
        st.session_state.render_chart_queue = []

    st.session_state.render_chart_queue.append({"symbol": clean_symbol, "period": period})

    print(f"🛠️ [Tool Called] 加入繪圖佇列 | 渲染: {clean_symbol}, 期間: {period}")

    return {"status": "success", "message": f"已將 {clean_symbol} 加入前端繪圖佇列。"}


def fetch_stock_data(
    symbol: str,
    interval: str = "1d",
    period: str = "1y",
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """
    獲取股票的歷史數據與技術指標。
    - 支援相對時間 period: "1y", "6mo", "1mo"
    - 也支援絕對日期 start_date 與 end_date: 格式為 "YYYY-MM-DD" (例如 "2026-09-01")
    若有提供 start_date，將優先使用精確日期區間。
    """
    clean_symbol = normalize_symbol(symbol)
    try:
        ticker = yf.Ticker(clean_symbol)

        if start_date:
            df = ticker.history(start=start_date, end=end_date, interval=interval)
        else:
            if "m" in interval and period in ["1y", "2y", "max", "ytd"]:
                period = "60d"
            df = ticker.history(period=period, interval=interval)

        if df.empty and clean_symbol.endswith(".TW"):
            clean_symbol = clean_symbol.replace(".TW", ".TWO")
            ticker = yf.Ticker(clean_symbol)
            if start_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)

        if df.empty:
            return {"error": f"找不到 {symbol} 的數據，請確認日期區間內是否有交易日。"}

        df.ta.sma(length=5, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=60, append=True)
        df.ta.rsi(length=14, append=True)

        kd = ta.stoch(df["High"], df["Low"], df["Close"], k=9, d=3, smooth_k=3)
        if kd is not None and not kd.empty:
            df = pd.concat([df, kd], axis=1)

        latest_d = df.iloc[-1]
        prev_d = df.iloc[-2] if len(df) > 1 else latest_d
        change_pct = ((latest_d["Close"] - prev_d["Close"]) / prev_d["Close"]) * 100

        def safe_get(key):
            val = latest_d.get(key)
            return round(float(val), 2) if pd.notna(val) else "N/A"

        return {
            "symbol": clean_symbol,
            "data_interval": interval,
            "latest_time": str(df.index[-1]),
            "close_price": round(float(latest_d["Close"]), 2),
            "change_percent": f"{round(float(change_pct), 2)}%",
            "moving_averages": {
                "sma5": safe_get("SMA_5"),
                "sma20": safe_get("SMA_20"),
                "sma60": safe_get("SMA_60"),
            },
            "momentum_indicators": {
                "rsi_14": safe_get("RSI_14"),
                "kd_k9": safe_get("STOCHk_9_3_3"),
                "kd_d3": safe_get("STOCHd_9_3_3"),
            },
        }
    except Exception as e:
        return {"error": f"抓取技術面數據時發生錯誤: {str(e)}"}


def fetch_chip_data(symbol: str, start_date: str = None) -> dict:
    """
    獲取台股的三大法人買賣超數據 (單位: 張)。
    Args:
        symbol: 股票代碼 (如 2330)
        start_date: 統計起始日期 (格式 YYYY-MM-DD)。
                    - 若未提供，預設僅抓取「最新一個交易日」的資料。
                    - 若查詢「今年累計」或特定區間累計，請帶入對應的起始日期 (例如 '2026-01-01')。
    """
    print(f"🛠️ [Tool Called] 呼叫 fetch_chip_data | 參數: symbol={symbol}, start_date={start_date}")

    clean_symbol = symbol.strip().upper().replace(".TW", "").replace(".TWO", "")
    if not clean_symbol.isdigit():
        return {"error": "籌碼數據僅支援台灣股市 (例如 2330, 0050)"}

    today = datetime.date.today()
    query_start = start_date if start_date else (today - datetime.timedelta(days=10)).strftime("%Y-%m-%d")

    url = "https://api.finmindtrade.com/api/v4/data"
    result = {"symbol": clean_symbol, "institutional": {}, "major_holders": {}}

    try:
        params_inst = {
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": clean_symbol,
            "start_date": query_start,
        }
        res_inst = requests.get(url, params=params_inst, timeout=5).json()

        if res_inst.get("data"):
            if not start_date:
                latest_date = res_inst["data"][-1]["date"]
                target_data = [d for d in res_inst["data"] if d["date"] == latest_date]
                period_label = latest_date
            else:
                target_data = res_inst["data"]
                end_date = res_inst["data"][-1]["date"]
                period_label = f"{query_start} 至 {end_date} 累計"

            summary = {}
            for item in target_data:
                name = item["name"]
                net_buy = (item["buy"] - item["sell"]) // 1000
                summary[name] = summary.get(name, 0) + net_buy

            result["institutional"] = {
                "period": period_label,
                "foreign_buy_share": summary.get("Foreign_Investor", 0),
                "trust_buy_share": summary.get("Investment_Trust", 0),
                "dealer_buy_share": summary.get("Dealer_Self", 0) + summary.get("Dealer_Hedging", 0),
            }
        return result
    except Exception as e:
        return {"error": f"抓取籌碼數據時發生錯誤: {str(e)}"}


def fetch_stock_news(symbol: str) -> dict:
    """抓取特定股票相關的最新 5 則焦點新聞。"""
    clean_symbol = symbol.strip().upper().replace(".TW", "").replace(".TWO", "")
    query = f"{clean_symbol} 股票"
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"

    try:
        feed = feedparser.parse(rss_url)
        news_items = [
            {"title": e.title, "published": getattr(e, "published", "N/A")}
            for e in feed.entries[:5]
        ]
        return {"symbol": clean_symbol, "news": news_items}
    except Exception as e:
        return {"error": f"抓取新聞時發生錯誤: {str(e)}"}
