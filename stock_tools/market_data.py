import datetime
import re
import urllib.parse
from html.parser import HTMLParser

import feedparser
import pandas as pd
import pandas_ta as ta
import requests
import streamlit as st
import yfinance as yf

from stock_tools.common import normalize_symbol


class _FubonChipTableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._in_target_table = False
        self._current_row = None
        self._current_cell = None
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            attributes = dict(attrs)
            if "t01" in attributes.get("class", "").split():
                self._in_target_table = True
        elif self._in_target_table and tag == "tr":
            self._current_row = []
        elif self._in_target_table and tag in {"td", "th"}:
            self._current_cell = []

    def handle_data(self, data):
        if self._in_target_table and self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag):
        if not self._in_target_table:
            return
        if tag in {"td", "th"} and self._current_cell is not None:
            cell_text = " ".join("".join(self._current_cell).split())
            self._current_row.append(cell_text)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None
        elif tag == "table":
            self._in_target_table = False


def _parse_fubon_chip_page(html: str, symbol: str) -> dict:
    parser = _FubonChipTableParser()
    parser.feed(html)

    date_match = re.search(r"日期\s*[:：]\s*(\d{1,2}/\d{1,2})", " ".join(
        cell for row in parser.rows for cell in row
    ))
    if not date_match:
        raise ValueError("富邦頁面未包含資料日期。")

    labels = {
        "董監持股": "directors_and_supervisors",
        "外資持股": "foreign_investor",
        "投信持股": "investment_trust",
        "自營商持股": "dealer",
        "融資餘額": "margin_balance",
        "融券餘額": "short_balance",
    }
    positions = {}
    for row in parser.rows:
        if len(row) < 4 or row[0] not in labels:
            continue

        values = [re.sub(r"[,\s]", "", value) for value in row[1:4]]
        if not re.fullmatch(r"\d+(?:\.\d+)?", values[0]):
            if row[0] == "投信持股":
                raise ValueError(f"富邦頁面 {row[0]} 的持股數格式無法辨識。")
            continue
        daily_change = (
            int(float(values[1]))
            if re.fullmatch(r"-?\d+(?:\.\d+)?", values[1])
            else None
        )
        percent = values[2].rstrip("%")
        ownership_pct = (
            float(percent)
            if re.fullmatch(r"\d+(?:\.\d+)?", percent)
            else None
        )
        if row[0] == "投信持股" and ownership_pct is None:
            raise ValueError(f"富邦頁面 {row[0]} 的持股比例格式無法辨識。")

        positions[labels[row[0]]] = {
            "lots": int(float(values[0])),
            "daily_change_lots": daily_change,
            "ownership_pct": ownership_pct,
        }

    if "investment_trust" not in positions:
        raise ValueError("富邦頁面未找到投信持股資料。")

    return {
        "available": True,
        "source": "Fubon eBroker",
        "data_date": date_match.group(1),
        "unit": "張",
        "positions": positions,
        "note": "頁面所列最新資料；程式快取 5 分鐘，頁面更新頻率未經確認，不代表盤中即時庫存。",
        "url": f"https://fubon-ebrokerdj.fbs.com.tw/z/zc/zcj/zcj_{symbol}.djhtm",
    }


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_fubon_chip_data(symbol: str) -> dict:
    url = f"https://fubon-ebrokerdj.fbs.com.tw/z/zc/zcj/zcj_{symbol}.djhtm"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Referer": "https://fubon-ebrokerdj.fbs.com.tw/",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.content.decode("big5", errors="replace")
        return _parse_fubon_chip_page(html, symbol)
    except (requests.RequestException, UnicodeError, ValueError) as error:
        return {
            "available": False,
            "source": "Fubon eBroker",
            "error": str(error),
            "url": url,
        }


def fetch_fubon_holdings(symbol: str) -> dict:
    """Fetch Fubon eBroker's latest institutional holdings snapshot for a Taiwan stock.

    Must be used for questions about investment-trust holdings, holdings lots,
    or ownership percentages. This is a holdings snapshot, not cumulative
    FinMind trading flow.
    """
    clean_symbol = symbol.strip().upper().replace(".TW", "").replace(".TWO", "")
    if not clean_symbol.isdigit():
        return {"error": "富邦法人持股資料僅支援台灣股票代碼 (例如 2330, 0050)。"}

    snapshot = _fetch_fubon_chip_data(clean_symbol)
    if not snapshot["available"]:
        return {
            "error": f"無法取得富邦 {clean_symbol} 法人持股資料：{snapshot['error']}",
            "source": snapshot["source"],
            "source_url": snapshot["url"],
        }

    trust_holding = snapshot["positions"]["investment_trust"]
    return {
        "symbol": clean_symbol,
        "source": snapshot["source"],
        "data_type": "institutional_holdings_snapshot",
        "data_date": snapshot["data_date"],
        "unit": snapshot["unit"],
        "investment_trust": trust_holding,
        "other_institutional_holdings": {
            name: position
            for name, position in snapshot["positions"].items()
            if name in {"directors_and_supervisors", "foreign_investor", "dealer"}
        },
        "source_url": snapshot["url"],
        "note": snapshot["note"],
    }


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
    獲取富邦頁面的法人持股快照與 FinMind 每日法人買賣資料 (單位: 張)。
    富邦頁面資料日期及更新頻率以來源頁面為準，不能保證為盤中即時資料。
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
    result = {
        "symbol": clean_symbol,
        "sources": [
            "Fubon eBroker 個股籌碼分佈頁面",
            "FinMind TaiwanStockInstitutionalInvestorsBuySell",
        ],
        "data_type": "fubon_institutional_holdings_snapshot_and_finmind_daily_trading",
        "institutional": {},
        "institutional_trading": {},
        "holding_positions": _fetch_fubon_chip_data(clean_symbol),
        "data_warnings": [],
        "major_holders": {},
    }

    try:
        params_inst = {
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": clean_symbol,
            "start_date": query_start,
        }
        response = requests.get(url, params=params_inst, timeout=5)
        response.raise_for_status()
        res_inst = response.json()
        if res_inst.get("status", 200) != 200:
            raise RuntimeError(res_inst.get("msg", "FinMind API 回傳錯誤"))

        data = res_inst.get("data", [])
        if not data:
            result["data_warnings"].append(
                f"FinMind 找不到 {clean_symbol} 在指定日期區間內的法人資料。"
            )
            if result["holding_positions"]["available"]:
                return result
            return {
                "error": result["data_warnings"][0],
                "holding_positions": result["holding_positions"],
            }

        latest_date = max(item["date"] for item in data)
        latest_data = [item for item in data if item["date"] == latest_date]
        period_data = data if start_date else latest_data

        def summarize_trading(items, names):
            buy_shares = sum(
                int(item.get("buy") or 0) for item in items if item["name"] in names
            )
            sell_shares = sum(
                int(item.get("sell") or 0) for item in items if item["name"] in names
            )
            return {
                "buy_lots": buy_shares // 1000,
                "sell_lots": sell_shares // 1000,
                "net_buy_lots": (buy_shares - sell_shares) // 1000,
            }

        foreign_period = summarize_trading(period_data, {"Foreign_Investor"})
        trust_period = summarize_trading(period_data, {"Investment_Trust"})
        dealer_period = summarize_trading(period_data, {"Dealer_Self", "Dealer_Hedging"})
        foreign_latest = summarize_trading(latest_data, {"Foreign_Investor"})
        trust_latest = summarize_trading(latest_data, {"Investment_Trust"})
        dealer_latest = summarize_trading(latest_data, {"Dealer_Self", "Dealer_Hedging"})

        period_label = latest_date
        if start_date:
            period_label = f"{query_start} 至 {latest_date} 累計"

        result["data_date"] = latest_date
        result["institutional"] = {
            "period": period_label,
            "foreign_buy_share": foreign_period["net_buy_lots"],
            "trust_buy_share": trust_period["net_buy_lots"],
            "dealer_buy_share": dealer_period["net_buy_lots"],
        }
        result["institutional_trading"] = {
            "unit": "張",
            "period": {
                "foreign_investor": foreign_period,
                "investment_trust": trust_period,
                "dealer_total": dealer_period,
            },
            "latest_day": {
                "date": latest_date,
                "foreign_investor": foreign_latest,
                "investment_trust": trust_latest,
                "dealer_total": dealer_latest,
            },
        }
        return result
    except Exception as e:
        error = f"抓取 FinMind 每日法人買賣資料時發生錯誤: {str(e)}"
        if result["holding_positions"]["available"]:
            result["data_warnings"].append(error)
            return result
        return {
            "error": (
                f"{error}；富邦持股資料也無法取得："
                f"{result['holding_positions'].get('error', '未知錯誤')}"
            ),
            "holding_positions": result["holding_positions"],
        }


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
