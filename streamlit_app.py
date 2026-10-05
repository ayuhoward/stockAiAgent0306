import os
import math
import datetime
import urllib.parse
import requests
import feedparser
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
import psycopg2
import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. 網頁基本設定與資安防護 (密碼鎖)
# ---------------------------------------------------------
st.set_page_config(page_title="AI 量化資產管理助手", page_icon="📈", layout="wide")

# 檢查是否已登入
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 系統已鎖定")
    pwd = st.text_input("請輸入系統密碼以解鎖：", type="password")
    if st.button("登入"):
        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("密碼錯誤，請重新輸入。")
    st.stop()  # 密碼錯誤前，停止載入後續所有程式碼

# 讀取安全金鑰 (從 Streamlit Secrets 讀取，不寫死在程式碼中)
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
AI_MODEL = "gemini-3.5-flash-lite"

# ---------------------------------------------------------
# 2. Supabase 資料庫初始化與連線
# ---------------------------------------------------------
def get_db_connection():
    """建立與 Supabase PostgreSQL 的連線"""
    return psycopg2.connect(SUPABASE_URL)

def init_db():
    """初始化 PostgreSQL 資料表 (若不存在)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 建立持股表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            symbol TEXT PRIMARY KEY,
            shares REAL NOT NULL,
            avg_cost REAL NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')
    
    # 建立交易日誌表 (PostgreSQL 使用 SERIAL 代替 SQLite 的 AUTOINCREMENT)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_logs (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            shares REAL NOT NULL,
            price REAL NOT NULL,
            trade_date TEXT NOT NULL,
            notes TEXT
        )
    ''')
    
    conn.commit()
    cursor.close()
    conn.close()

init_db()

def normalize_symbol(symbol: str) -> str:
    clean_symbol = symbol.strip().upper()
    if clean_symbol.isdigit():
        clean_symbol += ".TW"
    elif not clean_symbol.endswith(".TW") and not clean_symbol.endswith(".TWO") and not clean_symbol.isalpha():
        clean_symbol += ".TW"
    return clean_symbol

# ==========================================
# 更新 1：嚴格驗證的 record_trade 函式
# ==========================================
def record_trade(symbol: str, action: str, shares: float = 0.0, price: float = 0.0, notes: str = "") -> dict:
    """紀錄交易 (買入或賣出股票)。必須嚴格包含代碼、股數與價格。"""
    
    # 嚴格防呆檢查：如果沒有股數、沒有價格，或數值不合理，直接退回命令！
    if not symbol or shares <= 0 or price <= 0:
        return {
            "error": "❌ 拒絕執行：您遺漏了重要資訊！請明確告訴我「哪一檔股票」、「買/賣幾股」以及「成交價錢是多少」。請在了解所有資訊後再呼叫此工具。"
        }

    clean_symbol = normalize_symbol(symbol)
    action = action.upper()
    
    if action not in ['BUY', 'SELL']:
        return {"error": "交易動作必須為 BUY 或 SELL"}

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 寫入交易日誌
        cursor.execute(
            "INSERT INTO trade_logs (symbol, action, shares, price, trade_date, notes) VALUES (%s, %s, %s, %s, %s, %s)",
            (clean_symbol, action, shares, price, now_str, notes)
        )
        
        # 更新持股
        cursor.execute("SELECT shares, avg_cost FROM positions WHERE symbol = %s", (clean_symbol,))
        pos = cursor.fetchone()
        
        if action == "BUY":
            if pos:
                old_shares, old_cost = pos
                new_shares = old_shares + shares
                new_cost = ((old_shares * old_cost) + (shares * price)) / new_shares
                cursor.execute(
                    "UPDATE positions SET shares = %s, avg_cost = %s, updated_at = %s WHERE symbol = %s",
                    (new_shares, round(new_cost, 2), now_str, clean_symbol)
                )
            else:
                cursor.execute(
                    "INSERT INTO positions (symbol, shares, avg_cost, updated_at) VALUES (%s, %s, %s, %s)",
                    (clean_symbol, shares, price, now_str)
                )
        elif action == "SELL":
            if not pos: return {"error": "無持股紀錄"}
            old_shares, old_cost = pos
            if shares >= old_shares:
                cursor.execute("DELETE FROM positions WHERE symbol = %s", (clean_symbol,))
            else:
                new_shares = old_shares - shares
                cursor.execute(
                    "UPDATE positions SET shares = %s, updated_at = %s WHERE symbol = %s",
                    (new_shares, now_str, clean_symbol)
                )
                
        conn.commit()
        return {"status": "success", "message": f"成功紀錄 {action} {clean_symbol} {shares} 股，成交價 {price} 元。"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()

def get_portfolio_summary() -> dict:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, shares, avg_cost FROM positions")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not rows: return {"portfolio": [], "message": "目前無持股"}
        
    portfolio = []
    total_cost = 0.0
    total_market_val = 0.0
    
    for symbol, shares, avg_cost in rows:
        cost_val = shares * avg_cost
        total_cost += cost_val
        curr_price = avg_cost
        try:
            hist = yf.Ticker(symbol).history(period="1d")
            if not hist.empty: curr_price = round(float(hist['Close'].iloc[-1]), 2)
        except: pass
            
        mkt_val = shares * curr_price
        total_market_val += mkt_val
        unrealized_pnl = mkt_val - cost_val
        
        portfolio.append({
            "symbol": symbol, "shares": shares, "avg_cost": avg_cost,
            "current_price": curr_price, "unrealized_pnl": round(unrealized_pnl, 2)
        })
        
    return {
        "summary": {
            "total_cost": round(total_cost, 2),
            "total_unrealized_pnl": round(total_market_val - total_cost, 2)
        },
        "portfolio": portfolio
    }

# ---------------------------------------------------------
# 4. Function Tools
# ---------------------------------------------------------
def fetch_stock_data(symbol: str) -> dict:
    """獲取台股或美股的日線、週線、技術指標 (RSI, MACD, 布林通道) 與支撐壓力位。"""
    clean_symbol = normalize_symbol(symbol)
    try:
        ticker = yf.Ticker(clean_symbol)
        hist_daily = ticker.history(period="1y")
        
        if hist_daily.empty and clean_symbol.endswith(".TW"):
            clean_symbol = clean_symbol.replace(".TW", ".TWO")
            ticker = yf.Ticker(clean_symbol)
            hist_daily = ticker.history(period="1y")
            
        if hist_daily.empty:
            return {"error": f"找不到股票代碼 {symbol} 的相關數據。"}

        hist_weekly = ticker.history(period="2y", interval="1wk")

        hist_daily.ta.sma(length=5, append=True)
        hist_daily.ta.sma(length=20, append=True)
        hist_daily.ta.sma(length=60, append=True)
        hist_daily.ta.rsi(length=14, append=True)
        hist_daily.ta.macd(fast=12, slow=26, signal=9, append=True)
        hist_daily.ta.bbands(length=20, std=2, append=True)

        latest_d = hist_daily.iloc[-1]
        prev_d = hist_daily.iloc[-2] if len(hist_daily) > 1 else latest_d
        recent_30d = hist_daily.tail(30)

        weekly_trend = "N/A"
        if not hist_weekly.empty and len(hist_weekly) >= 20:
            hist_weekly.ta.sma(length=20, append=True)
            latest_w = hist_weekly.iloc[-1]
            if 'SMA_20' in latest_w and not pd.isna(latest_w['SMA_20']):
                weekly_trend = "多頭趨勢 (站上20週均線)" if latest_w['Close'] > latest_w['SMA_20'] else "整理/空頭趨勢"

        change_pct = ((latest_d['Close'] - prev_d['Close']) / prev_d['Close']) * 100

        return {
            "symbol": clean_symbol,
            "date": str(hist_daily.index[-1].strftime('%Y-%m-%d')),
            "close_price": round(float(latest_d['Close']), 2),
            "change_percent": f"{round(float(change_pct), 2)}%",
            "moving_averages": {
                "sma5": round(float(latest_d.get('SMA_5', 0)), 2) if not pd.isna(latest_d.get('SMA_5')) else "N/A",
                "sma20": round(float(latest_d.get('SMA_20', 0)), 2) if not pd.isna(latest_d.get('SMA_20')) else "N/A",
                "sma60": round(float(latest_d.get('SMA_60', 0)), 2) if not pd.isna(latest_d.get('SMA_60')) else "N/A",
            },
            "momentum_indicators": {
                "rsi_14": round(float(latest_d.get('RSI_14', 0)), 2) if not pd.isna(latest_d.get('RSI_14')) else "N/A",
            },
            "support_resistance": {
                "resistance_30d": round(float(recent_30d['High'].max()), 2),
                "support_30d": round(float(recent_30d['Low'].min()), 2)
            },
            "weekly_trend": weekly_trend
        }
    except Exception as e:
        return {"error": f"抓取技術面數據時發生錯誤: {str(e)}"}


def fetch_chip_data(symbol: str) -> dict:
    """獲取台股的三大法人買賣超與千張大戶持股比例數據 (台股專用)。"""
    clean_symbol = symbol.strip().upper().replace(".TW", "").replace(".TWO", "")
    if not clean_symbol.isdigit():
        return {"error": "籌碼數據僅支援台灣股市 (例如 2330, 0050)"}

    today = datetime.date.today()
    start_date = (today - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
    url = "https://api.finmindtrade.com/api/v4/data"
    result = {"symbol": clean_symbol, "institutional": {}, "major_holders": {}}

    try:
        params_inst = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": clean_symbol, "start_date": start_date}
        res_inst = requests.get(url, params=params_inst, timeout=5).json()

        if res_inst.get("data"):
            latest_date = res_inst["data"][-1]["date"]
            day_data = [d for d in res_inst["data"] if d["date"] == latest_date]
            summary = {}
            for item in day_data:
                name = item["name"]
                net_buy = (item["buy"] - item["sell"]) // 1000
                summary[name] = summary.get(name, 0) + net_buy

            result["institutional"] = {
                "date": latest_date,
                "foreign_buy_share": summary.get("Foreign_Investor", 0),
                "trust_buy_share": summary.get("Investment_Trust", 0),
                "dealer_buy_share": summary.get("Dealer_Self", 0) + summary.get("Dealer_Hedging", 0)
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
        news_items = [{"title": e.title, "published": getattr(e, 'published', 'N/A')} for e in feed.entries[:5]]
        return {"symbol": clean_symbol, "news": news_items}
    except Exception as e:
        return {"error": f"抓取新聞時發生錯誤: {str(e)}"}


def scan_market_opportunities(market: str = "TW") -> dict:
    """自動掃描熱門選股池，根據技術面動能與均線多頭排列過濾出潛力標的。"""
    watch_list = ["2330.TW", "2454.TW", "2317.TW", "2382.TW", "3231.TW", "2308.TW", "2379.TW"] if market.upper() == "TW" else ["AAPL", "NVDA", "TSLA", "MSFT", "AMD", "GOOGL", "AMZN"]
    opportunities = []

    for symbol in watch_list:
        try:
            df = yf.download(symbol, period="6mo", progress=False)
            if df.empty or len(df) < 30: continue
            close = df["Close"].squeeze()
            sma5 = ta.sma(close, length=5).iloc[-1]
            sma20 = ta.sma(close, length=20).iloc[-1]
            rsi14 = ta.rsi(close, length=14).iloc[-1]

            curr_price = float(close.iloc[-1])
            if curr_price > sma20 and sma5 > sma20 and 48 <= rsi14 <= 75:
                opportunities.append({
                    "symbol": symbol,
                    "current_price": round(curr_price, 2),
                    "sma5": round(float(sma5), 2),
                    "sma20": round(float(sma20), 2),
                    "rsi14": round(float(rsi14), 2)
                })
        except Exception:
            continue

    return {"market": market, "matched_count": len(opportunities), "candidates": opportunities[:5]}


def run_backtest(symbol: str, strategy: str = "sma_cross") -> dict:
    """使用 vectorbt 執行歷史數據量化回測，計算勝率、獲利因子、最大回撤 (MDD) 與總報酬率。"""
    clean_symbol = normalize_symbol(symbol)
    try:
        df = yf.download(clean_symbol, period="2y", progress=False)
        if df.empty or "Close" not in df:
            return {"error": f"找不到 {clean_symbol} 的歷史價格資料。"}

        price = df["Close"].squeeze()
        if strategy == "rsi_signal":
            rsi = ta.rsi(price, length=14)
            entries, exits = rsi < 30, rsi > 70
        else:
            fast_ma = ta.sma(price, length=5)
            slow_ma = ta.sma(price, length=20)
            entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        portfolio = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100000, fees=0.001425)

        def safe_float(val, multiplier=1.0, default=0.0):
            try:
                raw_val = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
                final_val = raw_val * multiplier
                return default if (math.isnan(final_val) or math.isinf(final_val)) else round(final_val, 2)
            except Exception:
                return default

        total_trades = int(portfolio.trades.count())
        win_rate = safe_float(portfolio.trades.win_rate(), multiplier=100.0) if total_trades > 0 else 0.0
        profit_factor = safe_float(portfolio.profit_factor()) if total_trades > 0 else 0.0
        max_dd = safe_float(portfolio.max_drawdown(), multiplier=100.0)
        total_return = safe_float(portfolio.total_return(), multiplier=100.0)

        return {
            "symbol": clean_symbol,
            "strategy": strategy,
            "total_trades": total_trades,
            "win_rate_pct": f"{win_rate}%",
            "profit_factor": profit_factor,
            "max_drawdown_mdd_pct": f"{max_dd}%",
            "total_return_pct": f"{total_return}%"
        }
    except Exception as e:
        return {"error": f"執行量化回測時發生錯誤: {str(e)}"}

def generate_daily_portfolio_report() -> dict:
    """生成當前持股組合的完整日報，包含個股技術指標、投資組合風險集中度與弱點診斷。"""
    print(f"🛠️ [Tool Called] 呼叫 generate_daily_portfolio_report")
    
    # 1. 取得當前持股
    port_data = get_portfolio_summary()
    if "portfolio" not in port_data or not port_data["portfolio"]:
        return {"error": "目前無持股，無法產生持股日報。"}
        
    report_details = []
    
    # 2. 針對每一檔持股，抓取最新的技術面狀態
    for item in port_data["portfolio"]:
        sym = item["symbol"]
        tech_data = fetch_stock_data(sym)
        report_details.append({
            "position": item,
            "technical": tech_data
        })
        
    # 3. 回傳綜合報告給 AI 分析
    return {
        "overall_summary": port_data["summary"],
        "holding_details": report_details
    }

# ==========================================
# 更新 2：包含所有工具與記憶機制的 UI 介面
# ==========================================
ALL_TOOLS = [
    record_trade, 
    get_portfolio_summary, 
    generate_daily_portfolio_report,
    fetch_stock_data, 
    fetch_chip_data, 
    fetch_stock_news, 
    scan_market_opportunities, 
    run_backtest
]

# ---------------------------------------------------------
# 5. Gemini AI 與 Streamlit Chat 介面
# ---------------------------------------------------------
ai_client = genai.Client(api_key=GEMINI_API_KEY)
system_instruction = "你是一位資深的量化交易員與 AI 個人資產管理顧問。請全程使用繁體中文，並以 Markdown 格式回覆。"

st.title("🤖 AI 量化股市助手 (Streamlit 版)")

# --- 側邊欄：清除記憶按鈕 ---
with st.sidebar:
    st.write("⚙️ 系統設定")
    if st.button("🗑️ 清除對話歷史"):
        # 清除 UI 上的文字紀錄
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 對話與記憶已清除！請問今天要查詢持股、回測策略，還是記錄交易呢？"}
        ]
        # 強制重新建立一個乾淨的 Chat Session
        if "chat_session" in st.session_state:
            del st.session_state.chat_session
        st.rerun()


# 初始化對話紀錄
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 你好！我是你的量化 AI 助手，已連接至 Supabase 雲端資料庫。請問今天要查詢持股、回測策略，還是記錄交易呢？"}
    ]

if "chat_session" not in st.session_state:
    # 將 Session 存入 session_state，讓它不會每次重整就被洗掉
    st.session_state.chat_session = ai_client.chats.create(
        model=AI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=ALL_TOOLS,  # 補全所有工具
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=5
            )
        )
    )

# --- 顯示歷史對話 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 處理使用者輸入 ---
if prompt := st.chat_input("請輸入指令 (例如: 幫我記錄買入 2330 1000股)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 AI 正在分析與存取資料庫..."):
            try:
                # 這裡直接呼叫保存在 session_state 中的 chat_session
                response = st.session_state.chat_session.send_message(prompt)
                
                final_text = response.text if response.text else "❌ AI 執行完畢，但未產出文字總結。"
                st.markdown(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                
            except Exception as e:
                error_msg = f"❌ 處理時發生錯誤：{str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
