import os
import math
import datetime
import urllib.parse
import requests
import feedparser
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.io as pio
import psycopg2
import streamlit as st
import subprocess
import tempfile
import uuid
import vectorbt as vbt
import yt_dlp
from google import genai
from google.genai import types
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

# ---------------------------------------------------------
# 1. 網頁基本設定與資安防護
# ---------------------------------------------------------
st.set_page_config(page_title="AI 量化資產管理助手", page_icon="📈", layout="wide")

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
    st.stop()

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_DB_URL"]
AI_MODEL = "gemini-3.5-flash-lite"
MAX_TOKENS_LIMIT = 50000  # Token 安全上限設定

# ---------------------------------------------------------
# 2. 資料庫初始化 (新增聊天紀錄與全域記憶表)
# ---------------------------------------------------------
def get_db_connection():
    return psycopg2.connect(SUPABASE_URL)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 持股與交易表
    cursor.execute('''CREATE TABLE IF NOT EXISTS positions (symbol TEXT PRIMARY KEY, shares REAL NOT NULL, avg_cost REAL NOT NULL, updated_at TEXT NOT NULL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS trade_logs (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, action TEXT NOT NULL, shares REAL NOT NULL, price REAL NOT NULL, trade_date TEXT NOT NULL, notes TEXT)''')
    
    # 聊天紀錄表 (多 Session)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 全域記憶表 (跨 Session 互通的長期摘要)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS global_memory (
            id SERIAL PRIMARY KEY,
            summary_text TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # chart_json欄位，用來存 Plotly 圖表的 JSON
    cursor.execute('''
        ALTER TABLE chat_history 
        ADD COLUMN IF NOT EXISTS chart_json TEXT
    ''')
    
    # 若全域記憶為空，插入預設值
    cursor.execute("SELECT COUNT(*) FROM global_memory")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO global_memory (summary_text) VALUES ('目前尚無特殊全域記憶設定。')")
        
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

# ---------------------------------------------------------
# 3. Session & 記憶管理 DB 操作
# ---------------------------------------------------------
def save_chat_message(session_id: str, role: str, content: str, chart_json: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO chat_history (session_id, role, content, chart_json) VALUES (%s, %s, %s, %s)",
        (session_id, role, content, chart_json)
    )
    
    conn.commit()
    cursor.close()
    conn.close()

def load_chat_history(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, chart_json FROM chat_history WHERE session_id = %s ORDER BY id ASC",
        (session_id,)
    )    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    # 將撈出來的資料打包成 Streamlit 看得懂的 dict
    return [{"role": row[0], "content": row[1], "chart_json": row[2]} for row in rows]

def get_all_sessions():
    """撈取所有歷史對話，確保每個 session_id 唯一，並取第一句話當作標題"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 按照時間由舊到新撈出所有使用者的對話
    cursor.execute("SELECT session_id, content, created_at FROM chat_history WHERE role = 'user' ORDER BY created_at ASC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # 利用 Python 字典過濾，確保同一個 session_id 只保留最一開始的那句話
    session_dict = {}
    for row in rows:
        sid = row[0]
        content = row[1]
        if sid not in session_dict:
            session_dict[sid] = content
            
    # 將結果轉換為串列，並反轉順序 (最新的對話排在側邊欄最上面)
    result = []
    for sid, content in reversed(list(session_dict.items())):
        result.append({
            "session_id": sid, 
            "title": content[:15] + "..." if len(content) > 15 else content
        })
        
    return result
    
def delete_session(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE session_id = %s", (session_id,))
    conn.commit()
    cursor.close()
    conn.close()

def load_global_memory():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT summary_text FROM global_memory ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row else "目前尚無特殊全域記憶設定。"

def update_global_memory(new_summary: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    # 這裡採直接覆寫最新一筆的方式
    cursor.execute("UPDATE global_memory SET summary_text = %s, updated_at = CURRENT_TIMESTAMP WHERE id = (SELECT id FROM global_memory ORDER BY id DESC LIMIT 1)", (new_summary,))
    conn.commit()
    cursor.close()
    conn.close()

# ---------------------------------------------------------
# 畫視覺畫圖形 Tool
# ---------------------------------------------------------
def show_stock_chart(symbol: str, period: str = "6mo") -> dict:
    """
    當使用者要求畫 K 線圖、走勢圖時呼叫此工具。
    Args:
        symbol: 股票代碼 (例如 '2330', 'AAPL')
        period: 資料長度。有效值必須為: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'。預設為 '6mo'。
    """
    clean_symbol = normalize_symbol(symbol)
    
    # 把代碼和「時間長度」一起存進網頁暫存記憶體
    st.session_state.render_chart_for = clean_symbol
    st.session_state.render_chart_period = period 
    
    print(f"🛠️ [Tool Called] 呼叫 show_stock_chart | 渲染: {clean_symbol}, 期間: {period}")
    
    return {
        "status": "success", 
        "message": f"已成功觸發網頁前端繪製 {clean_symbol} 的 {period} K 線圖。"
    }

# ---------------------------------------------------------
#  影片內容分析 Tool
# ---------------------------------------------------------
def fetch_youtube_transcript(url: str) -> dict:
    """
    獲取 YouTube 影片字幕。
    具備 CC 字幕秒抓機制，與超過 24MB 大檔案音軌自動切割 (ffmpeg) 轉 Whisper API 的雙層備援防呆。
    """
    # 1. 解析 Video ID
    video_id = None
    if "v=" in url:
        video_id = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("v", [None])[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
        
    if not video_id:
        return {"error": "無法解析 YouTube 影片 ID，請確認網址格式。"}

    # 2. 優先嘗試官方 CC 字幕
    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id, 
            languages=['zh-TW', 'zh-Hant', 'zh-Hans', 'zh-CN', 'en']
        )
        full_text = " ".join([t['text'] for t in transcript])
        return {
            "status": "success", 
            "method": "cc_subtitle",
            "message": "成功取得官方 CC 字幕", 
            "transcript_summary": full_text
        }
    except Exception:
        pass # 找不到官方字幕，默默進入下方 Whisper 備援
        
    # 3. 備援機制：下載音軌並依據大小決定是否切割
    try:
        openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, 'audio.m4a')
            
            # 僅下載 m4a 音軌
            ydl_opts = {
                'format': 'm4a/bestaudio/best',
                'outtmpl': audio_path,
                'quiet': True,
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'm4a'}]
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            file_size = os.path.getsize(audio_path)
            max_size = 24 * 1024 * 1024 # 設定安全閥值為 24MB
            full_transcript = ""

            if file_size > max_size:
                # 檔案過大，呼叫系統層級的 ffmpeg 無損切割 (每 600 秒 = 10 分鐘切一軌)
                segment_pattern = os.path.join(tmpdir, "chunk_%03d.m4a")
                subprocess.run([
                    "ffmpeg", "-i", audio_path,
                    "-f", "segment", "-segment_time", "600",
                    "-c", "copy", segment_pattern
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 找出所有切割後的檔案並排序
                chunk_files = sorted([f for f in os.listdir(tmpdir) if f.startswith("chunk_") and f.endswith(".m4a")])

                for chunk_file in chunk_files:
                    chunk_path = os.path.join(tmpdir, chunk_file)
                    with open(chunk_path, "rb") as audio_file:
                        res = openai_client.audio.transcriptions.create(
                            model="whisper-1", 
                            file=audio_file,
                            response_format="text"
                        )
                        full_transcript += res + " "
            else:
                # 檔案小於 24MB，直接送入 Whisper
                with open(audio_path, "rb") as audio_file:
                    full_transcript = openai_client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file,
                        response_format="text"
                    )

            return {
                "status": "success",
                "method": "whisper_api_split" if file_size > max_size else "whisper_api",
                "message": "成功透過 Whisper 取回完整逐字稿",
                "transcript_summary": full_transcript.strip()
            }
    except Exception as e:
        return {"error": f"無法獲取字幕且音軌辨識失敗: {str(e)}"}

# ---------------------------------------------------------
# 4. 財務與量化 Tools
# ---------------------------------------------------------
def record_trade(symbol: str, action: str, shares: float, price: float, notes: str = "") -> dict:
    # 移除自動市價，強制要求價格，避免 AI 產生幻覺重覆買進
    clean_symbol = normalize_symbol(symbol)
    action = action.upper()
    if action not in ['BUY', 'SELL']: return {"error": "交易動作必須為 BUY 或 SELL"}
    if price is None or price <= 0: return {"error": "請告知使用者必須提供具體的成交價格。"}

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO trade_logs (symbol, action, shares, price, trade_date, notes) VALUES (%s, %s, %s, %s, %s, %s)", (clean_symbol, action, shares, price, now_str, notes))
        cursor.execute("SELECT shares, avg_cost FROM positions WHERE symbol = %s", (clean_symbol,))
        pos = cursor.fetchone()
        
        if action == "BUY":
            if pos:
                new_shares = pos[0] + shares
                new_cost = ((pos[0] * pos[1]) + (shares * price)) / new_shares
                cursor.execute("UPDATE positions SET shares = %s, avg_cost = %s, updated_at = %s WHERE symbol = %s", (new_shares, round(new_cost, 2), now_str, clean_symbol))
            else:
                cursor.execute("INSERT INTO positions (symbol, shares, avg_cost, updated_at) VALUES (%s, %s, %s, %s)", (clean_symbol, shares, price, now_str))
        elif action == "SELL":
            if not pos: return {"error": "無持股紀錄"}
            if shares >= pos[0]:
                cursor.execute("DELETE FROM positions WHERE symbol = %s", (clean_symbol,))
            else:
                cursor.execute("UPDATE positions SET shares = %s, updated_at = %s WHERE symbol = %s", (pos[0] - shares, now_str, clean_symbol))
        conn.commit()
        return {"status": "success", "message": f"成功紀錄 {action} {clean_symbol} {shares} 股"}
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
    total_cost = total_market_val = 0.0
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
        portfolio.append({"symbol": symbol, "shares": shares, "avg_cost": avg_cost, "current_price": curr_price, "unrealized_pnl": round(mkt_val - cost_val, 2)})
    return {"summary": {"total_cost": round(total_cost, 2), "total_unrealized_pnl": round(total_market_val - total_cost, 2)}, "portfolio": portfolio}

def fetch_stock_data(symbol: str, interval: str = "1d", period: str = "1y") -> dict:
    """
    獲取股票的歷史數據與技術指標。
    - 支援相對時間 period: "1y", "6mo", "1mo"
    - 也支援絕對日期 start_date 與 end_date: 格式為 "YYYY-MM-DD" (例如 "2026-09-01")
    若有提供 start_date，將優先使用精確日期區間。
    """
    clean_symbol = normalize_symbol(symbol)
    try:
        ticker = yf.Ticker(clean_symbol)
        
        # 核心修改：判斷是否有傳入指定日期
        if start_date:
            # 若沒有提供 end_date，yfinance 預設會抓到今天
            df = ticker.history(start=start_date, end=end_date, interval=interval)
        else:
            # 防呆機制：yfinance 的分 K 資料 (含有 m) 最多只能抓最近 60 天
            if "m" in interval and period in ["1y", "2y", "max", "ytd"]:
                period = "60d"
            df = ticker.history(period=period, interval=interval)
        
        # 處理上櫃股票後綴
        if df.empty and clean_symbol.endswith(".TW"):
            clean_symbol = clean_symbol.replace(".TW", ".TWO")
            ticker = yf.Ticker(clean_symbol)
            if start_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)
                
        if df.empty:
            return {"error": f"找不到 {symbol} 的數據，請確認日期區間內是否有交易日。"}

        # 計算技術指標 (均線、RSI)
        df.ta.sma(length=5, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=60, append=True)
        df.ta.rsi(length=14, append=True)

        # 新增：計算 KD 指標 (Stochastic Oscillator)
        # k=9, d=3, smooth_k=3 是一般台股看盤軟體常用的 933 參數
        kd = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3, smooth_k=3)
        if kd is not None and not kd.empty:
            df = pd.concat([df, kd], axis=1)

        latest_d = df.iloc[-1]
        prev_d = df.iloc[-2] if len(df) > 1 else latest_d

        # 漲跌幅
        change_pct = ((latest_d['Close'] - prev_d['Close']) / prev_d['Close']) * 100

        # 安全取值函式，避免 NaN 報錯
        def safe_get(key):
            val = latest_d.get(key)
            return round(float(val), 2) if pd.notna(val) else "N/A"

        return {
            "symbol": clean_symbol,
            "data_interval": interval,  # 讓 AI 確認它目前拿到的是什麼層級的資料
            "latest_time": str(df.index[-1]),
            "close_price": round(float(latest_d['Close']), 2),
            "change_percent": f"{round(float(change_pct), 2)}%",
            "moving_averages": {
                "sma5": safe_get('SMA_5'),
                "sma20": safe_get('SMA_20'),
                "sma60": safe_get('SMA_60'),
            },
            "momentum_indicators": {
                "rsi_14": safe_get('RSI_14'),
                "kd_k9": safe_get('STOCHk_9_3_3'),  # KD 的 K 值
                "kd_d3": safe_get('STOCHd_9_3_3')   # KD 的 D 值
            }
        }
    except Exception as e:
        return {"error": f"抓取技術面數據時發生錯誤: {str(e)}"}
        
def fetch_chip_data(symbol: str) -> dict:
    """獲取台股的三大法人買賣超數據。"""
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

def scan_market_opportunities(market: str = "TW", custom_symbols: list = None, strategy: str = "5m_kd") -> dict:
    """
    掃描選股池，尋找潛力標的。
    - custom_symbols: 允許使用者指定股票代碼清單 (如 ["2330", "2303"])。未提供則使用預設名單。
    - strategy: 支援 "5m_kd" (五分K+KD交叉短線策略) 與 "daily_rsi" (日線+RSI波段策略)。
    """
    if custom_symbols and isinstance(custom_symbols, list) and len(custom_symbols) > 0:
        watch_list = [normalize_symbol(s) for s in custom_symbols]
    else:
        watch_list = ["2303.TW", "2481.TW", "5328.TW", "3264.TW", "3533.TW"] if market.upper() == "TW" else ["AAPL", "NVDA", "TSLA"]
    
    opportunities = []

    for symbol in watch_list:
        try:
            if strategy == "5m_kd":
                # 【短線策略】五分K戰法 + KD指標
                df = yf.download(symbol, period="5d", interval="5m", progress=False)
                if df.empty or len(df) < 30: continue
                
                close, high, low = df["Close"].squeeze(), df["High"].squeeze(), df["Low"].squeeze()
                sma5, sma20 = ta.sma(close, length=5).iloc[-1], ta.sma(close, length=20).iloc[-1]
                kd = ta.stoch(high, low, close, k=9, d=3, smooth_k=3)
                
                if kd is None or kd.empty: continue
                k_val, d_val = kd["STOCHk_9_3_3"].iloc[-1], kd["STOCHd_9_3_3"].iloc[-1]
                curr_price = float(close.iloc[-1])
                
                # 濾網條件：均線多頭且 KD 黃金交叉且未嚴重超買
                if curr_price > sma20 and sma5 > sma20 and k_val > d_val and k_val < 80:
                    opportunities.append({
                        "symbol": symbol, "current_price": round(curr_price, 2),
                        "sma5": round(float(sma5), 2), "sma20": round(float(sma20), 2),
                        "k_value": round(float(k_val), 2), "d_value": round(float(d_val), 2)
                    })
                    
            elif strategy == "daily_rsi":
                # 【波段策略】日線 + RSI指標
                df = yf.download(symbol, period="6mo", interval="1d", progress=False)
                if df.empty or len(df) < 30: continue
                
                close = df["Close"].squeeze()
                sma5, sma20 = ta.sma(close, length=5).iloc[-1], ta.sma(close, length=20).iloc[-1]
                rsi14 = ta.rsi(close, length=14).iloc[-1]
                curr_price = float(close.iloc[-1])
                
                # 濾網條件：均線多頭且 RSI 處於動能加溫區 (48~75)
                if curr_price > sma20 and sma5 > sma20 and 48 <= rsi14 <= 75:
                    opportunities.append({
                        "symbol": symbol, "current_price": round(curr_price, 2),
                        "sma5": round(float(sma5), 2), "sma20": round(float(sma20), 2),
                        "rsi14": round(float(rsi14), 2)
                    })
        except Exception:
            continue

    return {"market": market, "strategy": strategy, "scanned_count": len(watch_list), "matched_count": len(opportunities), "candidates": opportunities}

def run_backtest(symbol: str, strategy: str = "sma_cross", trailing_stop_pct: float = 0.08) -> dict:
    """
    執行量化回測。
    - period 支援: "1mo", "3mo", "6mo", "1y", "2y", "max" 等。
    注意：若 interval 為分K (如 5m)，period 強制最高只能設為 "60d"。
    """
    clean_symbol = normalize_symbol(symbol)
    
    # 防呆機制：yfinance 限制分K最多只能抓 60 天
    if "m" in interval and period in ["1y", "2y", "max", "ytd", "5y", "10y"]:
        period = "60d"
    try:
        # 下載資料時帶入動態的 period
        df = yf.download(clean_symbol, period=period, interval=interval, progress=False)
        if df.empty or "Close" not in df:
            return {"error": f"找不到 {clean_symbol} 的歷史價格資料。"}

        price = df["Close"].squeeze()
        
        # 產生進出場訊號
        if strategy == "rsi_signal":
            rsi = ta.rsi(price, length=14)
            entries = rsi < 30
            exits = rsi > 70
        else:
            fast_ma = ta.sma(price, length=5)
            slow_ma = ta.sma(price, length=20)
            entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        # 核心升級：加入移動停損與手續費設定
        portfolio = vbt.Portfolio.from_signals(
            price, 
            entries, 
            exits, 
            init_cash=100000, 
            fees=0.001425,
            sl_stop=trailing_stop_pct,  # 設定停損比例 (預設 0.08 代表 8%)
            sl_trail=True               # 啟動移動停損，隨最高價動態上移出場線
        )

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
            "trailing_stop_applied": f"{trailing_stop_pct * 100}%",
            "total_trades": total_trades,
            "win_rate_pct": f"{win_rate}%",
            "profit_factor": profit_factor,
            "max_drawdown_mdd_pct": f"{max_dd}%",
            "total_return_pct": f"{total_return}%"
        }
    except Exception as e:
        return {"error": f"執行量化回測時發生錯誤: {str(e)}"}

# ---------------------------------------------------------
# 5. Gemini 格式轉換與自動摘要工具
# ---------------------------------------------------------
ai_client = genai.Client(api_key=GEMINI_API_KEY)

def build_gemini_history(messages):
    """將 Session state 轉換為 Gemini 原生結構"""
    gemini_history = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    return gemini_history

def summarize_overflow(overflow_msgs: list, current_summary: str) -> str:
    """背景執行：提煉舊對話重點並融合進全域記憶"""
    old_text = "\n".join([f"{m['role']}: {m['content']}" for m in overflow_msgs])
    prompt = f"""
    你負責整理用戶長期記憶。請將以下被捨棄的舊對話濃縮，保留與「投資偏好、持有標的、關鍵決策、交易紀律」相關的事實。
    將新重點與現有記憶融合，不要廢話。
    【現有全域記憶】：{current_summary}
    【捨棄的舊對話】：
    {old_text}
    """
    try:
        res = ai_client.models.generate_content(model=AI_MODEL, contents=prompt)
        return res.text.strip()
    except Exception as e:
        print(f"摘要失敗: {e}")
        return current_summary

# ---------------------------------------------------------
# 6. Streamlit 介面與主程式邏輯
# ---------------------------------------------------------
# 1. 獨立定義載入對話的 Callback 邏輯
def handle_load_session(session_id):
    st.session_state.current_session_id = session_id
    st.session_state.messages = load_chat_history(session_id)

# 2. 獨立定義刪除對話的 Callback 邏輯
def handle_delete_session(session_id):
    delete_session(session_id) # 執行資料庫刪除
    # 如果刪除的剛好是目前正在看的對話，就重置畫面
    if st.session_state.current_session_id == session_id:
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.messages = [{"role": "assistant", "content": "👋 紀錄已清除。"}]
        
st.title("🤖 AI 量化股市助手 (V2.0.2)")

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.messages = [{"role": "assistant", "content": "👋 你好！已連線至 Supabase。請問今天想檢視 2303 聯電的五分K走勢，還是記錄交易呢?"}]

# 初始化 Token 累計變數
if "" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0

# === 側邊欄：Session 管理與檔案上傳 ===
with st.sidebar:
    st.header("💬 對話紀錄")
    if st.button("➕ 建立新對話", use_container_width=True):
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.messages = [{"role": "assistant", "content": "👋 已開啟新對話區塊，全域記憶已同步載入！"}]
        st.rerun()
        
    st.divider()
    past_sessions = get_all_sessions()
    for s in past_sessions:
        col1, col2 = st.columns([4, 1])
        with col1:
            # 使用 on_click 觸發函式，並透過 args 傳入參數
            st.button(
                f"📄 {s['title']}", 
                key=f"load_{s['session_id']}", 
                use_container_width=True,
                on_click=handle_load_session,
                args=(s['session_id'],)
            )
            
        with col2:
            st.button(
                "🗑️", 
                key=f"del_{s['session_id']}",
                on_click=handle_delete_session,
                args=(s['session_id'],)
            )
                
    st.divider()
    st.header("📎 附加檔案分析")
    uploaded_file = st.file_uploader("上傳任何檔案 (如影片、音檔、圖片、程式碼)", label_visibility="collapsed")
    if uploaded_file: st.success(f"已暫存：{uploaded_file.name}")

    st.markdown("---")
    st.markdown("### 📊 API 流量與成本監控")
    
    # 顯示即時累計數據
    col1, col2 = st.columns(2)
    col1.metric("總 Input Tokens", f"{st.session_state.:,}")
    col2.metric("總 Output Tokens", f"{st.session_state.total_output_tokens:,}")
    
    # 計算預估成本 (USD)
    estimated_cost = (st.session_state. / 1000000 * 0.075) + \
                     (st.session_state.total_output_tokens / 1000000 * 0.3)
    
    st.metric("預估花費 (USD)", f"${estimated_cost:.5f}")
    
    # 加上一個重置按鈕
    if st.button("🔄 重置計數器", use_container_width=True):
        st.session_state. = 0
        st.session_state.total_output_tokens = 0
        st.rerun()

# === 顯示歷史訊息 ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # 如果這筆訊息帶有圖表資料，就把它還原並畫出來！
        if msg.get("chart_json"):
            try:
                fig = pio.from_json(msg["chart_json"])
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error("圖表還原失敗")

# === 處理新訊息 ===
if prompt := st.chat_input("請輸入指令 (例如: 幫我分析這張技術線圖，或買入 2481 強茂 1000股)"):
   
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_message(st.session_state.current_session_id, "user", prompt)
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 正在檢索數據與思考中..."):
            try:
                # 1. 準備發送的當前內容與檔案
                message_contents = [prompt]
                gemini_file = None
                tmp_path = None
                
                if uploaded_file:
                    file_ext = uploaded_file.name.split('.')[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    gemini_file = ai_client.files.upload(file=tmp_path, config={'display_name': uploaded_file.name})
                    message_contents.append(gemini_file)
    
                # ==========================================
                # 2. 固定輪數截斷與單次 Token 計算 (徹底消滅 while 迴圈)
                # ==========================================
                MAX_HISTORY_MESSAGES = 30 
                history_messages = st.session_state.messages[:-1].copy() # 複製一份，不動到原始 UI 資料
                overflow_buffer = []
                
                if len(history_messages) > MAX_HISTORY_MESSAGES:
                    overflow_buffer = history_messages[:-MAX_HISTORY_MESSAGES]
                    history_messages = history_messages[-MAX_HISTORY_MESSAGES:] # 這裡只改變傳給 AI 的陣列
    
                # 確保結構完整，由 User 開頭
                if history_messages and history_messages[0]["role"] == "assistant":
                    if len(history_messages) > 1:
                        overflow_buffer.append(history_messages[0])
                    history_messages = history_messages[1:]
    
                # 若有剔除的對話，執行背景摘要並更新全域記憶
                if overflow_buffer:
                    curr_global_mem = load_global_memory()
                    new_global_mem = summarize_overflow(overflow_buffer, curr_global_mem)
                    update_global_memory(new_global_mem)
                    st.toast("🧠 AI 已將舊對話轉為全域記憶保存", icon="💡")
    
                gemini_history = build_gemini_history(history_messages)
                
                # 單次計算 Token，絕不卡頓
                try:
                    token_info = ai_client.models.count_tokens(
                        model=AI_MODEL, 
                        contents=gemini_history + message_contents
                    )
                    st.caption(f"📊 本次請求預估消耗 Token: {token_info.total_tokens}")
                except Exception:
                    pass
    
                # ==========================================
                # 3. 建立攜帶全域記憶的 Session 並送出
                # ==========================================
                global_memory_context = load_global_memory()
                dynamic_system_prompt = (f"你是一位量化交易員。請全程使用繁體中文，請務必善用 tools。\\n【不可遺忘的全域記憶】：\\n{global_memory_context} \n"
                "【專業分析規範】\n"
                "1. **市場掃描與交易機會 (需求 1)**：\n"
                "   * 當使用者要求「掃描市場」或「尋找交易機會」時，呼叫 `scan_market_opportunities`。\n"
                "   * 挑出高勝率標的，**必須為每檔標的精確計算出**：\n"
                "     - 🎯 建議進場價格區間\n"
                "     - 🚀 出場目標價 (Target)\n"
                "     - 🛡️ 停損價格 (Stop Loss)\n"
                "     - ⚖️ 風險報酬比 (Risk/Reward Ratio，需至少 1:1.5 以上)\n"
                "     - 💡 技術面與動能成立理由。\n"
                "2. **策略量化回測分析 (需求 4)**：\n"
                "   * 當使用者要求「驗證策略」、「回測」時，**務必呼叫 `run_backtest` 工具獲取精確數據**，嚴禁憑空猜測數字！\n"
                "   * 根據工具回傳的真實數字（勝率、獲利因子 Profit Factor、最大回撤 MDD），分析該策略優缺點，並提出 2~3 條具體的改進建議（如加入停損機制、濾網或動能指標）。\n\n"
                "【極重要工具使用規則】\n"
                "1. 當使用者要求『歷史回測』或『策略驗證』時，你【只能且必須只呼叫 `run_backtest`】！嚴禁同時呼叫其他任何工具 (如 fetch_stock_data 或 news)。\n"
                "2. 當使用者要求『市場掃描』時，你【只能且必須只呼叫 `scan_market_opportunities`】！\n"
                "3. 當使用者查詢『單一個股行情』時，才呼叫 `fetch_stock_data` 或 `fetch_chip_data`。\n"
                "4. 當使用者要求「畫圖」、「K線圖」或「走勢圖」時，你【必須】呼叫 `show_stock_chart` 工具！\n"
                "5. 呼叫畫圖工具後，【絕對禁止】道歉或說「受限於文字介面無法畫圖」。你必須以自信的語氣回覆：「已為您在畫面下方繪製最新的 K 線圖，請參考。」\n"
                "6. 當使用者貼上 YouTube 網址時，你【必須】呼叫 `fetch_youtube_transcript` 工具來獲取影片內容，並分析其中的財經觀點或提及的股票代碼。 \n\n"
                "【分析要求】\n"
                "• 執行 `run_backtest` 後，請根據回傳的勝率 (win_rate_pct)、獲利因子 (profit_factor) 與最大回撤 (max_drawdown_mdd_pct)，精確點出該策略優缺點，並給出 2~3 條具體改進建議。\n"
                "• 全程使用繁體中文，採用專業 Markdown 格式。")
                
                chat_session = ai_client.chats.create(
                    model=AI_MODEL,
                    history=gemini_history,
                    config=types.GenerateContentConfig(
                        system_instruction=dynamic_system_prompt,
                        tools=[record_trade, get_portfolio_summary, fetch_stock_data, fetch_chip_data, fetch_stock_news, scan_market_opportunities, run_backtest, show_stock_chart, fetch_youtube_transcript],
                        temperature=0.2,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=5)
                    )
                )
                
                # 發送請求
                response = chat_session.send_message(message_contents)
                final_text = response.text if response.text else "處理完畢。"
                
                st.markdown(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                save_chat_message(st.session_state.current_session_id, "assistant", final_text)

                # 🌟 新增：捕捉 Token 用量並累加
                if hasattr(response, 'usage_metadata'):
                    st.session_state.total_input_tokens += response.usage_metadata.prompt_token_count
                    st.session_state.total_output_tokens += response.usage_metadata.candidates_token_count
            
            except Exception as e:
                st.error(f"❌ 錯誤：{str(e)}")
            finally:
                if tmp_path and os.path.exists(tmp_path): 
                    os.remove(tmp_path)
                if gemini_file:
                    try: 
                        ai_client.files.delete(name=gemini_file.name)
                    except: 
                        pass

# 檢查 AI 是否有觸發畫圖指令
if "render_chart_for" in st.session_state and st.session_state.render_chart_for:
    chart_symbol = st.session_state.render_chart_for
    # 讀取 AI 決定的時間長度，如果沒給就預設 6mo
    chart_period = st.session_state.get("render_chart_period", "6mo") 
    
    st.markdown("---") 
    st.subheader(f"📊 {chart_symbol} K 線圖 (期間: {chart_period})")
    
    try:
        # 這裡把寫死的 "6mo" 換成動態變數 chart_period
        df = yf.download(chart_symbol, period=chart_period, progress=False)

        # 如果抓不到資料，且代碼是 .TW 結尾，自動切換成 .TWO (上櫃) 再抓一次
        if df.empty and chart_symbol.endswith(".TW"):
            chart_symbol = chart_symbol.replace(".TW", ".TWO")
            df = yf.download(chart_symbol, period=chart_period, progress=False)
        
        if not df.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'].squeeze(),
                high=df['High'].squeeze(),
                low=df['Low'].squeeze(),
                close=df['Close'].squeeze(),
                name=chart_symbol
            )])
            
            fig.update_layout(
                xaxis_rangeslider_visible=False, 
                template="plotly_dark",          
                margin=dict(l=20, r=20, t=20, b=20),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            ########### 將圖表轉json存入DB中 ##########
            # 1. 轉成 JSON 字串
            fig_json = fig.to_json()
            
            # 2. 顯示在畫面上
            # st.plotly_chart(fig)
            
            # 3. 將圖表 JSON 一併存入記憶與資料庫 (取代原本只存 content 的寫法)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_text, 
                "chart_json": fig_json  # <--- 把圖表掛在對話字典裡
            })
            save_chat_message(
                st.session_state.current_session_id, 
                "assistant", 
                final_text, 
                fig_json                # <--- 存進 Supabase
            )
            
        else:
            st.warning(f"無法獲取 {chart_symbol} 在 {chart_period} 期間的歷史資料。")
            
    except Exception as e:
        st.error(f"繪製圖表時發生錯誤：{str(e)}")
        
    # 畫完圖後清除暫存
    st.session_state.render_chart_for = None
    st.session_state.render_chart_period = None
