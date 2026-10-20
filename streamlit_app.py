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
import tempfile
import uuid
from google import genai
from google.genai import types

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
MAX_TOKENS_LIMIT = 4000  # Token 安全上限設定

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
def save_chat_message(session_id: str, role: str, content: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (session_id, role, content) VALUES (%s, %s, %s)", (session_id, role, content))
    conn.commit()
    cursor.close()
    conn.close()

def load_chat_history(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_history WHERE session_id = %s ORDER BY id ASC", (session_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def get_all_sessions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, content, MIN(created_at) FROM chat_history WHERE role = 'user' GROUP BY session_id, content ORDER BY MIN(created_at) DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"session_id": row[0], "title": row[1][:15] + "..." if len(row[1])>15 else row[1]} for row in rows]

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
# 4. 財務與量化 Tools (與你原本邏輯相同)
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

def fetch_stock_data(symbol: str) -> dict:
    clean_symbol = normalize_symbol(symbol)
    try:
        ticker = yf.Ticker(clean_symbol)
        hist_daily = ticker.history(period="1y")
        if hist_daily.empty: return {"error": f"找不到股票代碼 {symbol} 的數據。"}
        hist_daily.ta.sma(length=5, append=True)
        hist_daily.ta.sma(length=20, append=True)
        hist_daily.ta.rsi(length=14, append=True)
        latest_d = hist_daily.iloc[-1]
        return {
            "symbol": clean_symbol,
            "close_price": round(float(latest_d['Close']), 2),
            "sma5": round(float(latest_d.get('SMA_5', 0)), 2),
            "sma20": round(float(latest_d.get('SMA_20', 0)), 2),
            "rsi_14": round(float(latest_d.get('RSI_14', 0)), 2)
        }
    except Exception as e:
        return {"error": str(e)}

def scan_market_opportunities(market: str = "TW") -> dict:
    watch_list = ["2303.TW", "2481.TW", "3264.TW", "3533.TW"] if market.upper() == "TW" else ["AAPL", "NVDA", "TSLA"]
    opportunities = []
    for symbol in watch_list:
        try:
            df = yf.download(symbol, period="6mo", progress=False)
            if df.empty or len(df) < 30: continue
            close = df["Close"].squeeze()
            sma5, sma20 = ta.sma(close, length=5).iloc[-1], ta.sma(close, length=20).iloc[-1]
            if float(close.iloc[-1]) > sma20 and sma5 > sma20:
                opportunities.append({"symbol": symbol, "price": round(float(close.iloc[-1]), 2)})
        except: continue
    return {"market": market, "candidates": opportunities}

def run_backtest(symbol: str, strategy: str = "sma_cross") -> dict:
    clean_symbol = normalize_symbol(symbol)
    try:
        df = yf.download(clean_symbol, period="2y", progress=False)
        if df.empty: return {"error": "無資料"}
        price = df["Close"].squeeze()
        fast_ma, slow_ma = ta.sma(price, length=5), ta.sma(price, length=20)
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        portfolio = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100000)
        return {"symbol": clean_symbol, "win_rate_pct": f"{round(float(portfolio.trades.win_rate())*100,2)}%", "total_return_pct": f"{round(float(portfolio.total_return())*100,2)}%"}
    except Exception as e: return {"error": str(e)}

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
st.title("🤖 AI 量化股市助手 (全域記憶版)")

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.messages = [{"role": "assistant", "content": "👋 你好！已連線至 Supabase。請問今天想檢視 2303 聯電的五分K走勢，還是記錄交易呢？"}]

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
            if st.button(f"📄 {s['title']}", key=f"load_{s['session_id']}", use_container_width=True):
                st.session_state.current_session_id = s['session_id']
                st.session_state.messages = load_chat_history(s['session_id'])
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{s['session_id']}"):
                delete_session(s['session_id'])
                if st.session_state.current_session_id == s['session_id']:
                    st.session_state.current_session_id = str(uuid.uuid4())
                    st.session_state.messages = [{"role": "assistant", "content": "👋 紀錄已清除。"}]
                st.rerun()
                
    st.divider()
    st.header("📎 附加檔案分析")
    uploaded_file = st.file_uploader("上傳 K線截圖或財報", type=["png", "jpg", "pdf", "csv"], label_visibility="collapsed")
    if uploaded_file: st.success(f"已暫存：{uploaded_file.name}")

# === 顯示歷史訊息 ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === 處理新訊息 ===
if prompt := st.chat_input("請輸入指令 (例如: 幫我分析這張技術線圖，或買入 2481 強茂 1000股)"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_message(st.session_state.current_session_id, "user", prompt)
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 正在精算 Token 與檢索全域記憶..."):
            try:
                # 準備發送的當前內容與檔案
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
                # 動態 Token 截斷與背景摘要機制
                # ==========================================
                history_messages = st.session_state.messages[:-1]
                
                # 確保結構完整，由 User 開頭
                if history_messages and history_messages[0]["role"] == "assistant":
                    history_messages = history_messages[1:]

                overflow_buffer = []
                while True:
                    gemini_history = build_gemini_history(history_messages)
                    full_contents = gemini_history + message_contents
                    
                    if not history_messages: break # 歷史已清空，直接跳出
                    
                    # 使用 API 試算 Token
                    token_info = ai_client.models.count_tokens(model=AI_MODEL, contents=full_contents)
                    if token_info.total_tokens <= MAX_TOKENS_LIMIT:
                        st.caption(f"📊 即時 Token 估算: {token_info.total_tokens} / {MAX_TOKENS_LIMIT}")
                        break
                        
                    # 超標，剃除最舊的一組對話 (一問一答) 進入溢出緩衝區
                    overflow_buffer.extend(history_messages[:2])
                    history_messages = history_messages[2:]

                # 若有剔除的對話，執行背景摘要並更新全域記憶
                if overflow_buffer:
                    curr_global_mem = load_global_memory()
                    new_global_mem = summarize_overflow(overflow_buffer, curr_global_mem)
                    update_global_memory(new_global_mem)
                    st.toast("🧠 AI 已將舊對話轉為全域記憶保存", icon="💡")

                # ==========================================
                # 建立攜帶「全域記憶」與「截斷後歷史」的 Session
                # ==========================================
                global_memory_context = load_global_memory()
                dynamic_system_prompt = f"你是一位量化交易員。請全程使用繁體中文，善用 Tools。\\n【不可遺忘的全域記憶】：\\n{global_memory_context}"
                
                chat_session = ai_client.chats.create(
                    model=AI_MODEL,
                    history=build_gemini_history(history_messages),
                    config=types.GenerateContentConfig(
                        system_instruction=dynamic_system_prompt,
                        tools=[record_trade, get_portfolio_summary, fetch_stock_data, scan_market_opportunities, run_backtest],
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

            except Exception as e:
                st.error(f"❌ 錯誤：{str(e)}")
            finally:
                if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)
                if gemini_file:
                    try: ai_client.files.delete(name=gemini_file.name)
                    except: pass
