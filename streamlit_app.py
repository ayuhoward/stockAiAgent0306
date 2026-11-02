import json
import os
import tempfile
import uuid

import streamlit as st
from google import genai
from google.genai import types

from stock_tools.chat_history import (
    delete_session,
    get_all_sessions,
    initialize_chat_history_tables,
    load_chat_history,
    load_global_memory,
    save_chat_message,
    update_global_memory,
)
from stock_tools.charts import render_chart_queue, render_saved_chart, show_stock_chart
from stock_tools.common import get_db_connection
from stock_tools.market_data import (
    fetch_chip_data,
    fetch_fubon_holdings,
    fetch_stock_data,
    fetch_stock_news,
)
from stock_tools.portfolio import get_portfolio_summary, record_trade
from stock_tools.strategies import run_backtest, scan_market_opportunities
from stock_tools.youtube_tools import fetch_youtube_transcript

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
AI_MODEL = "gemini-3.5-flash-lite"
MAX_TOKENS_LIMIT = 50000  # Token 安全上限設定，超過會摘要至全域記憶
MAX_HISTORY_MESSAGES = 35 # Message 歷史上限設定，超過會摘要至全域記憶

# ---------------------------------------------------------
# 2. 資料庫初始化 (新增聊天紀錄與全域記憶表)
# ---------------------------------------------------------
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 持股與交易表
    cursor.execute('''CREATE TABLE IF NOT EXISTS positions (symbol TEXT PRIMARY KEY, shares REAL NOT NULL, avg_cost REAL NOT NULL, updated_at TEXT NOT NULL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS trade_logs (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, action TEXT NOT NULL, shares REAL NOT NULL, price REAL NOT NULL, trade_date TEXT NOT NULL, notes TEXT)''')
    
    initialize_chat_history_tables(cursor)
        
    conn.commit()
    cursor.close()
    conn.close()

init_db()

# ---------------------------------------------------------
# ---------------------------------------------------------
# Gemini 格式轉換與自動摘要工具
# ---------------------------------------------------------
ai_client = genai.Client(api_key=GEMINI_API_KEY)
AI_TOOL_FUNCTIONS = [
    record_trade,
    get_portfolio_summary,
    fetch_stock_data,
    fetch_fubon_holdings,
    fetch_chip_data,
    fetch_stock_news,
    scan_market_opportunities,
    run_backtest,
    show_stock_chart,
    fetch_youtube_transcript,
]
AI_TOOL_DECLARATIONS = [
    types.FunctionDeclaration.from_callable(client=ai_client, callable=tool)
    for tool in AI_TOOL_FUNCTIONS
]

def build_gemini_history(messages):
    """將 Session state 轉換為 Gemini 原生結構"""
    gemini_history = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    return gemini_history


def build_system_instruction(global_memory_context: str) -> str:
    return (
        f"你是一位量化交易員。請全程使用繁體中文，請務必善用 tools。\\n【不可遺忘的全域記憶】：\\n{global_memory_context} \n"
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
        "3. 當使用者查詢投信/外資/自營商的持股張數、持股比例或法人庫存時，必須只呼叫 `fetch_fubon_holdings`，不要呼叫 `fetch_chip_data`，也不要用 FinMind 區間買賣超推算持股。回答必須直接引用工具回傳的 `investment_trust` 數值，並註明 `data_date`、來源和快照不保證盤中即時；若工具回傳 error，明確告知抓取失敗，不得改用買賣超資料代替。\n"
        "4. 當使用者查詢單一個股的每日法人買進、賣出或淨買賣超/區間累計時，才呼叫 `fetch_chip_data`。FinMind 的每日買賣超不可描述成持股張數或持股比例。\n"
        "5. 當使用者查詢單一個股行情或技術指標時，呼叫 `fetch_stock_data`。\n"
        "6. 當使用者要求「畫圖」、「K線圖」或「走勢圖」時，你【必須】呼叫 `show_stock_chart` 工具！\n"
        "7. 呼叫畫圖工具後，【絕對禁止】道歉或說「受限於文字介面無法畫圖」。你必須以自信的語氣回覆：「已為您在畫面下方繪製最新的 K 線圖，請參考。」\n"
        "8. 當使用者貼上 YouTube 網址時，你【必須】呼叫 `fetch_youtube_transcript` 工具來獲取影片內容，並分析其中的財經觀點或提及的股票代碼。 \n\n"
        "【分析要求】\n"
        "• 執行 `run_backtest` 後，請根據回傳的勝率 (win_rate_pct)、獲利因子 (profit_factor) 與最大回撤 (max_drawdown_mdd_pct)，精確點出該策略優缺點，並給出 2~3 條具體改進建議。\n"
        "• 全程使用繁體中文，採用專業 Markdown 格式。"
    )


def count_request_tokens(history_messages: list, message_contents: list, system_instruction: str) -> int:
    contents = [
        system_instruction,
        json.dumps(
            [declaration.model_dump(exclude_none=True) for declaration in AI_TOOL_DECLARATIONS],
            ensure_ascii=False,
        ),
        *build_gemini_history(history_messages),
        *message_contents,
    ]
    token_info = ai_client.models.count_tokens(model=AI_MODEL, contents=contents)
    return token_info.total_tokens


def fit_history_to_token_limit(
    history_messages: list,
    message_contents: list,
    system_instruction: str,
) -> tuple[list, list, int]:
    token_count = count_request_tokens(history_messages, message_contents, system_instruction)
    if token_count <= MAX_TOKENS_LIMIT:
        return history_messages, [], token_count

    start_indexes = [
        index for index, message in enumerate(history_messages) if message["role"] == "user"
    ]
    start_indexes.append(len(history_messages))

    low = 0
    high = len(start_indexes) - 1
    best_fit = None
    for _ in range(len(start_indexes).bit_length()):
        if low > high:
            break
        middle = (low + high) // 2
        start_index = start_indexes[middle]
        candidate = history_messages[start_index:]
        candidate_tokens = count_request_tokens(candidate, message_contents, system_instruction)
        if candidate_tokens <= MAX_TOKENS_LIMIT:
            best_fit = (start_index, candidate, candidate_tokens)
            high = middle - 1
        else:
            low = middle + 1

    if best_fit is None:
        raise ValueError(
            f"目前訊息與系統提示已超過 {MAX_TOKENS_LIMIT:,} tokens，"
            "即使移除所有歷史訊息仍無法送出。請縮短訊息或移除附件後重試。"
        )

    start_index, kept_messages, token_count = best_fit
    return kept_messages, history_messages[:start_index], token_count


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
# Streamlit 介面與主程式邏輯
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
if "total_input_tokens" not in st.session_state:
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
    col1.metric("總 Input Tokens", f"{st.session_state.total_input_tokens:,}")
    col2.metric("總 Output Tokens", f"{st.session_state.total_output_tokens:,}")
    
    # 計算預估成本 (USD)
    estimated_cost = (st.session_state.total_input_tokens / 1000000 * 0.075) + \
                     (st.session_state.total_output_tokens / 1000000 * 0.3)
    
    st.metric("預估花費 (USD)", f"${estimated_cost:.5f}")
    
    # 加上一個重置按鈕
    if st.button("🔄 重置計數器", use_container_width=True):
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.rerun()

# === 顯示歷史訊息 ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # 如果這筆訊息帶有圖表資料，就把它還原並畫出來！
        if msg.get("chart_json"):
            render_saved_chart(msg["chart_json"])

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
                # 2. 固定輪數與 Token 預算截斷，超出部分摘要至全域記憶
                # ==========================================
                history_messages = st.session_state.messages[:-1].copy()
                overflow_buffer = []

                if len(history_messages) > MAX_HISTORY_MESSAGES:
                    overflow_buffer = history_messages[:-MAX_HISTORY_MESSAGES]
                    history_messages = history_messages[-MAX_HISTORY_MESSAGES:]

                if history_messages and history_messages[0]["role"] == "assistant":
                    overflow_buffer.append(history_messages.pop(0))

                global_memory_context = load_global_memory()
                summarized_history = False
                for _ in range(MAX_HISTORY_MESSAGES + 2):
                    if overflow_buffer:
                        new_global_mem = summarize_overflow(overflow_buffer, global_memory_context)
                        update_global_memory(new_global_mem)
                        global_memory_context = load_global_memory()
                        overflow_buffer = []
                        summarized_history = True

                    dynamic_system_prompt = build_system_instruction(global_memory_context)
                    history_messages, overflow_buffer, token_count = fit_history_to_token_limit(
                        history_messages,
                        message_contents,
                        dynamic_system_prompt,
                    )
                    if not overflow_buffer:
                        break
                else:
                    raise RuntimeError("無法在 Token 上限內整理對話歷史，請稍後重試。")

                if summarized_history:
                    st.toast("🧠 舊對話已摘要並保存至全域記憶", icon="💡")

                gemini_history = build_gemini_history(history_messages)
                st.caption(
                    f"📊 本次請求預估消耗 Token: {token_count:,} / {MAX_TOKENS_LIMIT:,}"
                )
                
                chat_session = ai_client.chats.create(
                    model=AI_MODEL,
                    history=gemini_history,
                    config=types.GenerateContentConfig(
                        system_instruction=dynamic_system_prompt,
                        tools=AI_TOOL_FUNCTIONS,
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

render_chart_queue(st.session_state.current_session_id, locals().get("final_text"))
