import json

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf

from stock_tools.chat_history import save_chat_message
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


def render_saved_chart(chart_json: str):
    """Render charts stored with a conversation, including legacy single-chart records."""
    try:
        try:
            parsed_json = json.loads(chart_json)
        except Exception:
            parsed_json = chart_json

        if isinstance(parsed_json, list):
            for chart in parsed_json:
                fig = pio.from_json(chart)
                st.plotly_chart(fig, use_container_width=True)
        else:
            fig = pio.from_json(chart_json)
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.error("圖表還原失敗")


def render_chart_queue(session_id: str, message_content: str = None):
    if "render_chart_queue" not in st.session_state or not st.session_state.render_chart_queue:
        return

    saved_chart_jsons = []
    st.markdown("---")

    for chart_req in st.session_state.render_chart_queue:
        chart_symbol = chart_req["symbol"]
        chart_period = chart_req["period"]

        st.subheader(f"📊 {chart_symbol} K 線圖 (期間: {chart_period})")

        try:
            df = yf.download(chart_symbol, period=chart_period, progress=False)

            if df.empty and chart_symbol.endswith(".TW"):
                chart_symbol = chart_symbol.replace(".TW", ".TWO")
                df = yf.download(chart_symbol, period=chart_period, progress=False)

            if not df.empty:
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=df.index,
                            open=df["Open"].squeeze(),
                            high=df["High"].squeeze(),
                            low=df["Low"].squeeze(),
                            close=df["Close"].squeeze(),
                            name=chart_symbol,
                        )
                    ]
                )

                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)
                saved_chart_jsons.append(fig.to_json())
            else:
                st.warning(f"無法獲取 {chart_symbol} 在 {chart_period} 期間的歷史資料。")

        except Exception as e:
            st.error(f"繪製 {chart_symbol} 時發生錯誤：{str(e)}")

    if saved_chart_jsons:
        final_json_str = json.dumps(saved_chart_jsons)
        content = message_content if message_content is not None else "為您繪製圖表如下："
        st.session_state.messages.append(
            {"role": "assistant", "content": content, "chart_json": final_json_str}
        )
        save_chat_message(session_id, "assistant", content, final_json_str)

    st.session_state.render_chart_queue = []
