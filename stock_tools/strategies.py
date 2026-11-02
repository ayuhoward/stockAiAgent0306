import math

import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
import yfinance as yf

from stock_tools.common import normalize_symbol


def scan_market_opportunities(
    market: str = "TW",
    custom_symbols: list = None,
    strategy: str = "5m_kd",
) -> dict:
    """
    掃描選股池，尋找潛力標的。
    - custom_symbols: 允許使用者指定股票代碼清單 (如 ["2330", "2303"])。未提供則使用預設名單。
    - strategy: 支援 "5m_kd" (五分K+KD交叉短線策略) 與 "daily_rsi" (日線+RSI波段策略)。
    """
    if custom_symbols and isinstance(custom_symbols, list) and len(custom_symbols) > 0:
        watch_list = [normalize_symbol(s) for s in custom_symbols]
    else:
        watch_list = (
            ["2303.TW", "2481.TW", "5328.TW", "3264.TW", "3533.TW"]
            if market.upper() == "TW"
            else ["AAPL", "NVDA", "TSLA"]
        )

    opportunities = []

    for symbol in watch_list:
        try:
            if strategy == "5m_kd":
                df = yf.download(symbol, period="5d", interval="5m", progress=False)
                if df.empty or len(df) < 30:
                    continue

                close, high, low = df["Close"].squeeze(), df["High"].squeeze(), df["Low"].squeeze()
                sma5 = ta.sma(close, length=5).iloc[-1]
                sma20 = ta.sma(close, length=20).iloc[-1]
                kd = ta.stoch(high, low, close, k=9, d=3, smooth_k=3)

                if kd is None or kd.empty:
                    continue
                k_val, d_val = kd["STOCHk_9_3_3"].iloc[-1], kd["STOCHd_9_3_3"].iloc[-1]
                curr_price = float(close.iloc[-1])

                if curr_price > sma20 and sma5 > sma20 and k_val > d_val and k_val < 80:
                    opportunities.append(
                        {
                            "symbol": symbol,
                            "current_price": round(curr_price, 2),
                            "sma5": round(float(sma5), 2),
                            "sma20": round(float(sma20), 2),
                            "k_value": round(float(k_val), 2),
                            "d_value": round(float(d_val), 2),
                        }
                    )

            elif strategy == "daily_rsi":
                df = yf.download(symbol, period="6mo", interval="1d", progress=False)
                if df.empty or len(df) < 30:
                    continue

                close = df["Close"].squeeze()
                sma5 = ta.sma(close, length=5).iloc[-1]
                sma20 = ta.sma(close, length=20).iloc[-1]
                rsi14 = ta.rsi(close, length=14).iloc[-1]
                curr_price = float(close.iloc[-1])

                if curr_price > sma20 and sma5 > sma20 and 48 <= rsi14 <= 75:
                    opportunities.append(
                        {
                            "symbol": symbol,
                            "current_price": round(curr_price, 2),
                            "sma5": round(float(sma5), 2),
                            "sma20": round(float(sma20), 2),
                            "rsi14": round(float(rsi14), 2),
                        }
                    )
        except Exception:
            continue

    return {
        "market": market,
        "strategy": strategy,
        "scanned_count": len(watch_list),
        "matched_count": len(opportunities),
        "candidates": opportunities,
    }


def run_backtest(
    symbol: str,
    strategy: str = "sma_cross",
    trailing_stop_pct: float = 0.08,
    interval: str = "1d",
    period: str = "1y",
) -> dict:
    """
    執行量化回測。
    - period 支援: "1mo", "3mo", "6mo", "1y", "2y", "max" 等。
    注意：若 interval 為分K (如 5m)，period 強制最高只能設為 "60d"。
    """
    clean_symbol = normalize_symbol(symbol)

    if "m" in interval and period in ["1y", "2y", "max", "ytd", "5y", "10y"]:
        period = "60d"
    try:
        df = yf.download(clean_symbol, period=period, interval=interval, progress=False)
        if df.empty or "Close" not in df:
            return {"error": f"找不到 {clean_symbol} 的歷史價格資料。"}

        price = df["Close"].squeeze()

        if strategy == "rsi_signal":
            rsi = ta.rsi(price, length=14)
            entries = rsi < 30
            exits = rsi > 70
        else:
            fast_ma = ta.sma(price, length=5)
            slow_ma = ta.sma(price, length=20)
            entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        portfolio = vbt.Portfolio.from_signals(
            price,
            entries,
            exits,
            init_cash=100000,
            fees=0.001425,
            sl_stop=trailing_stop_pct,
            sl_trail=True,
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
            "total_return_pct": f"{total_return}%",
        }
    except Exception as e:
        return {"error": f"執行量化回測時發生錯誤: {str(e)}"}
