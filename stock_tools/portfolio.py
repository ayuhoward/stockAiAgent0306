import datetime

import yfinance as yf

from stock_tools.common import get_db_connection, normalize_symbol


def record_trade(symbol: str, action: str, shares: float, price: float, notes: str = "") -> dict:
    # Require an explicit trade price to avoid recording a guessed price.
    clean_symbol = normalize_symbol(symbol)
    action = action.upper()
    if action not in ["BUY", "SELL"]:
        return {"error": "交易動作必須為 BUY 或 SELL"}
    if price is None or price <= 0:
        return {"error": "請告知使用者必須提供具體的成交價格。"}

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO trade_logs (symbol, action, shares, price, trade_date, notes) VALUES (%s, %s, %s, %s, %s, %s)",
            (clean_symbol, action, shares, price, now_str, notes),
        )
        cursor.execute("SELECT shares, avg_cost FROM positions WHERE symbol = %s", (clean_symbol,))
        pos = cursor.fetchone()

        if action == "BUY":
            if pos:
                new_shares = pos[0] + shares
                new_cost = ((pos[0] * pos[1]) + (shares * price)) / new_shares
                cursor.execute(
                    "UPDATE positions SET shares = %s, avg_cost = %s, updated_at = %s WHERE symbol = %s",
                    (new_shares, round(new_cost, 2), now_str, clean_symbol),
                )
            else:
                cursor.execute(
                    "INSERT INTO positions (symbol, shares, avg_cost, updated_at) VALUES (%s, %s, %s, %s)",
                    (clean_symbol, shares, price, now_str),
                )
        elif action == "SELL":
            if not pos:
                return {"error": "無持股紀錄"}
            if shares >= pos[0]:
                cursor.execute("DELETE FROM positions WHERE symbol = %s", (clean_symbol,))
            else:
                cursor.execute(
                    "UPDATE positions SET shares = %s, updated_at = %s WHERE symbol = %s",
                    (pos[0] - shares, now_str, clean_symbol),
                )
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
    if not rows:
        return {"portfolio": [], "message": "目前無持股"}

    portfolio = []
    total_cost = total_market_val = 0.0
    for symbol, shares, avg_cost in rows:
        cost_val = shares * avg_cost
        total_cost += cost_val
        curr_price = avg_cost
        try:
            hist = yf.Ticker(symbol).history(period="1d")
            if not hist.empty:
                curr_price = round(float(hist["Close"].iloc[-1]), 2)
        except Exception:
            pass
        mkt_val = shares * curr_price
        total_market_val += mkt_val
        portfolio.append(
            {
                "symbol": symbol,
                "shares": shares,
                "avg_cost": avg_cost,
                "current_price": curr_price,
                "unrealized_pnl": round(mkt_val - cost_val, 2),
            }
        )
    return {
        "summary": {
            "total_cost": round(total_cost, 2),
            "total_unrealized_pnl": round(total_market_val - total_cost, 2),
        },
        "portfolio": portfolio,
    }
