import concurrent.futures
import datetime
import re
import time

import requests
import streamlit as st

from stock_tools.market_data import fetch_fubon_holdings

_TWSE_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
_TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade"
_TPEX_REFERER = (
    "https://www.tpex.org.tw/zh-tw/mainboard/trading/"
    "major-institutional/detail/day.html"
)
_TPEX_TRUST_NET_BUY_INDEX = 13
_MAX_DATE_RANGE_DAYS = 366
_MAX_FUBON_CANDIDATE_CHECKS = 500
_DAILY_REPORT_MAX_ATTEMPTS = 3
_DAILY_REPORT_RETRY_DELAY_SECONDS = 0.5


def _parse_share_count(value) -> int:
    normalized = str(value).strip().replace(",", "")
    if normalized in {"", "-", "--", "—"}:
        return 0
    return int(normalized)


def _parse_twse_daily_report(payload: dict) -> dict:
    if payload.get("stat") != "OK":
        message = str(payload.get("stat", "未知錯誤"))
        if "沒有符合條件的資料" in message or "查無資料" in message:
            return {}
        raise ValueError(f"TWSE T86 回傳錯誤：{message}")

    fields = payload.get("fields", [])
    try:
        symbol_index = fields.index("證券代號")
        name_index = fields.index("證券名稱")
        trust_net_index = fields.index("投信買賣超股數")
    except ValueError as error:
        raise ValueError(f"TWSE T86 欄位格式無法辨識：{fields}") from error

    records = {}
    for row in payload.get("data", []):
        if len(row) <= max(symbol_index, name_index, trust_net_index):
            raise ValueError("TWSE T86 回傳了不完整的個股資料列。")
        symbol = str(row[symbol_index]).strip()
        if not re.fullmatch(r"\d{4,6}", symbol):
            continue
        records[symbol] = {
            "name": str(row[name_index]).strip(),
            "net_buy_shares": _parse_share_count(row[trust_net_index]),
        }
    return records


def _parse_tpex_daily_report(payload: dict, requested_date: datetime.date) -> dict:
    if payload.get("stat") != "ok":
        message = str(payload.get("stat", "未知錯誤"))
        if "沒有符合條件的資料" in message or "查無資料" in message:
            return {}
        raise ValueError(f"TPEx 三大法人日報回傳錯誤：{message}")

    expected_date = requested_date.strftime("%Y%m%d")
    if payload.get("date") != expected_date:
        raise ValueError(
            "TPEx 回傳日期與請求日期不符："
            f"requested={expected_date}, received={payload.get('date', '未知')}"
        )

    tables = payload.get("tables", [])
    if not tables or not isinstance(tables[0].get("data"), list):
        raise ValueError("TPEx 三大法人日報缺少個股資料表。")

    records = {}
    for row in tables[0]["data"]:
        if len(row) <= _TPEX_TRUST_NET_BUY_INDEX:
            raise ValueError("TPEx 三大法人日報回傳了不完整的個股資料列。")
        symbol = str(row[0]).strip()
        if not re.fullmatch(r"\d{4,6}", symbol):
            continue
        records[symbol] = {
            "name": str(row[1]).strip(),
            "net_buy_shares": _parse_share_count(
                row[_TPEX_TRUST_NET_BUY_INDEX]
            ),
        }
    return records


def _retry_daily_report(fetch_report):
    for attempt in range(1, _DAILY_REPORT_MAX_ATTEMPTS + 1):
        try:
            return fetch_report()
        except (requests.RequestException, ValueError):
            if attempt == _DAILY_REPORT_MAX_ATTEMPTS:
                raise
            time.sleep(_DAILY_REPORT_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)))


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_twse_daily_report(date_text: str) -> dict:
    def fetch_report():
        response = requests.get(
            _TWSE_URL,
            params={
                "date": date_text.replace("-", ""),
                "selectType": "ALLBUT0999",
                "response": "json",
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        response.raise_for_status()
        return _parse_twse_daily_report(response.json())

    return _retry_daily_report(fetch_report)


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_tpex_daily_report(date_text: str) -> dict:
    report_date = datetime.date.fromisoformat(date_text)

    def fetch_report():
        response = requests.get(
            _TPEX_URL,
            params={
                "type": "Daily",
                "sect": "AL",
                "date": report_date.strftime("%Y/%m/%d"),
                "response": "json",
            },
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": _TPEX_REFERER,
            },
            timeout=20,
        )
        response.raise_for_status()
        return _parse_tpex_daily_report(response.json(), report_date)

    return _retry_daily_report(fetch_report)


def _trading_weekdays(start_date: datetime.date, end_date: datetime.date):
    day_count = (end_date - start_date).days
    return [
        start_date + datetime.timedelta(days=offset)
        for offset in range(day_count + 1)
        if (start_date + datetime.timedelta(days=offset)).weekday() < 5
    ]


def _potential_runup_lots(daily_net_shares: dict, trading_dates: list[str]) -> float:
    cumulative_lots = 0
    lowest_prior_lots = 0
    largest_runup_lots = 0
    for date_text in trading_dates:
        cumulative_lots += daily_net_shares.get(date_text, 0) / 1000
        largest_runup_lots = max(
            largest_runup_lots,
            cumulative_lots - lowest_prior_lots,
        )
        lowest_prior_lots = min(lowest_prior_lots, cumulative_lots)
    return largest_runup_lots


def _snapshot_date(snapshot_date: str, end_date: datetime.date) -> datetime.date:
    month_text, day_text = snapshot_date.split("/")
    snapshot = datetime.date(end_date.year, int(month_text), int(day_text))
    if snapshot > end_date:
        snapshot = snapshot.replace(year=snapshot.year - 1)
    return snapshot


def _reconstruct_holdings(
    current_lots: int,
    daily_net_shares: dict,
    trading_dates: list[str],
    start_date: datetime.date,
    snapshot_date: datetime.date,
) -> dict:
    applicable_dates = [
        datetime.date.fromisoformat(date_text)
        for date_text in trading_dates
        if start_date <= datetime.date.fromisoformat(date_text) <= snapshot_date
    ]
    total_net_lots = sum(
        daily_net_shares.get(day.isoformat(), 0) for day in applicable_dates
    ) / 1000

    estimates = [
        (start_date.isoformat(), current_lots - total_net_lots)
    ]
    cumulative_net_lots = 0
    for day in applicable_dates:
        cumulative_net_lots += daily_net_shares.get(day.isoformat(), 0) / 1000
        estimate = current_lots - total_net_lots + cumulative_net_lots
        estimates.append((day.isoformat(), estimate))

    min_date, min_lots = min(estimates, key=lambda item: (item[1], item[0]))
    return {
        "minimum_date": min_date,
        "minimum_lots": min_lots,
        "current_lots": current_lots,
        "estimated_accumulation_lots": current_lots - min_lots,
        "ytd_net_buy_lots": total_net_lots,
    }


def scan_institutional_accumulation(
    start_date: str = "",
    end_date: str = "",
    near_zero_lots: int = 1000,
    min_accumulation_lots: int = 5000,
    min_current_holding_lots: int = 5000,
    max_candidate_checks: int = 200,
    max_results: int = 20,
) -> dict:
    """Scan TWSE/TPEx flows and estimate stocks accumulated from near-zero holdings.

    Holdings before the latest Fubon snapshot are reconstructed by reversing
    official daily investment-trust net-buy data; historical holdings are not
    directly reported by this method.
    """
    today = datetime.date.today()
    if not start_date:
        start_date = f"{today.year}-01-01"
    if not end_date:
        end_date = today.isoformat()

    try:
        start = datetime.date.fromisoformat(start_date)
        end = datetime.date.fromisoformat(end_date)
    except ValueError:
        return {"error": "日期格式須為 YYYY-MM-DD。"}

    if end < start:
        return {"error": "結束日期不可早於起始日期。"}
    if end > today:
        return {"error": "結束日期不可晚於今天。"}
    if (end - start).days > _MAX_DATE_RANGE_DAYS:
        return {"error": "單次最多掃描 366 天，請縮短日期區間。"}
    if near_zero_lots < 0 or min_accumulation_lots <= 0 or min_current_holding_lots < 0:
        return {"error": "持股門檻必須為非負值，累積買進門檻必須大於 0。"}
    if not 1 <= max_candidate_checks <= _MAX_FUBON_CANDIDATE_CHECKS:
        return {
            "error": (
                f"max_candidate_checks 必須介於 1 至 "
                f"{_MAX_FUBON_CANDIDATE_CHECKS}。"
            )
        }
    if not 1 <= max_results <= 100:
        return {"error": "max_results 必須介於 1 至 100。"}

    weekdays = _trading_weekdays(start, end)
    requests_to_make = [
        (market, day)
        for day in weekdays
        for market in ("twse", "tpex")
    ]
    reports = {}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_request = {
            executor.submit(
                _fetch_twse_daily_report
                if market == "twse"
                else _fetch_tpex_daily_report,
                day.isoformat(),
            ): (market, day)
            for market, day in requests_to_make
        }
        for future in concurrent.futures.as_completed(future_to_request):
            market, day = future_to_request[future]
            try:
                reports[(market, day.isoformat())] = future.result()
            except (requests.RequestException, ValueError) as error:
                failures.append(
                    {
                        "market": market.upper(),
                        "date": day.isoformat(),
                        "error": str(error),
                    }
                )

    daily_net_shares_by_symbol = {}
    stock_names = {}
    trading_dates = set()
    for (market, date_text), records in sorted(reports.items()):
        if not records:
            continue
        trading_dates.add(date_text)
        for symbol, record in records.items():
            stock_names[symbol] = record["name"]
            daily_net_shares_by_symbol.setdefault(symbol, {})[date_text] = (
                record["net_buy_shares"]
            )

    ordered_trading_dates = sorted(trading_dates)
    if not ordered_trading_dates:
        result = {"error": "指定日期區間內查無上市或上櫃交易資料。"}
        if failures:
            result["failed_daily_report_count"] = len(failures)
            result["daily_report_failures"] = failures[:50]
        return result

    potential_candidates = []
    for symbol, daily_flows in daily_net_shares_by_symbol.items():
        max_runup = _potential_runup_lots(daily_flows, ordered_trading_dates)
        if max_runup >= min_accumulation_lots:
            potential_candidates.append((symbol, max_runup))
    potential_candidates.sort(key=lambda item: item[1], reverse=True)

    candidates_to_check = potential_candidates[:max_candidate_checks]
    snapshot_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_symbol = {
            executor.submit(fetch_fubon_holdings, symbol): symbol
            for symbol, _ in candidates_to_check
        }
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                snapshot_results[symbol] = future.result()
            except Exception as error:
                snapshot_results[symbol] = {"error": str(error)}

    matches = []
    snapshot_failures = []
    for symbol, max_runup in candidates_to_check:
        snapshot = snapshot_results[symbol]
        if not isinstance(snapshot, dict):
            snapshot_failures.append(
                {"symbol": symbol, "error": "富邦頁面回傳資料格式無效。"}
            )
            continue
        if "error" in snapshot:
            snapshot_failures.append(
                {"symbol": symbol, "error": snapshot["error"]}
            )
            continue

        try:
            snapshot_date = _snapshot_date(snapshot["data_date"], end)
            investment_trust = snapshot["investment_trust"]
            if not isinstance(investment_trust, dict):
                raise ValueError("投信持股欄位格式無法辨識。")
            current_lots = int(investment_trust["lots"])
            ownership_pct = float(investment_trust["ownership_pct"])
            source_url = snapshot["source_url"]
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            snapshot_failures.append(
                {
                    "symbol": symbol,
                    "error": f"富邦頁面快照格式無法辨識：{error}",
                }
            )
            continue

        if snapshot_date < start or snapshot_date > end:
            snapshot_failures.append(
                {
                    "symbol": symbol,
                    "error": f"富邦資料日期 {snapshot['data_date']} 不在掃描區間內。",
                }
            )
            continue

        estimates = _reconstruct_holdings(
            current_lots=current_lots,
            daily_net_shares=daily_net_shares_by_symbol.get(symbol, {}),
            trading_dates=ordered_trading_dates,
            start_date=start,
            snapshot_date=snapshot_date,
        )
        minimum_lots = estimates["minimum_lots"]
        increase_lots = estimates["estimated_accumulation_lots"]
        if (
            0 <= minimum_lots <= near_zero_lots
            and current_lots >= min_current_holding_lots
            and increase_lots >= min_accumulation_lots
        ):
            matches.append(
                {
                    "symbol": symbol,
                    "name": stock_names.get(symbol, ""),
                    "near_zero_date": estimates["minimum_date"],
                    "estimated_near_zero_holding_lots": round(minimum_lots, 1),
                    "current_holding_lots": current_lots,
                    "current_ownership_pct": ownership_pct,
                    "estimated_accumulation_lots": round(increase_lots, 1),
                    "ytd_net_buy_lots": round(estimates["ytd_net_buy_lots"], 1),
                    "snapshot_date": snapshot["data_date"],
                    "source_url": source_url,
                }
            )

    matches.sort(
        key=lambda item: item["estimated_accumulation_lots"],
        reverse=True,
    )
    truncated_candidate_count = len(potential_candidates) - len(candidates_to_check)
    truncated_result_count = max(0, len(matches) - max_results)
    return {
        "data_type": "estimated_institutional_accumulation_screen",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "near_zero_lots_threshold": near_zero_lots,
        "minimum_accumulation_lots": min_accumulation_lots,
        "minimum_current_holding_lots": min_current_holding_lots,
        "trading_days_with_data": len(ordered_trading_dates),
        "stocks_in_daily_flow_data": len(daily_net_shares_by_symbol),
        "potential_candidates": len(potential_candidates),
        "fubon_snapshots_checked": len(candidates_to_check),
        "candidate_checks_truncated": truncated_candidate_count,
        "results_truncated": truncated_result_count,
        "failed_daily_report_count": len(failures),
        "daily_report_failures": failures[:50],
        "daily_report_failure_details_truncated": len(failures) > 50,
        "snapshot_failure_count": len(snapshot_failures),
        "snapshot_failures": snapshot_failures[:20],
        "scan_complete": (
            truncated_candidate_count == 0
            and truncated_result_count == 0
            and not failures
            and not snapshot_failures
        ),
        "matched_count": len(matches),
        "candidates": matches[:max_results],
        "sources": [
            "TWSE T86 三大法人買賣超日報",
            "TPEx 三大法人每日交易資訊",
            "Fubon eBroker 個股籌碼分佈頁面",
        ],
        "method": (
            "以最新富邦投信持股快照，逐日扣除該日之後至快照日的投信淨買賣超，"
            "反推掃描區間內的估算持股；再篩選曾估算低於趨近零門檻且其後累積買進達標者。"
        ),
        "caveats": [
            "期間內歷史持股是依每日淨買賣超反推的估算值，不是富邦歷史持股快照。",
            "估算未涵蓋可能影響庫存的非交易因素；負持股估算不列為符合條件。",
            "掃描僅核對累積買超潛力最高的指定數量股票富邦頁面；若 candidate_checks_truncated 大於 0，結果不是完整名單。",
            "富邦資料更新頻率未經確認，不保證為盤中即時。",
        ] + (
            [
                "部分交易日或個股快照未能取得；結果只列出成功核對的候選，可能漏列或估算失準。請查看 daily_report_failures 與 snapshot_failures。"
            ]
            if failures or snapshot_failures
            else []
        ),
    }
