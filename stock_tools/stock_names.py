import re

import requests
import streamlit as st

_TWSE_COMPANIES_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
_TPEX_COMPANIES_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_company_names(
    url: str, code_key: str, short_name_key: str, name_key: str
) -> dict[str, str]:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    records = response.json()
    if not isinstance(records, list):
        raise ValueError(f"公司清單格式無法辨識：{url}")

    names = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        code = str(record.get(code_key, "")).strip()
        name = str(record.get(short_name_key) or record.get(name_key) or "").strip()
        if re.fullmatch(r"\d{4,6}", code) and name:
            names[code] = name
    return names


def get_stock_names(symbols: list[str]) -> tuple[dict[str, str], list[str]]:
    taiwan_codes = {
        re.sub(r"\.(?:TW|TWO)$", "", symbol.upper())
        for symbol in symbols
        if re.fullmatch(r"\d{4,6}\.(?:TW|TWO)", symbol.upper())
    }
    if not taiwan_codes:
        return {}, []

    names = {}
    errors = []
    sources = (
        (
            _TWSE_COMPANIES_URL,
            "公司代號",
            "公司簡稱",
            "公司名稱",
        ),
        (
            _TPEX_COMPANIES_URL,
            "SecuritiesCompanyCode",
            "CompanyAbbreviation",
            "CompanyName",
        ),
    )
    for url, code_key, short_name_key, name_key in sources:
        try:
            names.update(
                _fetch_company_names(url, code_key, short_name_key, name_key)
            )
        except (requests.RequestException, ValueError) as error:
            errors.append(f"{url}：{error}")

    return {code: names[code] for code in taiwan_codes if code in names}, errors
