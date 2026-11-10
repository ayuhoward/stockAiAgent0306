import unittest
from unittest.mock import patch

import requests

from stock_tools.portfolio import get_positions
from stock_tools.stock_names import _fetch_company_names, get_stock_names


class GetPositionsTests(unittest.TestCase):
    @patch("stock_tools.portfolio.get_db_connection")
    def test_returns_database_positions_and_closes_connection(self, connect):
        cursor = connect.return_value.cursor.return_value
        cursor.fetchall.return_value = [
            ("2330.TW", 10, 800.5),
            ("AAPL", 2, 175.25),
        ]

        positions = get_positions()

        self.assertEqual(
            positions,
            [("2330.TW", 10.0, 800.5), ("AAPL", 2.0, 175.25)],
        )
        cursor.execute.assert_called_once_with(
            "SELECT symbol, shares, avg_cost FROM positions ORDER BY symbol"
        )
        cursor.close.assert_called_once()
        connect.return_value.close.assert_called_once()


class GetStockNamesTests(unittest.TestCase):
    @patch("stock_tools.stock_names._fetch_company_names")
    def test_returns_chinese_names_for_twse_and_tpex_symbols(self, fetch_names):
        fetch_names.side_effect = [
            {"2330": "台積電"},
            {"6488": "環球晶圓"},
        ]

        names, errors = get_stock_names(["2330.TW", "6488.TWO", "AAPL"])

        self.assertEqual(names, {"2330": "台積電", "6488": "環球晶圓"})
        self.assertEqual(errors, [])
        self.assertEqual(fetch_names.call_count, 2)

    @patch("stock_tools.stock_names._fetch_company_names")
    def test_reports_lookup_errors_without_failing_the_portfolio(self, fetch_names):
        fetch_names.side_effect = [
            requests.RequestException("network unavailable"),
            {"6488": "環球晶圓"},
        ]

        names, errors = get_stock_names(["2330.TW", "6488.TWO"])

        self.assertEqual(names, {"6488": "環球晶圓"})
        self.assertEqual(len(errors), 1)

    @patch("stock_tools.stock_names._fetch_company_names")
    def test_does_not_query_taiwan_company_lists_for_non_taiwan_symbols(
        self, fetch_names
    ):
        names, errors = get_stock_names(["AAPL", "MSFT"])

        self.assertEqual(names, {})
        self.assertEqual(errors, [])
        fetch_names.assert_not_called()

    @patch("stock_tools.stock_names.requests.get")
    def test_prefers_company_abbreviation(self, get):
        response = get.return_value
        response.json.return_value = [
            {"公司代號": "2330", "公司簡稱": "台積電", "公司名稱": "台灣積體電路"},
            {"公司代號": "2330A", "公司簡稱": "台積特"},
            {"公司代號": "0050", "公司名稱": "元大台灣50"},
        ]

        names = _fetch_company_names(
            "https://example.test/companies",
            "公司代號",
            "公司簡稱",
            "公司名稱",
        )

        self.assertEqual(names, {"2330": "台積電", "0050": "元大台灣50"})
        get.assert_called_once_with("https://example.test/companies", timeout=10)


if __name__ == "__main__":
    unittest.main()
