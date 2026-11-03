import datetime
import requests
import unittest
from unittest.mock import Mock, patch

from stock_tools.institutional_screening import (
    _parse_tpex_daily_report,
    _parse_twse_daily_report,
    _potential_runup_lots,
    _reconstruct_holdings,
    _retry_daily_report,
    scan_institutional_accumulation,
)


class InstitutionalScreeningTests(unittest.TestCase):
    def test_parses_twse_investment_trust_net_shares(self):
        payload = {
            "stat": "OK",
            "fields": ["證券代號", "證券名稱", "投信買賣超股數"],
            "data": [["6213", "聯茂", "7,000,000"]],
        }

        result = _parse_twse_daily_report(payload)

        self.assertEqual(
            result["6213"],
            {"name": "聯茂", "net_buy_shares": 7_000_000},
        )

    def test_parses_tpex_trust_column_and_rejects_stale_response_date(self):
        requested_date = datetime.date(2026, 1, 2)
        row = ["6213", "聯茂"] + ["0"] * 22
        row[13] = "7,000,000"
        payload = {
            "stat": "ok",
            "date": "20260102",
            "tables": [{"data": [row]}],
        }

        result = _parse_tpex_daily_report(payload, requested_date)

        self.assertEqual(result["6213"]["net_buy_shares"], 7_000_000)
        payload["date"] = "20261102"
        with self.assertRaisesRegex(ValueError, "回傳日期與請求日期不符"):
            _parse_tpex_daily_report(payload, requested_date)

    def test_finds_runup_and_reconstructs_near_zero_start(self):
        trading_dates = ["2026-01-02", "2026-01-05"]
        daily_net_shares = {
            "2026-01-02": 7_000_000,
            "2026-01-05": 1_000_000,
        }

        self.assertEqual(
            _potential_runup_lots(daily_net_shares, trading_dates),
            8000,
        )
        result = _reconstruct_holdings(
            current_lots=9000,
            daily_net_shares=daily_net_shares,
            trading_dates=trading_dates,
            start_date=datetime.date(2026, 1, 1),
            snapshot_date=datetime.date(2026, 1, 5),
        )
        self.assertEqual(result["minimum_date"], "2026-01-01")
        self.assertEqual(result["minimum_lots"], 1000)
        self.assertEqual(result["estimated_accumulation_lots"], 8000)

    def test_retries_a_transient_daily_report_failure(self):
        fetch_report = Mock(
            side_effect=[
                requests.ConnectionError("connection reset"),
                {"6213": {"name": "聯茂", "net_buy_shares": 7_000_000}},
            ]
        )

        with patch("stock_tools.institutional_screening.time.sleep") as mock_sleep:
            result = _retry_daily_report(fetch_report)

        self.assertIn("6213", result)
        self.assertEqual(fetch_report.call_count, 2)
        mock_sleep.assert_called_once_with(0.5)

    @patch("stock_tools.institutional_screening.fetch_fubon_holdings")
    @patch("stock_tools.institutional_screening._fetch_tpex_daily_report")
    @patch("stock_tools.institutional_screening._fetch_twse_daily_report")
    def test_scan_matches_stocks_bought_from_near_zero(
        self,
        mock_twse,
        mock_tpex,
        mock_fubon,
    ):
        mock_twse.side_effect = lambda day: (
            {
                "6213": {
                    "name": "聯茂",
                    "net_buy_shares": 7_000_000,
                }
            }
            if day == "2026-01-02"
            else {}
        )
        mock_tpex.return_value = {}
        mock_fubon.return_value = {
            "symbol": "6213",
            "source": "Fubon eBroker",
            "data_date": "01/02",
            "unit": "張",
            "investment_trust": {
                "lots": 8000,
                "daily_change_lots": 100,
                "ownership_pct": 4.2,
            },
            "source_url": "https://example.test/6213",
        }

        result = scan_institutional_accumulation(
            start_date="2026-01-01",
            end_date="2026-01-02",
            near_zero_lots=1000,
            min_accumulation_lots=5000,
            min_current_holding_lots=5000,
        )

        mock_fubon.assert_called_once_with("6213")
        self.assertTrue(result["scan_complete"])
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["candidates"][0]["symbol"], "6213")
        self.assertEqual(
            result["candidates"][0]["estimated_near_zero_holding_lots"],
            1000,
        )

    @patch("stock_tools.institutional_screening.fetch_fubon_holdings")
    @patch("stock_tools.institutional_screening._fetch_tpex_daily_report")
    @patch("stock_tools.institutional_screening._fetch_twse_daily_report")
    def test_scan_marks_truncated_results_incomplete(
        self,
        mock_twse,
        mock_tpex,
        mock_fubon,
    ):
        mock_twse.side_effect = lambda day: (
            {
                symbol: {"name": symbol, "net_buy_shares": 7_000_000}
                for symbol in ("6213", "6488")
            }
            if day == "2026-01-02"
            else {}
        )
        mock_tpex.return_value = {}
        mock_fubon.side_effect = lambda symbol: {
            "symbol": symbol,
            "source": "Fubon eBroker",
            "data_date": "01/02",
            "unit": "張",
            "investment_trust": {
                "lots": 8000,
                "daily_change_lots": 100,
                "ownership_pct": 4.2,
            },
            "source_url": f"https://example.test/{symbol}",
        }

        result = scan_institutional_accumulation(
            start_date="2026-01-01",
            end_date="2026-01-02",
            max_candidate_checks=2,
            max_results=1,
        )

        self.assertEqual(result["matched_count"], 2)
        self.assertEqual(result["results_truncated"], 1)
        self.assertFalse(result["scan_complete"])
        self.assertEqual(len(result["candidates"]), 1)

    @patch("stock_tools.institutional_screening.fetch_fubon_holdings")
    @patch("stock_tools.institutional_screening._fetch_tpex_daily_report")
    @patch("stock_tools.institutional_screening._fetch_twse_daily_report")
    def test_scan_returns_partial_results_with_failed_report_details(
        self,
        mock_twse,
        mock_tpex,
        mock_fubon,
    ):
        def fetch_twse(day):
            if day == "2026-01-02":
                raise requests.ConnectionError("connection reset")
            return {}

        mock_twse.side_effect = fetch_twse
        mock_tpex.side_effect = lambda day: (
            {
                "6213": {
                    "name": "聯茂",
                    "net_buy_shares": 7_000_000,
                }
            }
            if day == "2026-01-05"
            else {}
        )
        mock_fubon.return_value = {
            "symbol": "6213",
            "source": "Fubon eBroker",
            "data_date": "01/05",
            "unit": "張",
            "investment_trust": {
                "lots": 8000,
                "daily_change_lots": 100,
                "ownership_pct": 4.2,
            },
            "source_url": "https://example.test/6213",
        }

        result = scan_institutional_accumulation(
            start_date="2026-01-02",
            end_date="2026-01-05",
        )

        self.assertNotIn("error", result)
        self.assertFalse(result["scan_complete"])
        self.assertEqual(result["failed_daily_report_count"], 1)
        self.assertEqual(
            result["daily_report_failures"][0]["date"],
            "2026-01-02",
        )
        self.assertEqual(result["candidates"][0]["symbol"], "6213")
        self.assertTrue(
            any("可能漏列或估算失準" in caveat for caveat in result["caveats"])
        )

    @patch("stock_tools.institutional_screening.fetch_fubon_holdings")
    @patch("stock_tools.institutional_screening._fetch_tpex_daily_report")
    @patch("stock_tools.institutional_screening._fetch_twse_daily_report")
    def test_scan_skips_failed_snapshot_and_keeps_other_candidates(
        self,
        mock_twse,
        mock_tpex,
        mock_fubon,
    ):
        flows = {
            symbol: {"name": symbol, "net_buy_shares": 7_000_000}
            for symbol in ("6213", "6488")
        }
        mock_twse.side_effect = lambda day: flows if day == "2026-01-02" else {}
        mock_tpex.return_value = {}

        def fetch_fubon(symbol):
            if symbol == "6213":
                raise requests.ConnectionError("Fubon connection reset")
            return {
                "symbol": symbol,
                "source": "Fubon eBroker",
                "data_date": "01/02",
                "unit": "張",
                "investment_trust": {
                    "lots": 8000,
                    "daily_change_lots": 100,
                    "ownership_pct": 4.2,
                },
                "source_url": f"https://example.test/{symbol}",
            }

        mock_fubon.side_effect = fetch_fubon

        result = scan_institutional_accumulation(
            start_date="2026-01-01",
            end_date="2026-01-02",
        )

        self.assertEqual(result["matched_count"], 1)
        self.assertEqual([item["symbol"] for item in result["candidates"]], ["6488"])
        self.assertFalse(result["scan_complete"])
        self.assertEqual(result["snapshot_failure_count"], 1)
        self.assertEqual(result["snapshot_failures"][0]["symbol"], "6213")
        self.assertIn("Fubon connection reset", result["snapshot_failures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
