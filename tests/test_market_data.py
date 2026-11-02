import unittest
from unittest.mock import Mock, patch

from stock_tools.market_data import _parse_fubon_chip_page, fetch_chip_data


class FetchChipDataTests(unittest.TestCase):
    @patch(
        "stock_tools.market_data._fetch_fubon_chip_data",
        return_value={
            "available": True,
            "data_date": "11/02",
            "positions": {
                "investment_trust": {
                    "lots": 15335,
                    "daily_change_lots": 32,
                    "ownership_pct": 4.22,
                }
            },
        },
    )
    @patch("stock_tools.market_data.requests.get")
    def test_returns_daily_flow_and_fubon_holding_snapshot(
        self, mock_get, mock_fubon
    ):
        response = Mock()
        response.json.return_value = {
            "status": 200,
            "data": [
                {
                    "date": "2026-03-05",
                    "name": "Investment_Trust",
                    "buy": 2500,
                    "sell": 500,
                },
                {
                    "date": "2026-03-05",
                    "name": "Dealer_Self",
                    "buy": 500,
                    "sell": 1500,
                },
                {
                    "date": "2026-03-05",
                    "name": "Dealer_Hedging",
                    "buy": 2000,
                    "sell": 500,
                },
                {
                    "date": "2026-03-04",
                    "name": "Investment_Trust",
                    "buy": 9000,
                    "sell": 0,
                },
            ],
        }
        mock_get.return_value = response

        result = fetch_chip_data("2330")

        mock_fubon.assert_called_once_with("2330")
        response.raise_for_status.assert_called_once_with()
        self.assertEqual(result["data_date"], "2026-03-05")
        self.assertEqual(result["institutional"]["trust_buy_share"], 2)
        self.assertEqual(result["institutional"]["dealer_buy_share"], 0)
        self.assertEqual(
            result["institutional_trading"]["latest_day"]["investment_trust"],
            {"buy_lots": 2, "sell_lots": 0, "net_buy_lots": 2},
        )
        self.assertTrue(result["holding_positions"]["available"])
        self.assertEqual(
            result["holding_positions"]["positions"]["investment_trust"]["lots"],
            15335,
        )

    @patch(
        "stock_tools.market_data._fetch_fubon_chip_data",
        return_value={"available": False, "error": "mocked"},
    )
    @patch("stock_tools.market_data.requests.get")
    def test_returns_error_when_finmind_has_no_data(self, mock_get, mock_fubon):
        response = Mock()
        response.json.return_value = {"status": 200, "data": []}
        mock_get.return_value = response

        result = fetch_chip_data("2330")

        mock_fubon.assert_called_once_with("2330")
        self.assertIn("找不到 2330", result["error"])

    @patch(
        "stock_tools.market_data._fetch_fubon_chip_data",
        return_value={"available": False, "error": "mocked"},
    )
    @patch("stock_tools.market_data.requests.get")
    def test_preserves_requested_period_totals(self, mock_get, mock_fubon):
        response = Mock()
        response.json.return_value = {
            "status": 200,
            "data": [
                {
                    "date": "2026-03-04",
                    "name": "Investment_Trust",
                    "buy": 9000,
                    "sell": 0,
                },
                {
                    "date": "2026-03-05",
                    "name": "Investment_Trust",
                    "buy": 2500,
                    "sell": 500,
                },
            ],
        }
        mock_get.return_value = response

        result = fetch_chip_data("2330", start_date="2026-03-01")

        mock_fubon.assert_called_once_with("2330")
        self.assertEqual(result["institutional"]["trust_buy_share"], 11)
        self.assertEqual(
            result["institutional_trading"]["latest_day"]["investment_trust"]["net_buy_lots"],
            2,
        )

    def test_parses_fubon_institutional_holding_rows(self):
        html = """
        <table class="t01">
          <tr><td>聯茂(6213)籌碼流向<div>日期：11/02</div></td></tr>
          <tr><td></td><td>最新張數</td><td>當日(11/02)增減</td><td>佔股本比例</td></tr>
          <tr><td>投信持股</td><td>15,335</td><td>32</td><td>4.22%</td></tr>
        </table>
        """

        result = _parse_fubon_chip_page(html, "6213")

        self.assertEqual(result["data_date"], "11/02")
        self.assertEqual(
            result["positions"]["investment_trust"],
            {
                "lots": 15335,
                "daily_change_lots": 32,
                "ownership_pct": 4.22,
            },
        )


if __name__ == "__main__":
    unittest.main()
