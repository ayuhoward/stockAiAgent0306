import unittest
from unittest.mock import patch

from stock_tools.portfolio import get_positions


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


if __name__ == "__main__":
    unittest.main()
