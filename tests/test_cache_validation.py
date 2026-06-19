import unittest

from backend.services.cache_validation import classify_bridge_result


class CacheValidationTests(unittest.TestCase):
    def test_rows_are_success(self):
        status, row_count = classify_bridge_result(
            {"rows": [{"Value": 12}], "rowCount": 1}
        )
        self.assertEqual(status, "success")
        self.assertEqual(row_count, 1)

    def test_empty_result_is_no_data(self):
        status, row_count = classify_bridge_result({"rows": [], "rowCount": 0})
        self.assertEqual(status, "no_data")
        self.assertEqual(row_count, 0)


if __name__ == "__main__":
    unittest.main()
