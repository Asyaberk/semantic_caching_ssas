import unittest

from backend.services.question_guard import quick_validate_question, route_question_to_cube


class FakeProvider:
    def get_cubes(self):
        return [
            {"name": "cubeAccruement", "caption": "Accruement"},
            {"name": "cubeWaiting", "caption": "Waiting"},
            {"name": "cubeVesselOrder", "caption": "Vessel Order"},
        ]

    def get_dimensions(self, cube_name):
        return {
            "cubeAccruement": [
                {"name": "AccruementDate", "caption": "Date"},
                {"name": "AccruementCompany", "caption": "Country Company"},
            ],
            "cubeWaiting": [
                {"name": "Vessel", "caption": "Vessel"},
                {"name": "WaitingStartDate", "caption": "Waiting Date"},
            ],
            "cubeVesselOrder": [
                {"name": "Vessel", "caption": "Vessel"},
                {"name": "MoorageDate", "caption": "Moorage Date"},
            ],
        }[cube_name]

    def get_measures(self, cube_name):
        return {
            "cubeAccruement": [{"name": "Accruement Count", "caption": "Accruement Count"}],
            "cubeWaiting": [{"name": "Waitings Count", "caption": "Waitings Count"}],
            "cubeVesselOrder": [{"name": "Vessel Order Count", "caption": "Vessel Order Count"}],
        }[cube_name]


class QuestionGuardTests(unittest.TestCase):
    def test_rejects_chatter_before_llm(self):
        result = quick_validate_question("merhaba")
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "needs_clarification")

    def test_routes_waiting_question_to_waiting_cube(self):
        result = route_question_to_cube(
            "2024 gemi waiting sürelerini göster",
            FakeProvider(),
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.suggested_cube, "cubeWaiting")

    def test_routes_accruement_question_to_accruement_cube(self):
        result = route_question_to_cube(
            "2025 Türkiye total accruement count nedir?",
            FakeProvider(),
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.suggested_cube, "cubeAccruement")

    def test_rejects_out_of_schema_question(self):
        result = route_question_to_cube("bitcoin fiyatı bugün kaç dolar?", FakeProvider())
        self.assertFalse(result.valid)
        self.assertEqual(result.status, "not_answerable")

    def test_requested_cube_is_respected_when_known(self):
        result = route_question_to_cube(
            "2025 toplam count nedir?",
            FakeProvider(),
            requested_cube="cubeVesselOrder",
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.suggested_cube, "cubeVesselOrder")


if __name__ == "__main__":
    unittest.main()
