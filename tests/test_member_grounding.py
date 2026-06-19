import unittest

from backend.services.member_grounding import (
    extract_member_candidates,
    find_grounded_members,
    format_grounding_for_prompt,
)


class FakeMemberProvider:
    def get_dimensions(self, cube_name):
        return [
            {"name": "Company", "unique_name": "[Company]"},
            {"name": "Vessel", "unique_name": "[Vessel]"},
            {"name": "Measures", "unique_name": "[Measures]"},
        ]

    def search_members(self, cube_name, query, dimension_name=None, limit=10):
        values = {
            ("Turkey", "Company"): [
                {
                    "caption": "Turkey",
                    "unique_name": "[Company].[Country].&[TR]",
                    "dimension_name": "Company",
                }
            ],
            ("Ever Given", "Vessel"): [
                {
                    "caption": "EVER GIVEN",
                    "unique_name": "[Vessel].[Vessel Name].&[EVER GIVEN]",
                    "dimension_name": "Vessel",
                }
            ],
        }
        return values.get((query, dimension_name), [])[:limit]

    def get_measures(self, cube_name):
        return [{"name": "Waitings Count", "caption": "Waitings Count", "unique_name": "[Measures].[Waitings Count]"}]


class MemberGroundingTests(unittest.TestCase):
    def test_extracts_likely_member_values(self):
        self.assertEqual(
            extract_member_candidates('Show waiting count for "Ever Given" in Turkey for 2025'),
            ["Ever Given", "Turkey"],
        )

    def test_finds_exact_member_unique_names(self):
        result = find_grounded_members(
            'Show waiting count for "Ever Given" in Turkey',
            "cubeWaiting",
            FakeMemberProvider(),
        )
        unique_names = {item["unique_name"] for item in result["matches"]}
        self.assertIn("[Company].[Country].&[TR]", unique_names)
        self.assertIn("[Vessel].[Vessel Name].&[EVER GIVEN]", unique_names)
        self.assertEqual(result["unmatched"], [])

    def test_unmatched_values_are_explicitly_blocked_in_prompt(self):
        result = find_grounded_members(
            "Show waiting count for Atlantis",
            "cubeWaiting",
            FakeMemberProvider(),
        )
        prompt = format_grounding_for_prompt(result)
        self.assertIn("Unmatched candidate values: Atlantis", prompt)
        self.assertIn("Do not add filters", prompt)


if __name__ == "__main__":
    unittest.main()
