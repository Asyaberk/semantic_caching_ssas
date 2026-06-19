import unittest

from backend.services.cube_explorer import (
    build_member_preview_mdx,
    shape_result,
    validate_readonly_mdx,
)


class CubeExplorerTests(unittest.TestCase):
    def test_member_preview_is_bounded(self):
        mdx = build_member_preview_mdx(
            cube_name="cubeWaiting",
            hierarchy_unique_name="[Vessel].[Vessel Name]",
            measure_unique_name="[Measures].[Waitings Count]",
            limit=100,
        )
        self.assertIn("HEAD([Vessel].[Vessel Name].Members, 100)", mdx)
        self.assertTrue(mdx.endswith("FROM [cubeWaiting]"))

    def test_console_rejects_cube_mismatch(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_readonly_mdx(
                "SELECT {} ON 0 FROM [cubeWaiting]", "cubeAccruement"
            )

    def test_console_rejects_non_query_command(self):
        with self.assertRaisesRegex(ValueError, "SELECT or WITH"):
            validate_readonly_mdx("DRILLTHROUGH SELECT FROM [cubeWaiting]", "cubeWaiting")

    def test_result_is_limited_and_marked_truncated(self):
        result = shape_result(
            {"columns": [], "rows": [{"v": n} for n in range(5)], "rowCount": 5},
            "SELECT ...",
            2,
        )
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["source_row_count"], 5)
        self.assertTrue(result["truncated"])


if __name__ == "__main__":
    unittest.main()
