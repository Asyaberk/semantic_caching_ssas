import unittest

from backend.services.mdx_execution import (
    execute_with_repair,
    extract_cube_name,
    fix_bare_year_keys,
)


class MDXExecutionTests(unittest.TestCase):
    def test_extracts_cube_name(self):
        self.assertEqual(
            extract_cube_name("SELECT {} ON 0 FROM [cubeWaiting]"),
            "cubeWaiting",
        )

    def test_unknown_cube_does_not_guess_date_dimension(self):
        mdx = "SELECT {} ON 0 FROM [Unknown] WHERE [Date].[Year].&[2025]"
        self.assertEqual(fix_bare_year_keys(mdx, "Unknown"), mdx)

    def test_known_cube_uses_its_date_dimension(self):
        mdx = "SELECT {} ON 0 FROM [cubeWaiting] WHERE [Date].[Year].&[2025]"
        fixed = fix_bare_year_keys(mdx, "cubeWaiting")
        self.assertIn("[WaitingStartDate].[Date].&[2025-01-01T00:00:00]", fixed)

    def test_empty_rows_are_truthful_no_data_not_a_repair(self):
        repairs = []

        result = execute_with_repair(
            mdx="SELECT {} ON 0 FROM [cubeWaiting]",
            cube_name="cubeWaiting",
            question="test",
            run_mdx=lambda _: {"columns": [], "rows": [], "rowCount": 0},
            repair_mdx=lambda *args: repairs.append(args),
        )

        self.assertEqual(result.status, "no_data")
        self.assertTrue(result.validated)
        self.assertEqual(repairs, [])

    def test_successful_repair_reports_exact_executed_mdx(self):
        original = "SELECT broken ON 0 FROM [cubeWaiting]"
        repaired = "SELECT {[Measures].[Count]} ON 0 FROM [cubeWaiting]"

        def run(mdx):
            if mdx == original:
                raise RuntimeError("bad hierarchy")
            return {"columns": [{"name": "Count"}], "rows": [{"Count": 4}]}

        result = execute_with_repair(
            mdx=original,
            cube_name="cubeWaiting",
            question="How many records are there?",
            run_mdx=run,
            repair_mdx=lambda *_: repaired,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.attempt, "llm_repair")
        self.assertEqual(result.executed_mdx, repaired)

    def test_failed_query_never_falls_back_to_unfiltered_aggregate(self):
        calls = []

        def run(mdx):
            calls.append(mdx)
            raise RuntimeError("invalid MDX")

        result = execute_with_repair(
            mdx="SELECT broken ON 0 FROM [cubeWaiting]",
            cube_name="cubeWaiting",
            question="test",
            run_mdx=run,
            repair_mdx=lambda *_: None,
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(len(calls), 1)

    def test_repair_cannot_switch_to_another_cube(self):
        def run(_):
            raise RuntimeError("invalid MDX")

        result = execute_with_repair(
            mdx="SELECT broken ON 0 FROM [cubeWaiting]",
            cube_name="cubeWaiting",
            question="test",
            run_mdx=run,
            repair_mdx=lambda *_: "SELECT {} ON 0 FROM [cubeAccruement]",
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("target cube", result.error)


if __name__ == "__main__":
    unittest.main()
