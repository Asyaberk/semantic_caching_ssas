import unittest

from backend.models.schemas import QAPair
from backend.services.mdx_schema_guard import validate_hierarchy_references
from backend.services.pipeline_validation import validate_pair_candidates


class PipelineValidationTests(unittest.TestCase):
    def test_ambiguous_dimension_children_is_rejected(self):
        dimensions = [
            {"name": "Vessel", "unique_name": "[Vessel]", "hierarchy_count": 4}
        ]
        with self.assertRaisesRegex(ValueError, "ambiguous dimension"):
            validate_hierarchy_references(
                "SELECT [Vessel].Children ON ROWS FROM [cubeWaiting]",
                dimensions,
            )

    def test_explicit_hierarchy_is_allowed(self):
        dimensions = [
            {"name": "Vessel", "unique_name": "[Vessel]", "hierarchy_count": 4}
        ]
        validate_hierarchy_references(
            "SELECT [Vessel].[Vessel Name].Members ON ROWS FROM [cubeWaiting]",
            dimensions,
        )

    def test_only_successful_candidates_are_distinguishable(self):
        pairs = [
            QAPair(question="good", mdx="good", cube_name="cubeWaiting"),
            QAPair(question="empty", mdx="empty", cube_name="cubeWaiting"),
            QAPair(question="bad", mdx="bad", cube_name="cubeWaiting"),
        ]

        def run(mdx):
            if mdx == "good":
                return {"rows": [{"Value": 1}], "rowCount": 1}
            if mdx == "empty":
                return {"rows": [], "rowCount": 0}
            raise RuntimeError("invalid hierarchy")

        outcomes = validate_pair_candidates(pairs, run_mdx=run)
        self.assertEqual([item.status for item in outcomes], ["success", "no_data", "failed"])


if __name__ == "__main__":
    unittest.main()
