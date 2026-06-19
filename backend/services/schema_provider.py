"""
Schema provider abstraction layer.

This module defines a common interface (SchemaProvider) for retrieving
cube metadata. Two concrete implementations are provided:

  - MockSchemaProvider: reads from the local mock data file.
    Used during development when no SSAS server is available.

  - SSASSchemaProvider: connects to the SSAS Bridge REST API
    (https://daloglumert.com) and discovers cubes, dimensions,
    measures, and members automatically.

Use get_schema_provider() to obtain the correct implementation based on
the USE_MOCK_CUBE environment variable. The rest of the application should
only interact with the SchemaProvider interface, never with mock or SSAS
classes directly.
"""

import logging
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)


# ── Abstract interface ───────────────────────────────────────────────────────

class SchemaProvider(ABC):
    """
    Common interface that every schema provider must implement.
    Callers use this type so they remain independent of the data source.
    """

    @abstractmethod
    def get_cubes(self) -> list[dict]:
        """Return a list of all available cubes."""
        ...

    @abstractmethod
    def get_dimensions(self, cube_name: str) -> list[dict]:
        """Return all dimensions for the given cube."""
        ...

    @abstractmethod
    def get_measures(self, cube_name: str) -> list[dict]:
        """Return all measures for the given cube."""
        ...

    @abstractmethod
    def get_members(self, cube_name: str) -> dict:
        """Return known dimension members for the given cube."""
        ...

    @abstractmethod
    def get_hierarchies(self, cube_name: str) -> dict:
        """Return hierarchy definitions for the given cube."""
        ...

    def get_dimension_hierarchies(self, cube_name: str, dimension_name: str) -> list[dict]:
        """Return hierarchies for one dimension.

        Providers may override this to avoid fetching every cube dimension.
        """
        dimensions = self.get_dimensions(cube_name)
        dimension = next(
            (
                item for item in dimensions
                if dimension_name in {item.get("name"), item.get("unique_name")}
            ),
            None,
        )
        if not dimension:
            return []
        return self.get_hierarchies(cube_name).get(dimension["unique_name"], [])

    def search_members(
        self,
        cube_name: str,
        query: str,
        dimension_name: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search exact cube members by caption/key text."""
        members = self.get_members(cube_name)
        query_text = query.casefold()
        results: list[dict] = []
        for dimension, items in members.items():
            if dimension_name and dimension_name not in dimension:
                continue
            for item in items:
                caption = str(item.get("caption") or "")
                unique_name = str(
                    item.get("unique_name") or item.get("member_unique_name") or ""
                )
                if query_text in caption.casefold() or query_text in unique_name.casefold():
                    results.append({
                        "caption": caption,
                        "unique_name": unique_name,
                        "dimension_name": dimension,
                    })
                    if len(results) >= limit:
                        return results
        return results


# ── Mock implementation ──────────────────────────────────────────────────────

class MockSchemaProvider(SchemaProvider):
    """
    Returns hardcoded cube schema from the local mock data file.
    No network connection is required.
    """

    def get_cubes(self) -> list[dict]:
        from backend.mock.cube_schema import CUBE
        return [CUBE]

    def get_dimensions(self, cube_name: str) -> list[dict]:
        from backend.mock.cube_schema import DIMENSIONS
        return DIMENSIONS

    def get_measures(self, cube_name: str) -> list[dict]:
        from backend.mock.cube_schema import MEASURES
        return MEASURES

    def get_members(self, cube_name: str) -> dict:
        from backend.mock.cube_schema import MEMBERS
        return MEMBERS

    def get_hierarchies(self, cube_name: str) -> dict:
        from backend.mock.cube_schema import HIERARCHIES
        return HIERARCHIES


# ── SSAS REST API implementation ─────────────────────────────────────────────

class SSASSchemaProvider(SchemaProvider):
    """
    Fetches cube metadata from the SSAS Bridge REST API.

    The API wraps an SSAS server and exposes cube/dimension/measure
    discovery via HTTP. All responses are normalised to the same
    field names used by MockSchemaProvider so the rest of the
    application requires no changes when switching data sources.

    Base URL  : settings.ssas_url   (e.g. https://daloglumert.com)
    Auth      : X-API-Key header    (settings.ssas_api_key)
    DataSource: settings.ssas_data_source (default: "main")
    """

    def __init__(self, connection_url: str, api_key: str = "", data_source: str = "main"):
        self.base_url    = connection_url.rstrip("/")
        self.data_source = data_source
        self._headers    = {
            "X-API-Key":    api_key,
            "Content-Type": "application/json",
        }

    # ── Private helper ────────────────────────────────────────────────────

    def _get(self, path: str, **params) -> dict:
        """Make an authenticated GET request and return parsed JSON."""
        params.setdefault("dataSource", self.data_source)
        resp = httpx.get(
            f"{self.base_url}{path}",
            headers=self._headers,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # ── SchemaProvider interface ──────────────────────────────────────────

    def get_cubes(self) -> list[dict]:
        data = self._get("/api/v1/metadata/cubes")
        return [
            {
                "name":        c["name"],
                "caption":     c.get("caption", c["name"]),
                "description": c.get("description", ""),
                "last_schema_update": c.get("lastSchemaUpdate"),
                "aliases": c.get("aliases", []),
            }
            for c in data.get("cubes", [])
        ]

    def get_dimensions(self, cube_name: str) -> list[dict]:
        data = self._get(f"/api/v1/metadata/cubes/{cube_name}/dimensions")
        return [
            {
                "name":             d["name"],
                "unique_name":      d.get("uniqueName", d["name"]),
                "caption":          d.get("caption", d["name"]),
                "description":      d.get("description", ""),
                "type":             d.get("type", ""),
                "default_hierarchy":d.get("defaultHierarchy", ""),
                "hierarchy_count":  d.get("hierarchyCount", 0),
                "aliases":          d.get("aliases", []),
            }
            for d in data.get("dimensions", [])
        ]

    def get_measures(self, cube_name: str) -> list[dict]:
        data = self._get(f"/api/v1/metadata/cubes/{cube_name}/measures")
        return [
            {
                "name":           m["name"],
                "unique_name":    m.get("uniqueName", m["name"]),
                "caption":        m.get("caption", m["name"]),
                "description":    m.get("description", ""),
                "aggregation":    m.get("aggregation", ""),
                "display_folder": m.get("displayFolder", ""),
                "measure_group":  m.get("measureGroup", ""),
                "format_string":  m.get("formatString", ""),
                "is_calculated":  m.get("isCalculated", False),
                "aliases":        m.get("aliases", []),
            }
            for m in data.get("measures", [])
        ]

    def get_members(self, cube_name: str) -> dict:
        """
        Fetch sample members for each dimension using the search endpoint.
        Uses a single-character query ('a') to retrieve the first batch
        of members for each dimension. Skips dimensions that return errors.
        """
        dimensions = self.get_dimensions(cube_name)
        members: dict = {}

        for dim in dimensions:
            dim_unique = dim["unique_name"]
            try:
                data = self._get(
                    f"/api/v1/metadata/cubes/{cube_name}/members/search",
                    q="a",
                    dimensionName=dim["name"],
                    limit=25,
                )
                members[dim_unique] = [
                    {
                        "caption":     m.get("caption", m.get("name", "")),
                        "unique_name": m.get("uniqueName", ""),
                    }
                    for m in data.get("members", [])
                ]
            except Exception as exc:
                logger.debug(
                    "Could not fetch members for dimension '%s': %s", dim["name"], exc
                )
                members[dim_unique] = []

        return members

    def get_hierarchies(self, cube_name: str) -> dict:
        """
        Fetch hierarchy and level info for each dimension.
        """
        dimensions = self.get_dimensions(cube_name)
        hierarchies: dict = {}

        for dim in dimensions:
            dim_unique = dim["unique_name"]
            try:
                data = self._get(
                    f"/api/v1/metadata/cubes/{cube_name}/dimensions/{dim['name']}/hierarchies"
                )
                hierarchies[dim_unique] = data.get("hierarchies", [])
            except Exception as exc:
                logger.debug(
                    "Could not fetch hierarchies for dimension '%s': %s", dim["name"], exc
                )
                hierarchies[dim_unique] = []

        return hierarchies

    def get_dimension_hierarchies(self, cube_name: str, dimension_name: str) -> list[dict]:
        data = self._get(
            f"/api/v1/metadata/cubes/{cube_name}/dimensions/{dimension_name}/hierarchies"
        )
        return data.get("hierarchies", [])

    def search_members(
        self,
        cube_name: str,
        query: str,
        dimension_name: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        params = {"q": query, "limit": limit}
        if dimension_name:
            params["dimensionName"] = dimension_name
        data = self._get(
            f"/api/v1/metadata/cubes/{cube_name}/members/search",
            **params,
        )
        return [
            {
                "caption": item.get("caption", item.get("name", "")),
                "unique_name": item.get("uniqueName", ""),
                "dimension_name": item.get("dimensionName", dimension_name or ""),
                "hierarchy_name": item.get("hierarchyName", ""),
                "level_name": item.get("levelName", ""),
            }
            for item in data.get("members", [])
        ]


# ── Factory ──────────────────────────────────────────────────────────────────

def get_schema_provider() -> SchemaProvider:
    """
    Returns the appropriate SchemaProvider based on the USE_MOCK_CUBE
    environment variable.

    USE_MOCK_CUBE=true  → MockSchemaProvider  (default, no server needed)
    USE_MOCK_CUBE=false → SSASSchemaProvider  (requires a live SSAS endpoint)
    """
    from backend.config import settings

    if settings.use_mock_cube:
        return MockSchemaProvider()

    return SSASSchemaProvider(
        connection_url=settings.ssas_url,
        api_key=settings.ssas_api_key,
        data_source=getattr(settings, "ssas_data_source", "main"),
    )
