"""
Schema provider abstraction layer.

This module defines a common interface (SchemaProvider) for retrieving
cube metadata. Two concrete implementations are provided:

  - MockSchemaProvider: reads from the local mock data file.
    Used during development when no SSAS server is available.

  - SSASSchemaProvider: connects to a live SSAS server via XMLA
    and discovers cubes, dimensions, measures, and members automatically.
    The internals are left as a placeholder until a real endpoint is available.

Use get_schema_provider() to obtain the correct implementation based on
the USE_MOCK_CUBE environment variable. The rest of the application should
only interact with the SchemaProvider interface, never with mock or SSAS
classes directly.
"""

from abc import ABC, abstractmethod


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


# ── SSAS (real) implementation ───────────────────────────────────────────────

class SSASSchemaProvider(SchemaProvider):
    """
    Connects to a live SSAS server and fetches cube metadata via XMLA.

    When instantiated, it opens a connection to the given endpoint.
    All methods query the server on demand and return normalised dicts
    in the same shape as MockSchemaProvider so callers need no changes.

    TODO: implement using the xmla or adodbapi library once a real
    SSAS endpoint is reachable (e.g. exposed via a secure tunnel).
    """

    def __init__(self, connection_url: str, api_key: str = ""):
        self.connection_url = connection_url
        self.api_key = api_key
        # TODO: open XMLA session here

    def get_cubes(self) -> list[dict]:
        # TODO: run DISCOVER CUBES XMLA command and normalise the response
        raise NotImplementedError("Real SSAS connection is not yet implemented.")

    def get_dimensions(self, cube_name: str) -> list[dict]:
        # TODO: run DISCOVER DIMENSIONS XMLA command
        raise NotImplementedError("Real SSAS connection is not yet implemented.")

    def get_measures(self, cube_name: str) -> list[dict]:
        # TODO: run DISCOVER MEASURES XMLA command
        raise NotImplementedError("Real SSAS connection is not yet implemented.")

    def get_members(self, cube_name: str) -> dict:
        # TODO: run DISCOVER MEMBERS XMLA command per dimension
        raise NotImplementedError("Real SSAS connection is not yet implemented.")

    def get_hierarchies(self, cube_name: str) -> dict:
        # TODO: run DISCOVER HIERARCHIES XMLA command
        raise NotImplementedError("Real SSAS connection is not yet implemented.")


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
    )
