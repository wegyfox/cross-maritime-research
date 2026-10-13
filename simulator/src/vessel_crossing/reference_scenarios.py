"""Reference geometries expressed in the modern simulation units."""

from __future__ import annotations

from .geometry import CrossingGeometry, Route


def build_diploma_reference_geometry_nm() -> CrossingGeometry:
    """Return the diploma geometry converted from cables to nautical miles.

    1 cable = 0.1 NM, therefore the original 100 x 100 cable area becomes
    10 x 10 NM and the 10-cable separation of opposing lanes becomes 1 NM.
    """

    return CrossingGeometry(
        routes=(
            Route("A", (4.5, 10.0), (4.5, 0.0)),
            Route("B", (5.5, 0.0), (5.5, 10.0)),
            Route("C", (0.0, 4.5), (10.0, 4.5)),
            Route("D", (10.0, 5.5), (0.0, 5.5)),
        )
    )
