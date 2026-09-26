"""Optional, stdlib-only client for the HERE Location Services APIs.

Endpoints (verified against docs.here.com):

* Geocoding     — ``https://geocode.search.hereapi.com/v1/geocode``
* Reverse geo   — ``https://revgeocode.search.hereapi.com/v1/revgeocode``
* Routing       — ``https://router.hereapi.com/v8/routes`` (traffic enabled)
* Traffic flow  — ``https://data.traffic.hereapi.com/v7/flow`` (bbox + shape)

Design mirrors ``src/data/osm_ingestion.py``'s optional-dependency rule:

* **No new dependency.** The platform requirements file carries no HTTP
  library, so this module speaks HTTP through :mod:`urllib` and parses with
  :mod:`json` only.
* **No key, no lies.** Without an API key every call raises
  :class:`HERENotConfigured` with an actionable message — nothing is faked,
  stubbed, or replayed from an old run.
* **Project units.** Metres, seconds, metres/second. HERE's v7 flow
  ``speed``/``freeFlow`` values are natively m/s and pass through unchanged
  (the legacy v6 ``S`` kph field is deliberately not used).

A key can come from the environment (``HERE_API_KEY`` or
``QUANTA_HERE_API_KEY``) or from the repo ``.env`` / ``frontend/.env.local``
files, mirroring how ``app.auth.config`` resolves every other setting.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

GEOCODE_URL = "https://geocode.search.hereapi.com/v1/geocode"
REVGEOCODE_URL = "https://revgeocode.search.hereapi.com/v1/revgeocode"
ROUTER_URL = "https://router.hereapi.com/v8/routes"
FLOW_URL = "https://data.traffic.hereapi.com/v7/flow"

#: Environment variables checked, in order, for the API key.
KEY_ENV_VARS = ("HERE_API_KEY", "QUANTA_HERE_API_KEY")
#: Free key self-service portal (where a new user is sent by our errors).
PORTAL_URL = "https://portal.here.com"
#: Where keys live in the repo-level env files (mirrors ``app.auth.config``).
_REPO_ROOT = Path(__file__).resolve().parents[2]


class HEREError(RuntimeError):
    """HERE rejected the request, was unreachable, or broke its own schema."""


class HERENotConfigured(HEREError):
    """No HERE API key is present, so the request was never sent."""


def _read_env(name: str) -> str:
    """Resolve one setting: process env first, then the repo env files.

    Same precedence as ``app.auth.config`` (which reads ``.env`` and
    ``frontend/.env.local``, including the ``VITE_`` alias) without making
    ``src/`` import from ``app/``.
    """
    value = (os.environ.get(name) or "").strip()
    if value:
        return value
    for path in (_REPO_ROOT / ".env", _REPO_ROOT / "frontend" / ".env.local"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, sep, raw = line.strip().partition("=")
            if not sep or key not in (name, f"VITE_{name}"):
                continue
            raw = raw.strip().strip('"').strip("'")
            if raw:
                return raw
    return ""


@dataclass(frozen=True)
class HEREConfig:
    """Key + timeout for every HERE call; ``api_key=None`` means unconfigured."""

    api_key: str | None = None
    timeout_s: float = 10.0

    @classmethod
    def from_env(cls, timeout_s: float = 10.0) -> "HEREConfig":
        for name in KEY_ENV_VARS:
            key = _read_env(name)
            if key:
                return cls(api_key=key, timeout_s=timeout_s)
        return cls(api_key=None, timeout_s=timeout_s)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def key_source(self) -> str | None:
        """Which env var supplied the key (``"injected"`` when passed in)."""
        if not self.api_key:
            return None
        for name in KEY_ENV_VARS:
            if _read_env(name) == self.api_key:
                return name
        return "injected"


@dataclass(frozen=True)
class GeocodeItem:
    """One HERE geocoding result (address label + WGS84 position)."""

    label: str
    lat: float
    lon: float
    city: str | None = None
    state: str | None = None
    country: str | None = None
    result_type: str | None = None
    distance_m: float | None = None


@dataclass(frozen=True)
class RouteSummary:
    """HERE v8 route summary. ``delay_s`` is duration minus base duration."""

    duration_s: float
    length_m: float
    base_duration_s: float | None
    delay_s: float
    transport_mode: str


@dataclass(frozen=True)
class FlowSegment:
    """One HERE v7 flow result: link shape polylines + current speeds (m/s)."""

    #: One polyline per link, each a tuple of ``(lat, lon)`` shape points.
    polylines: tuple[tuple[tuple[float, float], ...], ...]
    speed_mps: float
    free_flow_mps: float | None = None
    jam_factor: float | None = None
    confidence: float | None = None
    traversability: str | None = None


def _error_detail(raw: bytes) -> str:
    """Pull the human message out of an HERE error body (best effort)."""
    try:
        payload: Any = json.loads(raw.decode("utf-8", errors="replace"))
    except (ValueError, AttributeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    message = payload.get("message") or payload.get("error_description")
    if not message and isinstance(payload.get("error"), str):
        message = payload["error"]
    cause = payload.get("cause")
    if message and cause:
        return f"{message} ({cause})"
    return str(message) if message else ""


class HEREClient:
    """Thin HTTP facade over the four HERE endpoints listed in the module."""

    def __init__(self, config: HEREConfig | None = None) -> None:
        self.config = config or HEREConfig.from_env()

    # ── plumbing ────────────────────────────────────────────────────────

    def _require_key(self) -> str:
        if not self.config.api_key:
            raise HERENotConfigured(
                "HERE API key is not configured — set HERE_API_KEY (or put it "
                f"in .env) from {PORTAL_URL} → 'Access keys'. No request was "
                "sent, so no HERE values are shown."
            )
        return self.config.api_key

    def _get(self, url: str, params: dict[str, Any]) -> dict:
        query = {key: value for key, value in params.items() if value is not None}
        query["apiKey"] = self._require_key()
        request = Request(
            f"{url}?{urlencode(query)}",
            headers={
                "Accept": "application/json",
                "User-Agent": "quanta-sih26137/0.4 (+https://github.com/quanta)",
            },
        )
        try:
            with urlopen(request, timeout=self.config.timeout_s) as response:
                body = response.read()
        except HTTPError as exc:
            try:
                detail = _error_detail(exc.read())
            except Exception:
                detail = ""
            suffix = f": {detail}" if detail else ""
            hint = ""
            if exc.code in (401, 403):
                hint = " — check that the HERE key is valid and has this API enabled"
            raise HEREError(f"HERE API returned HTTP {exc.code}{suffix}{hint}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise HEREError(f"HERE API is unreachable ({reason})") from exc
        try:
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise HEREError("HERE API returned a non-JSON body") from exc
        if isinstance(payload, dict) and payload.get("error"):
            detail = _error_detail(body) or str(payload.get("error"))
            raise HEREError(f"HERE API error: {detail}")
        if not isinstance(payload, dict):
            raise HEREError("HERE API returned an unexpected payload shape")
        return payload

    # ── geocoding ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_items(payload: dict, fallback_label: str) -> list[GeocodeItem]:
        items: list[GeocodeItem] = []
        for raw in payload.get("items") or []:
            if not isinstance(raw, dict):
                continue
            position = raw.get("position") or {}
            if "lat" not in position or "lng" not in position:
                continue
            address = raw.get("address") or {}
            distance = raw.get("distance")
            items.append(
                GeocodeItem(
                    label=str(
                        address.get("label")
                        or raw.get("title")
                        or fallback_label
                    ),
                    lat=float(position["lat"]),
                    lon=float(position["lng"]),
                    city=address.get("city"),
                    state=address.get("state"),
                    country=address.get("countryName"),
                    result_type=raw.get("resultType"),
                    distance_m=float(distance) if distance is not None else None,
                )
            )
        return items

    def geocode(
        self,
        query: str,
        *,
        limit: int = 5,
        at: tuple[float, float] | None = None,
    ) -> list[GeocodeItem]:
        """Forward geocode a free-text address (optionally biased to ``at``)."""
        params: dict[str, Any] = {
            "q": query,
            "limit": max(1, min(int(limit), 20)),
        }
        if at is not None:
            params["at"] = f"{at[0]:.7f},{at[1]:.7f}"
        payload = self._get(GEOCODE_URL, params)
        return self._parse_items(payload, fallback_label=query)

    def reverse_geocode(
        self, lat: float, lon: float, *, limit: int = 1
    ) -> list[GeocodeItem]:
        """Resolve a coordinate to the nearest address label(s)."""
        payload = self._get(
            REVGEOCODE_URL,
            {"at": f"{lat:.7f},{lon:.7f}", "limit": max(1, min(int(limit), 20))},
        )
        return self._parse_items(payload, fallback_label=f"{lat:.5f}, {lon:.5f}")

    # ── routing ─────────────────────────────────────────────────────────

    def route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        *,
        transport_mode: str = "car",
    ) -> RouteSummary:
        """HERE's own traffic-enabled drive time between two coordinates."""
        payload = self._get(
            ROUTER_URL,
            {
                "origin": f"{origin[0]:.7f},{origin[1]:.7f}",
                "destination": f"{destination[0]:.7f},{destination[1]:.7f}",
                "transportMode": transport_mode,
                "return": "summary",
                "routingMode": "fast",
                "traffic[mode]": "enabled",
            },
        )
        try:
            summary = payload["routes"][0]["sections"][0]["summary"]
            duration = float(summary["duration"])
            length = float(summary["length"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise HEREError("HERE routing returned no usable route section") from exc
        base = summary.get("baseDuration")
        base_duration = float(base) if base is not None else None
        return RouteSummary(
            duration_s=duration,
            length_m=length,
            base_duration_s=base_duration,
            delay_s=max(0.0, duration - base_duration) if base_duration is not None else 0.0,
            transport_mode=transport_mode,
        )

    # ── traffic flow ────────────────────────────────────────────────────

    def flow_bbox(
        self, bbox: Sequence[float]
    ) -> tuple[list[FlowSegment], str | None]:
        """Current flow for a ``(west, south, east, north)`` box.

        Returns ``(segments, source_updated)`` where speeds are m/s and
        ``source_updated`` is HERE's own observation timestamp (or None).
        """
        west, south, east, north = (float(value) for value in bbox)
        payload = self._get(
            FLOW_URL,
            {
                "in": f"bbox:{west:.6f},{south:.6f},{east:.6f},{north:.6f}",
                "locationReferencing": "shape",
            },
        )
        segments: list[FlowSegment] = []
        for raw in payload.get("results") or []:
            if not isinstance(raw, dict):
                continue
            shape = raw.get("location") or {}
            if not isinstance(shape, dict):
                shape = {}
            polylines: list[tuple[tuple[float, float], ...]] = []
            for link in shape.get("links") or []:
                points = tuple(
                    (float(point["lat"]), float(point["lng"]))
                    for point in link.get("points") or []
                    if isinstance(point, dict)
                    and "lat" in point
                    and "lng" in point
                )
                if len(points) >= 2:
                    polylines.append(points)
            if not polylines:
                continue
            current = raw.get("currentFlow") or {}
            if not isinstance(current, dict) or current.get("speed") is None:
                continue
            free_flow = current.get("freeFlow")
            jam = current.get("jamFactor")
            confidence = current.get("confidence")
            segments.append(
                FlowSegment(
                    polylines=tuple(polylines),
                    speed_mps=float(current["speed"]),
                    free_flow_mps=float(free_flow) if free_flow is not None else None,
                    jam_factor=float(jam) if jam is not None else None,
                    confidence=float(confidence) if confidence is not None else None,
                    traversability=(
                        str(current["traversability"])
                        if current.get("traversability") is not None
                        else None
                    ),
                )
            )
        updated = payload.get("sourceUpdated")
        return segments, (str(updated) if updated else None)
