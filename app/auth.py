"""Verify company access with Supabase; demo computation has no private storage access."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from fastapi import HTTPException


def config(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    root = Path(__file__).resolve().parents[1]
    for path in (root / ".env", root / "frontend" / ".env.local"):
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                key, sep, value = line.strip().partition("=")
                if sep and key in (name, f"VITE_{name}"):
                    value = value.strip().strip('"').strip("'")
                    if value:
                        return value
    return default


def supabase_json(path: str, token: str):
    base = config("SUPABASE_URL").rstrip("/")
    key = config("SUPABASE_PUBLISHABLE_KEY") or config("SUPABASE_ANON_KEY")
    if not base or not key:
        raise HTTPException(503, "Company access is not configured on the API server.")
    parsed = urlparse(base)
    if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1"):
        raise HTTPException(503, "Supabase must use HTTPS.")
    request = Request(base + path, headers={"apikey": key, "Authorization": f"Bearer {token}"})
    try:
        with urlopen(request, timeout=10) as response:
            return json.load(response)
    except HTTPError as exc:
        raise HTTPException(401 if exc.code in (401, 403) else 503,
                            "Session could not be verified. Sign in again." if exc.code in (401, 403)
                            else "Company access verification is unavailable.") from exc
    except (URLError, TimeoutError) as exc:
        raise HTTPException(503, "Company access verification is unavailable.") from exc


def verify_company(authorization: str, company_id: str):
    if not authorization.startswith("Bearer ") or not company_id:
        raise HTTPException(401, "Sign in and select a company workspace.")
    token = authorization[7:]
    user = supabase_json("/auth/v1/user", token)
    query = urlencode({"company_id": f"eq.{company_id}", "user_id": f"eq.{user['id']}",
                       "select": "role", "limit": "1"})
    rows = supabase_json("/rest/v1/company_members?" + query, token)
    if not rows:
        raise HTTPException(403, "You are not a member of this company.")
    return rows[0]["role"]
