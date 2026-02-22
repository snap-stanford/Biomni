"""Knowledge-Provider (KP) tool suite for Translator/BioThings queries.

Provides a ``KPClient`` class that encapsulates the SmartAPI registry cache
and KP calling logic, plus module-level wrapper functions compatible with the
biomni tool-dispatch framework (``getattr(module, name)``).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import random
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    import builtins

__all__ = [
    "KPClient",
]

log = logging.getLogger(__name__)


class KPClient:
    """Stateful client for the SmartAPI KP registry and KP calls.

    Caches the registry in memory so repeated ``query`` / ``describe`` calls
    within the same session skip redundant disk reads and SmartAPI fetches.

    Parameters
    ----------
    cache_path : str | Path | None
        Filesystem path for the JSON registry cache.  ``None`` uses the
        default ``~/.cache/biomni/kp_registry/biothings_transltr_kps.json``.
    max_age_seconds : int | float | None
        Maximum age of the on-disk cache before an automatic refresh is
        triggered.  ``None`` disables age-based refresh.
    """

    # -- class-level constants ----------------------------------------------

    SMARTAPI_BASE = os.environ.get("SMARTAPI_ENDPOINT", "https://smart-api.info/api").rstrip("/")
    SMARTAPI_QUERY_URL = os.environ.get("SMARTAPI_REGISTRY_URL", f"{SMARTAPI_BASE}/query")
    SMARTAPI_TRAPI_TAG = os.environ.get("SMARTAPI_TRAPI_TAG", "trapi")
    SMARTAPI_BIOTHINGS_TAG = os.environ.get("SMARTAPI_BIOTHINGS_TAG", "biothings")

    _HTTP_VERBS = {"get", "post", "put", "delete", "patch", "head", "options"}
    _RETRY_STATUS = {429, 500, 502, 503, 504}

    _DEFAULT_CACHE = None  # lazily resolved in _cache_root()

    # -- construction -------------------------------------------------------

    def __init__(
        self,
        cache_path: str | Path | None = None,
        max_age_seconds: int | float | None = 24 * 3600,
    ) -> None:
        self._cache = self._coerce_path(cache_path, default=self._default_cache_path())
        self._max_age = max_age_seconds
        self._registry: dict[str, Any] | None = None
        self._last_fresh: dict[str, Any] | None = None

    # -----------------------------------------------------------------------
    # Time / misc helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _now_ts() -> float:
        return time.time()

    @staticmethod
    def _utc_epoch() -> int:
        return int(time.time())

    @staticmethod
    def _now_iso() -> str:
        return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        base = min(2**attempt, 10)
        time.sleep(base + random.random() * 0.25)

    @staticmethod
    def _normalize_server_url(url: Any) -> str:
        if not isinstance(url, str):
            return ""
        return url.strip().rstrip("/")

    # -----------------------------------------------------------------------
    # Cache I/O
    # -----------------------------------------------------------------------

    @staticmethod
    def _cache_root() -> Path:
        root = os.environ.get("BIOMNI_CACHE_DIR")
        if root:
            return Path(root).expanduser().resolve()
        return Path.home() / ".cache" / "biomni"

    @classmethod
    def _default_cache_path(cls) -> Path:
        return cls._cache_root() / "kp_registry" / "biothings_transltr_kps.json"

    @staticmethod
    def _coerce_path(p: str | Path | None, *, default: str | Path) -> Path:
        if p is None:
            p = default
        return Path(str(p)).expanduser().resolve()

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        try:
            if not path.exists():
                return default
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Failed to read JSON %s: %s", path, e)
            return default

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True, default=str)
            os.replace(tmp, path)
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def _cache_age_seconds(self) -> float:
        try:
            if not self._cache.exists():
                return float("inf")
            return self._now_ts() - float(self._cache.stat().st_mtime)
        except Exception:
            return float("inf")

    def _registry_exists(self) -> bool:
        try:
            return self._cache.exists() and self._cache.is_file() and self._cache.stat().st_size > 0
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Error / wrapper helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _err(
        type_: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "type": type_,
            "message": message,
            "retryable": bool(retryable),
            "details": details or {},
        }

    @staticmethod
    def _wrap(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Backward-compatible wrapper constructor.

        Style A:
          _wrap(ok, data, errors=None, meta=None)

        Style B (legacy passthrough keys):
          _wrap(kp_id=..., query=..., data=..., ...)
        """
        passthrough = {"kp_id", "query", "normalized", "provenance"}
        if passthrough.intersection(kwargs.keys()):
            errors = kwargs.get("errors") or []
            ok = kwargs.get("ok")
            if ok is None:
                ok = len(errors) == 0
            data = kwargs.get("data")
            meta = kwargs.get("meta") or {}
            provenance = kwargs.get("provenance") or {}
            out: dict[str, Any] = {
                "ok": bool(ok),
                "data": data,
                "errors": errors,
                "meta": {**meta, "provenance": provenance},
            }
            for k in passthrough:
                if k in kwargs:
                    out[k] = kwargs.get(k)
            return out

        if len(args) >= 2 and isinstance(args[0], bool):
            ok = bool(args[0])
            data = args[1]
            errors = args[2] if len(args) >= 3 else kwargs.get("errors")
            meta = kwargs.get("meta")
            return {"ok": ok, "data": data, "errors": errors or [], "meta": meta or {}}

        if "ok" in kwargs and "data" in kwargs:
            return {
                "ok": bool(kwargs["ok"]),
                "data": kwargs["data"],
                "errors": kwargs.get("errors") or [],
                "meta": kwargs.get("meta") or {},
            }

        return {
            "ok": False,
            "data": None,
            "errors": [{"type": "wrap_bad_call", "message": "Invalid _wrap() call signature"}],
            "meta": {
                "args_len": len(args),
                "kwargs_keys": sorted(kwargs.keys()),
            },
        }

    # -----------------------------------------------------------------------
    # HTTP
    # -----------------------------------------------------------------------

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        timeout_s: float = 30.0,
        retries: int = 2,
    ) -> dict[str, Any]:
        start = self._now_ts()
        attempt = 0

        while True:
            try:
                connect = min(10.0, float(timeout_s))
                timeout = httpx.Timeout(connect=connect, read=timeout_s, write=timeout_s, pool=connect)
                with httpx.Client(timeout=timeout, follow_redirects=True) as c:
                    r = c.request(method, url, params=params, json=json_body)

                meta = {
                    "timestamp": self._utc_epoch(),
                    "url": str(r.request.url),
                    "method": method.upper(),
                    "status_code": r.status_code,
                    "elapsed_ms": int((self._now_ts() - start) * 1000),
                    "attempts": attempt + 1,
                }

                if r.status_code >= 400:
                    retryable = (r.status_code in self._RETRY_STATUS) and attempt < int(retries)
                    e = self._err(
                        "HTTPError",
                        f"HTTP {r.status_code}",
                        retryable=retryable,
                        details={"text": r.text[:500]},
                    )
                    if retryable:
                        attempt += 1
                        self._sleep_backoff(attempt)
                        continue
                    return self._wrap(False, None, [e], meta=meta)

                try:
                    data = r.json()
                except Exception as je:
                    e = self._err("JSONDecodeError", str(je), details={"text": r.text[:500]})
                    return self._wrap(False, None, [e], meta=meta)

                return self._wrap(True, data, [], meta=meta)

            except httpx.TimeoutException as te:
                e = self._err("Timeout", str(te), retryable=attempt < int(retries))
            except httpx.HTTPError as he:
                e = self._err("NetworkError", str(he), retryable=attempt < int(retries))
            except Exception as ue:
                e = self._err("UnknownError", str(ue), retryable=False)

            if e["retryable"]:
                attempt += 1
                self._sleep_backoff(attempt)
                continue

            meta = {
                "timestamp": self._utc_epoch(),
                "url": url,
                "method": method.upper(),
                "status_code": None,
                "elapsed_ms": int((self._now_ts() - start) * 1000),
                "attempts": attempt + 1,
            }
            return self._wrap(False, None, [e], meta=meta)

    # -----------------------------------------------------------------------
    # SmartAPI → registry normalization
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_smartapi_hits(payload: Any) -> builtins.list[dict]:
        if not isinstance(payload, dict):
            return []
        hits = payload.get("hits")
        if isinstance(hits, list):
            return hits
        if isinstance(hits, dict) and isinstance(hits.get("hits"), list):
            return hits["hits"]
        return []

    @staticmethod
    def _unwrap_smartapi_hit(hit: Any) -> tuple[str | None, dict]:
        if not isinstance(hit, dict):
            return None, {}
        if isinstance(hit.get("doc"), dict):
            return hit.get("_id") or hit.get("id"), hit["doc"]
        if isinstance(hit.get("_source"), dict):
            return hit.get("_id") or hit.get("id"), hit["_source"]
        if "info" in hit and ("paths" in hit or "openapi" in hit or "swagger" in hit):
            return hit.get("_id") or hit.get("id"), hit
        return hit.get("_id") or hit.get("id"), hit

    @classmethod
    def _stable_kp_id_from_doc(cls, doc: dict, smartapi_id: str | None) -> str:
        if smartapi_id:
            return str(smartapi_id)

        servers = doc.get("servers")
        if isinstance(servers, list) and servers and isinstance(servers[0], dict):
            u = cls._normalize_server_url(servers[0].get("url"))
            if u:
                return u

        info = doc.get("info") if isinstance(doc.get("info"), dict) else {}
        title = str(info.get("title") or "")
        version = str(info.get("version") or "")
        key = f"{title}||{version}||{json.dumps(servers, sort_keys=True, default=str) if servers else ''}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return f"kp_{h}"

    @staticmethod
    def _infer_kp_type_from_openapi(doc: dict) -> str:
        paths = doc.get("paths")
        if not isinstance(paths, dict):
            return "unknown"

        has_query = "/query" in paths
        has_metadata = ("/metadata" in paths) or ("/metadata/fields" in paths)

        if has_query and has_metadata:
            return "biothings"

        if has_query and not has_metadata:
            qops = paths.get("/query")
            if isinstance(qops, dict) and "post" in qops and "get" not in qops:
                return "trapi"

        return "unknown"

    @classmethod
    def _endpoint_index(cls, openapi: dict) -> dict[str, builtins.list[str]]:
        out: dict[str, list[str]] = {}
        paths = openapi.get("paths") if isinstance(openapi, dict) else None
        if not isinstance(paths, dict):
            return out
        for p, methods in paths.items():
            if not isinstance(p, str) or not isinstance(methods, dict):
                continue
            m = [
                k.lower().strip() for k in methods.keys() if isinstance(k, str) and k.lower().strip() in cls._HTTP_VERBS
            ]
            if m:
                out[p] = sorted(set(m))
        return out

    @classmethod
    def _endpoint_flags(cls, openapi: dict, kp_type: str) -> dict[str, Any]:
        idx = cls._endpoint_index(openapi)
        has_query = "/query" in idx
        has_querymany = ("/querymany" in idx) or any("querymany" in p for p in idx.keys())
        has_metadata = ("/metadata" in idx) or ("/meta" in idx)
        has_trapi_query = bool(kp_type == "trapi" and "/query" in idx and "post" in (idx.get("/query") or []))
        return {
            "has_query": bool(has_query),
            "has_querymany": bool(has_querymany),
            "has_metadata": bool(has_metadata),
            "has_trapi_query": bool(has_trapi_query),
            "paths": idx,
        }

    @classmethod
    def _pick_base_url(cls, kp: dict) -> str:
        for v in (kp.get("base_url"), kp.get("url")):
            u = cls._normalize_server_url(v)
            if u:
                return u

        servers = kp.get("servers")
        if isinstance(servers, list) and servers:
            s0 = servers[0]
            if isinstance(s0, dict):
                u = cls._normalize_server_url(s0.get("url"))
                if u:
                    return u
            if isinstance(s0, str):
                u = cls._normalize_server_url(s0)
                if u:
                    return u

        doc = kp.get("_doc")
        if isinstance(doc, dict):
            servers = doc.get("servers")
            if isinstance(servers, list) and servers and isinstance(servers[0], dict):
                u = cls._normalize_server_url(servers[0].get("url"))
                if u:
                    return u

        openapi = kp.get("openapi")
        if isinstance(openapi, dict):
            servers = openapi.get("servers")
            if isinstance(servers, list) and servers and isinstance(servers[0], dict):
                u = cls._normalize_server_url(servers[0].get("url"))
                if u:
                    return u

        return ""

    @classmethod
    def _build_kp_record(cls, hit: dict) -> dict | None:
        smartapi_id, doc = cls._unwrap_smartapi_hit(hit)
        if not isinstance(doc, dict) or not doc:
            return None

        kid = cls._stable_kp_id_from_doc(doc, smartapi_id=smartapi_id)

        info = doc.get("info") if isinstance(doc.get("info"), dict) else {}
        name = info.get("title") or kid

        servers = doc.get("servers")
        if not isinstance(servers, list):
            servers = []

        base_url = ""
        if servers and isinstance(servers[0], dict):
            base_url = cls._normalize_server_url(servers[0].get("url"))

        kp_type = cls._infer_kp_type_from_openapi(doc)
        flags = cls._endpoint_flags(doc, kp_type=kp_type)

        endpoints = {
            "has_query": flags["has_query"],
            "has_querymany": flags["has_querymany"],
            "has_metadata": flags["has_metadata"],
            "has_trapi_query": flags["has_trapi_query"],
            "supports_trapi_query": bool(flags["has_trapi_query"]),
            "paths": flags["paths"],
        }

        trapi = bool(kp_type == "trapi" or endpoints.get("supports_trapi_query"))

        return {
            "id": kid,
            "smartapi_id": smartapi_id,
            "name": name,
            "description": info.get("description"),
            "version": info.get("version"),
            "servers": servers,
            "base_url": base_url,
            "url": base_url,
            "kp_type": kp_type,
            "type": kp_type,
            "x_translator": doc.get("x-translator"),
            "paths": (list((doc.get("paths") or {}).keys()) if isinstance(doc.get("paths"), dict) else []),
            "endpoints": endpoints,
            "trapi": trapi,
            "_doc": doc,
            "fetched_at": cls._utc_epoch(),
        }

    def _fetch_registry_from_smartapi(self, *, max_size: int = 500, timeout_s: float = 30.0) -> dict[str, Any]:
        q = f'tags.name:("{self.SMARTAPI_TRAPI_TAG}" OR "{self.SMARTAPI_BIOTHINGS_TAG}")'
        res = self._request_json(
            method="GET",
            url=self.SMARTAPI_QUERY_URL,
            params={"q": q, "size": max_size},
            json_body=None,
            timeout_s=timeout_s,
            retries=2,
        )

        if not res.get("ok"):
            msg = (res.get("errors") or [{}])[0].get("message", "SmartAPI fetch failed")
            return self._wrap(
                False,
                None,
                [
                    {
                        "type": "smartapi_fetch_failed",
                        "message": str(msg),
                        "retryable": True,
                    }
                ],
                meta={
                    **(res.get("meta") or {}),
                    "source": self.SMARTAPI_QUERY_URL,
                    "requested_size": max_size,
                    "query": q,
                },
            )

        payload = res.get("data") if isinstance(res.get("data"), dict) else {}
        hits = self._extract_smartapi_hits(payload)

        items: list[dict[str, Any]] = []
        seen: set = set()
        dup_count = 0

        for h in hits:
            rec = self._build_kp_record(h)
            if not rec:
                continue

            base_url = rec.get("url") or rec.get("base_url") or rec.get("id") or ""
            smartapi_id = rec.get("smartapi_id")
            key = (base_url, smartapi_id)

            if key in seen:
                dup_count += 1
                continue
            seen.add(key)

            rec_id = rec.get("id") or base_url
            if rec_id and any((isinstance(x, dict) and x.get("id") == rec_id) for x in items):
                if smartapi_id:
                    rec["id"] = f"{rec_id}|{smartapi_id}"
                else:
                    rec["id"] = f"{rec_id}|dup{dup_count}"

            if not rec.get("base_url"):
                rec["base_url"] = self._pick_base_url(rec)
                rec["url"] = rec["base_url"]

            items.append(rec)

        return self._wrap(
            True,
            {"timestamp": self._now_iso(), "items": items, "source": self.SMARTAPI_QUERY_URL},
            meta={
                "requested_size": max_size,
                "hits": len(hits),
                "kept": len(items),
                "duplicates_dropped": dup_count,
                "query": q,
            },
        )

    @staticmethod
    def _normalize_registry(obj: Any) -> dict[str, Any]:
        """Canonical shape: ``{"kps": list, "by_id": dict, "meta": dict}``."""
        kps: list[dict] = []
        meta: dict[str, Any] = {}

        if isinstance(obj, dict) and "kps" in obj:
            kps = obj.get("kps") if isinstance(obj.get("kps"), list) else []
            meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}

        elif isinstance(obj, dict) and isinstance(obj.get("items"), list):
            kps = obj.get("items") or []
            meta = {"timestamp": obj.get("timestamp"), "source": obj.get("source")}

        elif isinstance(obj, dict) and obj.get("ok") is True and isinstance(obj.get("data"), dict):
            data = obj.get("data") or {}
            if isinstance(data.get("items"), list):
                kps = data.get("items") or []
                meta = {
                    "timestamp": data.get("timestamp"),
                    "source": data.get("source"),
                    "smartapi_meta": obj.get("meta") or {},
                }

        by_id: dict[str, dict] = {}
        for kp in kps:
            if isinstance(kp, dict) and kp.get("id") is not None:
                by_id[str(kp["id"])] = kp

        if not isinstance(meta, dict):
            meta = {}

        return {"kps": kps, "by_id": by_id, "meta": meta}

    def _load_registry(self) -> dict[str, Any]:
        obj = self._read_json(self._cache, default={"kps": [], "by_id": {}, "meta": {}})
        return self._normalize_registry(obj)

    def _save_registry(self, reg: Any) -> None:
        reg2 = self._normalize_registry(reg)
        reg2["meta"] = {
            **(reg2.get("meta") or {}),
            "updated_at_unix": self._now_ts(),
            "cache_path": str(self._cache),
        }
        self._atomic_write_json(self._cache, reg2)

    # -----------------------------------------------------------------------
    # KP calling helpers
    # -----------------------------------------------------------------------

    def _call_kp(
        self,
        kp_meta: dict[str, Any] | str,
        path: str,
        *,
        method: str = "POST",
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        timeout_s: float = 30.0,
        retries: int = 2,
    ) -> dict[str, Any]:
        if isinstance(kp_meta, str):
            base_url = kp_meta.rstrip("/")
        elif isinstance(kp_meta, dict):
            base_url = self._pick_base_url(kp_meta).rstrip("/")
        else:
            return self._wrap(
                False,
                None,
                [
                    {
                        "type": "bad_kp_meta",
                        "message": "kp_meta must be dict or base_url string",
                    }
                ],
                meta={"path": path, "method": method},
            )

        if not base_url:
            return self._wrap(
                False,
                None,
                [{"type": "no_base_url", "message": "No base URL available for KP call"}],
                meta={"path": path, "method": method},
            )

        if not path.startswith("/"):
            path = "/" + path

        return self._request_json(
            method=method,
            url=base_url + path,
            params=params,
            json_body=json_body,
            timeout_s=float(timeout_s),
            retries=int(retries),
        )

    @staticmethod
    def _is_trapi_kp(kp: dict) -> bool:
        endpoints = kp.get("endpoints") if isinstance(kp.get("endpoints"), dict) else {}
        return bool(
            kp.get("trapi")
            or kp.get("kp_type") == "trapi"
            or kp.get("type") == "trapi"
            or (
                isinstance(endpoints, dict)
                and (endpoints.get("supports_trapi_query") or endpoints.get("has_trapi_query"))
            )
        )

    @staticmethod
    def _normalize_biothings_result(raw: Any, *, include_raw: bool = False) -> dict[str, Any]:
        if raw is None:
            out: dict[str, Any] = {"hits": [], "total": 0}
            return {**out, "raw": raw} if include_raw else out

        if isinstance(raw, list):
            out = {"hits": raw, "total": len(raw)}
            return {**out, "raw": raw} if include_raw else out

        if isinstance(raw, dict):
            hits = raw.get("hits")
            total = raw.get("total")
            scroll_id = raw.get("_scroll_id") or raw.get("scroll_id")

            if isinstance(hits, list):
                if isinstance(total, int):
                    out = {"hits": hits, "total": total, "scroll_id": scroll_id}
                elif isinstance(total, dict) and isinstance(total.get("value"), int):
                    out = {
                        "hits": hits,
                        "total": int(total["value"]),
                        "scroll_id": scroll_id,
                    }
                else:
                    out = {"hits": hits, "total": len(hits), "scroll_id": scroll_id}
                return {**out, "raw": raw} if include_raw else out

            if "_id" in raw:
                out = {"hits": [raw], "total": 1, "scroll_id": scroll_id}
                return {**out, "raw": raw} if include_raw else out

        out = {"hits": [], "total": 0}
        return {**out, "raw": raw} if include_raw else out

    # -----------------------------------------------------------------------
    # Shared plumbing (registry init, KP lookup, response post-processing)
    # -----------------------------------------------------------------------

    def _ensure_registry(self, force: bool = False) -> dict[str, Any]:
        """Refresh if needed, (re)load from disk, cache in ``self._registry``."""
        fresh = self._do_refresh(force=force)
        self._last_fresh = fresh
        self._registry = self._load_registry()
        return fresh

    def _do_refresh(self, force: bool = False) -> dict[str, Any]:
        t0 = self._now_ts()
        try:
            age = self._cache_age_seconds()
            threshold = float("inf") if self._max_age is None else float(self._max_age)
            need = bool(force) or (not self._registry_exists()) or (age > threshold)

            if not need:
                return self._wrap(
                    True,
                    {"refreshed": False, "age_seconds": age},
                    meta={
                        "cache_path": str(self._cache),
                        "elapsed_s": self._now_ts() - t0,
                    },
                )

            fetched = self._fetch_registry_from_smartapi(max_size=500, timeout_s=30.0)
            if not fetched.get("ok"):
                fallback_age = self._cache_age_seconds()
                return self._wrap(
                    False,
                    {
                        "refreshed": False,
                        "age_seconds": (None if fallback_age == float("inf") else fallback_age),
                        "fallback_cache_used": bool(self._registry_exists()),
                    },
                    fetched.get("errors")
                    or [
                        {
                            "type": "refresh_failed",
                            "message": "SmartAPI registry fetch failed",
                            "retryable": True,
                        }
                    ],
                    meta={
                        "cache_path": str(self._cache),
                        "elapsed_s": self._now_ts() - t0,
                    },
                )

            reg = self._normalize_registry(fetched)
            self._save_registry(reg)

            age2 = self._cache_age_seconds()
            return self._wrap(
                True,
                {"refreshed": True, "age_seconds": age2},
                meta={
                    "cache_path": str(self._cache),
                    "elapsed_s": self._now_ts() - t0,
                },
            )
        except Exception as e:
            fallback_age = self._cache_age_seconds()
            return self._wrap(
                False,
                {
                    "refreshed": False,
                    "age_seconds": (None if fallback_age == float("inf") else fallback_age),
                    "fallback_cache_used": bool(self._registry_exists()),
                },
                [
                    {
                        "type": "refresh_failed",
                        "message": str(e),
                        "retryable": True,
                    }
                ],
                meta={
                    "cache_path": str(self._cache),
                    "elapsed_s": self._now_ts() - t0,
                },
            )

    def _get_kp(self, kp_id: str) -> tuple[dict | None, dict, dict]:
        """Return ``(kp_record | None, fresh_result, error_wrapper | {})``."""
        fresh = self._ensure_registry()
        reg = self._registry

        if reg is None:
            return (
                None,
                fresh,
                self._wrap(
                    False,
                    None,
                    [
                        {
                            "type": "load_registry_failed",
                            "message": "Registry not loaded",
                        }
                    ],
                    meta={
                        "kp_id": kp_id,
                        "cache_path": str(self._cache),
                        "refresh": fresh.get("data"),
                    },
                ),
            )

        kp = None
        by_id = reg.get("by_id")
        if isinstance(by_id, dict):
            kp = by_id.get(kp_id)
        if kp is None and isinstance(reg.get("kps"), list):
            kp = next(
                (x for x in reg["kps"] if isinstance(x, dict) and x.get("id") == kp_id),
                None,
            )

        if kp is None:
            return (
                None,
                fresh,
                self._wrap(
                    False,
                    None,
                    [
                        {
                            "type": "kp_not_found",
                            "message": f"kp_id={kp_id} not found",
                        }
                    ],
                    meta={
                        "kp_id": kp_id,
                        "cache_path": str(self._cache),
                        "refresh": fresh.get("data"),
                    },
                ),
            )

        kp = dict(kp)
        base_url = self._pick_base_url(kp)
        if base_url:
            kp["base_url"] = base_url
            kp["url"] = base_url

        return kp, fresh, {}

    @staticmethod
    def _validate_trapi_response(res: dict[str, Any]) -> None:
        """Validate / unwrap a TRAPI response in-place."""
        if res.get("ok") and isinstance(res.get("data"), dict):
            d = res["data"]
            if "message" not in d and isinstance(d.get("data"), dict) and "message" in d["data"]:
                res["data"] = d["data"]
            elif "message" not in d:
                res.setdefault("errors", []).append(
                    {
                        "type": "trapi_bad_response",
                        "message": "TRAPI response missing top-level 'message' field",
                    }
                )
                res["ok"] = False

    @staticmethod
    def _maybe_normalize_biothings(
        res: dict[str, Any],
        normalize: bool,
        include_raw_hits: bool,
    ) -> None:
        """Attach ``meta.normalized`` to a BioThings result in-place."""
        if normalize and res.get("ok"):
            try:
                res["meta"]["normalized"] = KPClient._normalize_biothings_result(
                    res.get("data"), include_raw=include_raw_hits
                )
            except Exception as e:
                res.setdefault("errors", []).append({"type": "normalize_failed", "message": str(e)})
                res["ok"] = False
                res["meta"]["normalized"] = None

    @staticmethod
    def _attach_refresh_errors(res: dict[str, Any], fresh: dict[str, Any]) -> None:
        """Propagate registry-refresh warnings into *res*."""
        if not fresh.get("ok"):
            res.setdefault("meta", {})["refresh_errors"] = fresh.get("errors") or []

    # ===================================================================
    # Public API
    # ===================================================================

    def refresh(
        self,
        max_age_seconds: int | float | None = None,
        force: bool = False,
        **_ignored_kwargs: Any,
    ) -> dict:
        if max_age_seconds is not None:
            old = self._max_age
            self._max_age = max_age_seconds
            result = self._do_refresh(force=force)
            self._max_age = old
        else:
            result = self._do_refresh(force=force)
        self._registry = None
        return result

    def list(self) -> dict:
        fresh = self._ensure_registry()
        reg = self._registry or {}
        kps = reg.get("kps") if isinstance(reg, dict) else []

        errors: list[dict] = []
        if not fresh.get("ok"):
            errors.extend(fresh.get("errors") or [])
        if not isinstance(kps, list) or not kps:
            errors.append(
                {
                    "type": "registry_empty",
                    "message": "No KPs available in registry cache.",
                }
            )

        ok = isinstance(kps, list) and len(kps) > 0
        return self._wrap(
            ok,
            kps if isinstance(kps, list) else [],
            errors,
            meta={
                "cache_path": str(self._cache),
                "cache_age_seconds": self._cache_age_seconds(),
                "registry_meta": reg.get("meta", {}) if isinstance(reg, dict) else {},
                "refresh": fresh.get("data"),
            },
        )

    def describe(self, kp_id: str) -> dict:
        kp, fresh, err = self._get_kp(kp_id)
        if err:
            return err
        errs = [] if fresh.get("ok") else (fresh.get("errors") or [])
        return self._wrap(
            True,
            kp,
            errs,
            meta={
                "kp_id": kp_id,
                "cache_path": str(self._cache),
                "refresh": fresh.get("data"),
            },
        )

    def query(
        self,
        kp_id: str,
        q: Any,
        *,
        normalize: bool = False,
        include_raw_hits: bool = False,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        kp, fresh, err = self._get_kp(kp_id)
        if err:
            return err

        is_trapi = self._is_trapi_kp(kp)

        if is_trapi:
            trapi_body = q if isinstance(q, dict) else {"message": q}
            res = self._call_kp(
                kp,
                "/query",
                method="POST",
                json_body=trapi_body,
                timeout_s=float(timeout_s),
            )
            res.setdefault("meta", {})
            res["meta"].update({"kp_id": kp_id, "kp_type": "trapi"})
            self._validate_trapi_response(res)
            self._attach_refresh_errors(res, fresh)
            return res

        # BioThings
        if isinstance(q, str):
            res = self._call_kp(
                kp,
                "/query",
                method="GET",
                params={"q": q},
                timeout_s=float(timeout_s),
            )
        elif isinstance(q, dict):
            res = self._call_kp(
                kp,
                "/query",
                method="POST",
                json_body=q,
                timeout_s=float(timeout_s),
            )
        else:
            return self._wrap(
                False,
                None,
                [{"type": "bad_query", "message": "q must be str or dict"}],
                meta={"kp_id": kp_id},
            )

        res.setdefault("meta", {})
        res["meta"].update({"kp_id": kp_id, "kp_type": "biothings", "query": q})
        self._maybe_normalize_biothings(res, normalize, include_raw_hits)
        self._attach_refresh_errors(res, fresh)
        return res

    def query_batch(
        self,
        kp_id: str,
        queries: Any,
        *,
        normalize: bool = False,
        include_raw_hits: bool = False,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        kp, fresh, err = self._get_kp(kp_id)
        if err:
            return err

        is_trapi = self._is_trapi_kp(kp)

        if is_trapi:
            if isinstance(queries, dict):
                return self.query(
                    kp_id=kp_id,
                    q=queries,
                    normalize=normalize,
                    include_raw_hits=include_raw_hits,
                    timeout_s=timeout_s,
                )
            if not isinstance(queries, list):
                return self._wrap(
                    False,
                    None,
                    [
                        {
                            "type": "bad_queries",
                            "message": "TRAPI queries must be dict or list",
                        }
                    ],
                    meta={"kp_id": kp_id},
                )

            out: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            for i, item in enumerate(queries):
                body = item if isinstance(item, dict) else {"message": item}
                r = self._call_kp(
                    kp,
                    "/query",
                    method="POST",
                    json_body=body,
                    timeout_s=float(timeout_s),
                )
                r.setdefault("meta", {})
                r["meta"].update({"kp_id": kp_id, "kp_type": "trapi"})
                self._validate_trapi_response(r)
                out.append(r)
                if not r.get("ok"):
                    errors.append(
                        {
                            "type": "trapi_item_failed",
                            "message": f"item {i} failed",
                            "details": {
                                "index": i,
                                "errors": r.get("errors"),
                            },
                        }
                    )

            res = self._wrap(
                ok=(len(errors) == 0),
                data=out,
                errors=errors,
                meta={
                    "kp_id": kp_id,
                    "kp_type": "trapi",
                    "count": len(out),
                },
            )
            self._attach_refresh_errors(res, fresh)
            return res

        # BioThings batch
        if isinstance(queries, list):
            payload: dict[str, Any] = {"q": queries}
        elif isinstance(queries, dict):
            payload = dict(queries)
        else:
            return self._wrap(
                False,
                None,
                [
                    {
                        "type": "bad_queries",
                        "message": "queries must be list or dict",
                    }
                ],
                meta={"kp_id": kp_id},
            )

        res = self._call_kp(
            kp,
            "/querymany",
            method="POST",
            json_body=payload,
            timeout_s=float(timeout_s),
        )
        status = (res.get("meta") or {}).get("status_code")
        if not res.get("ok") and status in (404, 405):
            res = self._call_kp(
                kp,
                "/query",
                method="POST",
                json_body=payload,
                timeout_s=float(timeout_s),
            )

        res.setdefault("meta", {})
        res["meta"].update({"kp_id": kp_id, "kp_type": "biothings", "queries": queries})
        self._maybe_normalize_biothings(res, normalize, include_raw_hits)
        self._attach_refresh_errors(res, fresh)
        return res

    def scroll(
        self,
        kp_id: str,
        *,
        q: str | None = None,
        scroll_id: str | None = None,
        size: int = 100,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        kp, fresh, err = self._get_kp(kp_id)
        if err:
            return err

        if self._is_trapi_kp(kp):
            return self._wrap(
                False,
                None,
                [
                    {
                        "type": "not_supported",
                        "message": "scroll not supported for TRAPI KPs",
                    }
                ],
                meta={"kp_id": kp_id},
            )

        params: dict[str, Any] = {"scroll_id": scroll_id, "size": int(size)}
        if q is not None:
            params["q"] = q

        res = self._call_kp(
            kp,
            "/query",
            method="GET",
            params=params,
            timeout_s=float(timeout_s),
        )
        res.setdefault("meta", {})
        res["meta"].update(
            {
                "kp_id": kp_id,
                "kp_type": "biothings",
                "q": q,
                "scroll_id": scroll_id,
                "size": size,
            }
        )
        self._attach_refresh_errors(res, fresh)
        return res
