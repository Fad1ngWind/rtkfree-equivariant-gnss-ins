"""Fail-closed guards for deployable/training information paths and configs.

Phase 0 deliberately provides no sealed-reference override or loader.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence, Set
from pathlib import Path
from typing import Any


class InformationPolicyError(ValueError):
    """Raised when a path or configuration violates the frozen information policy."""


_FORBIDDEN_COMPACT = {
    "rtk", "rtkresult", "rtkresults", "rtksolution", "rtksolutions", "rtktrajectory",
    "ppk", "ppkresult", "ppkresults", "ppksolution", "ppksolutions",
    "groundtruth", "groundtruthpath", "groundtruthtrajectory", "truthtrajectory",
    "highprecision", "highprecisionpath", "highprecisiontrajectory",
    "postprocessedtruth", "postprocessedtrajectory", "referencetrajectory",
    "sealedreference", "sealedtruth",
}

_APPROVED_RTKFREE_COMPONENTS = {
    "rtkfree",
    "rtkfreeequivariantgnssins",
}


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _matches_forbidden(text: str) -> str | None:
    normalized = text.replace("\\", "/")
    for raw_component in normalized.split("/"):
        component = _compact(raw_component)
        if not component:
            continue
        if ("rtk" in component or "ppk" in component) and component not in _APPROVED_RTKFREE_COMPONENTS:
            return raw_component
        if component in _FORBIDDEN_COMPACT:
            return raw_component
        if any(marker in component for marker in (
            "groundtruth", "highprecision", "postprocessedtruth",
            "referencetrajectory", "sealedreference",
        )):
            return raw_component
    return None


def _symlink_in_chain(candidate: Path) -> Path | None:
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    current = Path(absolute.anchor)
    for part in absolute.parts[1:] if absolute.anchor else absolute.parts:
        current = current / part
        try:
            if current.is_symlink():
                return current
        except OSError as exc:
            raise InformationPolicyError(
                f"cannot inspect deployable path component {current}: {type(exc).__name__}"
            ) from exc
    return None


def validate_deployable_path(path: str | os.PathLike[str]) -> Path:
    """Validate raw and resolved forms and reject any symbolic-link chain."""

    try:
        raw = os.fspath(path)
    except TypeError as exc:
        raise InformationPolicyError("deployable path must be str or PathLike[str]") from exc
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise InformationPolicyError("deployable path must be a non-empty text path without NUL")

    raw_match = _matches_forbidden(raw)
    if raw_match:
        raise InformationPolicyError(
            f"raw deployable path contains forbidden high-precision marker: {raw_match}"
        )

    candidate = Path(raw)
    link = _symlink_in_chain(candidate)
    if link is not None:
        raise InformationPolicyError(f"symbolic links are forbidden in deployable path chain: {link}")

    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise InformationPolicyError(
            f"cannot resolve deployable path: {type(exc).__name__}"
        ) from exc
    resolved_match = _matches_forbidden(resolved.as_posix())
    if resolved_match:
        raise InformationPolicyError(
            f"resolved deployable path contains forbidden high-precision marker: {resolved_match}"
        )
    return candidate


def _walk_config(value: Any, location: str, seen: set[int]) -> None:
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise InformationPolicyError(f"cyclic configuration container at {location}")
        seen.add(identity)
        try:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise InformationPolicyError(
                        f"configuration key at {location} must be text, got {type(key).__name__}"
                    )
                match = _matches_forbidden(key)
                if match:
                    raise InformationPolicyError(
                        f"forbidden configuration key at {location}.{key}: {match}"
                    )
                _walk_config(child, f"{location}.{key}", seen)
        finally:
            seen.remove(identity)
        return

    if isinstance(value, (Sequence, Set)) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in seen:
            raise InformationPolicyError(f"cyclic configuration container at {location}")
        seen.add(identity)
        try:
            for index, child in enumerate(value):
                _walk_config(child, f"{location}[{index}]", seen)
        finally:
            seen.remove(identity)
        return

    if isinstance(value, os.PathLike):
        validate_deployable_path(value)
        return
    if isinstance(value, str):
        match = _matches_forbidden(value)
        if match:
            raise InformationPolicyError(f"forbidden configuration value at {location}: {match}")
        return
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InformationPolicyError(f"non-finite numeric value at {location}")
        return
    raise InformationPolicyError(
        f"unsupported configuration leaf at {location}: {type(value).__name__}"
    )


def validate_deployable_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively reject forbidden markers and unsupported configuration values."""

    if not isinstance(config, Mapping):
        raise InformationPolicyError("top-level configuration must be a mapping")
    _walk_config(config, "$", set())
    return config


def load_deployable_json(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    """Load JSON only after validating the path and every decoded value."""

    checked = validate_deployable_path(path)
    try:
        with checked.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InformationPolicyError(
            f"cannot load deployable JSON: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, Mapping):
        raise InformationPolicyError("top-level configuration must be an object")
    return validate_deployable_config(value)
