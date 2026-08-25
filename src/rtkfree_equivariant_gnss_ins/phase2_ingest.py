"""Minimal Phase 2 acquisition and positive-list ZIP extraction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit


class Phase2IngestError(RuntimeError):
    """Raised when selected deployable data fails a minimal integrity check."""


_FLOAT_TOKEN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise Phase2IngestError(f"cannot hash file: {type(exc).__name__}") from exc
    return size, digest.hexdigest()


def _safe_selected_member(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise Phase2IngestError("selected ZIP member has an invalid name")
    member = PurePosixPath(name)
    if member.is_absolute() or any(part in {"", ".", ".."} for part in member.parts):
        raise Phase2IngestError(f"selected ZIP member has an unsafe path: {name}")
    if any(":" in part for part in member.parts):
        raise Phase2IngestError(f"selected ZIP member has a drive-like path: {name}")
    return member


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)


def locate_selected_members(
    archive_path: Path, selected_basenames: set[str]
) -> dict[str, str]:
    """Locate only positive-listed basenames without opening any other member."""

    wanted = {name.casefold(): name for name in selected_basenames}
    found: dict[str, str] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                basename = PurePosixPath(info.filename).name.casefold()
                if basename not in wanted:
                    continue
                member = _safe_selected_member(info.filename)
                if info.is_dir() or _is_symlink(info):
                    raise Phase2IngestError(
                        f"selected ZIP member is not a regular file: {info.filename}"
                    )
                if basename in found:
                    raise Phase2IngestError(
                        f"selected ZIP basename occurs more than once: {member.name}"
                    )
                found[basename] = info.filename
    except (OSError, zipfile.BadZipFile) as exc:
        raise Phase2IngestError(f"cannot inspect ZIP: {type(exc).__name__}") from exc
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise Phase2IngestError(f"selected ZIP members are missing: {', '.join(missing)}")
    return found


def _anonymous_https(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise Phase2IngestError("source URL must be anonymous HTTPS")


def _download_once(url: str, destination: Path, max_bytes: int = 2_000_000_000) -> None:
    _anonymous_https(url)
    request = urllib.request.Request(url, headers={"User-Agent": "rtkfree-phase2/1"})
    if destination.exists():
        raise Phase2IngestError(f"refusing to overwrite: {destination}")
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            destination.open("xb") as out,
        ):
            _anonymous_https(response.geturl())
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise Phase2IngestError("download exceeds safety size limit")
                out.write(chunk)
    except Phase2IngestError:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise Phase2IngestError(f"download failed: {type(exc).__name__}") from exc


def _load_lock(path: Path) -> dict[str, Any]:
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase2IngestError(f"cannot load source manifest: {type(exc).__name__}") from exc
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise Phase2IngestError("unsupported source manifest")
    return lock


def _write_selected_extrinsic(
    source_path: Path,
    destination: Path,
    artifact: dict[str, Any],
) -> tuple[int, str]:
    """Write only the GNSS-antenna/IMU transform from a mixed official file."""

    source_size, source_digest = sha256_file(source_path)
    if (
        source_size != artifact.get("observed_byte_size")
        or source_digest != artifact.get("observed_sha256")
    ):
        raise Phase2IngestError("extrinsic source differs from recorded acquisition")
    try:
        text = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise Phase2IngestError("cannot read extrinsic source") from exc
    selected_key = artifact.get("selected_key")
    if not isinstance(selected_key, str):
        raise Phase2IngestError("extrinsic source lacks a selected key")
    key_match = re.search(rf"(?m)^{re.escape(selected_key)}\s*:", text)
    if key_match is None:
        raise Phase2IngestError("selected extrinsic key is missing")
    selected_section = text[key_match.end() :]
    next_key = re.search(r"(?m)^[A-Za-z][A-Za-z0-9_]*\s*:", selected_section)
    if next_key is not None:
        selected_section = selected_section[: next_key.start()]
    data_match = re.search(r"\bdata\s*:\s*\[([^\]]+)\]", selected_section)
    if data_match is None:
        raise Phase2IngestError("selected extrinsic matrix data is missing")
    values = [float(token) for token in _FLOAT_TOKEN.findall(data_match.group(1))]
    if len(values) != 16:
        raise Phase2IngestError("selected extrinsic matrix is not 4 by 4")
    record = {
        "schema_version": 1,
        "source_id": artifact["id"],
        "source_sha256": source_digest,
        "selected_key": selected_key,
        "direction_as_labeled_by_source": "GNSS antennas to IMU",
        "body_frame": "IMU",
        "body_axes_as_declared_by_source": {
            "x": "right",
            "y": "forward",
            "z": "up",
        },
        "translation_unit": "m",
        "unit_basis": "official-file internal ROS transform examples + ROS REP-103",
        "unit_status": (
            "evidence-backed interpretation; not explicitly annotated beside "
            "ANTENNA_T_IMU"
        ),
        "phase2_use": "not applied; WLS/SPP remains at the GNSS antenna phase center",
        "matrix_4x4_row_major": [values[index : index + 4] for index in range(0, 16, 4)],
    }
    if destination.exists():
        raise Phase2IngestError("refusing to overwrite selected extrinsic")
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        raise Phase2IngestError("cannot write selected extrinsic") from exc
    destination.chmod(0o600)
    return sha256_file(destination)


def refresh_selected_medium_extrinsic(
    source_lock_path: Path,
    data_root: Path,
    preserved_staging_source: Path,
) -> tuple[Path, int, str, Path]:
    """Refresh the bounded selected record after an evidence interpretation changes."""

    lock = _load_lock(source_lock_path)
    artifacts = {item["id"]: item for item in lock.get("artifacts", [])}
    artifact = artifacts.get("extrinsic_calibration")
    if not isinstance(artifact, dict):
        raise Phase2IngestError("extrinsic source is missing")
    expected_source = (
        artifact.get("observed_byte_size"),
        artifact.get("observed_sha256"),
    )
    if sha256_file(preserved_staging_source) != expected_source:
        raise Phase2IngestError("preserved staging source differs from acquisition")

    session_root = (
        data_root
        / "deployable"
        / lock["dataset"]
        / lock["repository_commit"]
        / lock["session"]
    )
    selected_path = session_root / "calibration" / artifact["selected_output_name"]
    expected_selected = (
        artifact.get("selected_output_byte_size"),
        artifact.get("selected_output_sha256"),
    )
    if selected_path.is_symlink() or sha256_file(selected_path) != expected_selected:
        raise Phase2IngestError("selected extrinsic differs from the source record")

    manifest_path = session_root / "provenance_manifest.json"
    manifest = _load_lock(manifest_path)
    selected_relative = selected_path.relative_to(session_root).as_posix()
    selected_entries = [
        item
        for item in manifest["selected_outputs"]
        if item.get("path") == selected_relative
    ]
    if len(selected_entries) != 1 or (
        selected_entries[0].get("size"), selected_entries[0].get("sha256")
    ) != expected_selected:
        raise Phase2IngestError("unified manifest does not identify the selected extrinsic")

    temporary_selected = selected_path.with_suffix(".json.tmp")
    temporary_selected.unlink(missing_ok=True)
    selected_size, selected_digest = _write_selected_extrinsic(
        preserved_staging_source, temporary_selected, artifact
    )
    selected_entries[0].update(
        {
            "size": selected_size,
            "sha256": selected_digest,
            "unit_interpretation": "m",
        }
    )
    temporary_manifest = session_root / ".provenance_manifest.json.tmp"
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_selected, selected_path)
        os.replace(temporary_manifest, manifest_path)
    except OSError as exc:
        temporary_selected.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise Phase2IngestError("cannot refresh selected extrinsic record") from exc
    return selected_path, selected_size, selected_digest, manifest_path


def prepare_medium(
    source_lock_path: Path,
    data_root: Path,
    existing_gnss_zip: Path,
    existing_direct_dir: Path | None = None,
) -> Path:
    """Reuse one verified ZIP, acquire/reuse direct files, and write one manifest."""

    if not data_root.is_absolute() or not existing_gnss_zip.is_absolute():
        raise Phase2IngestError("data root and existing ZIP must be absolute paths")
    data_root = data_root.resolve(strict=True)
    repository_root = source_lock_path.resolve(strict=True).parents[2]
    if data_root == repository_root or repository_root in data_root.parents:
        raise Phase2IngestError("data root must be outside the source repository")
    if existing_gnss_zip.is_symlink():
        raise Phase2IngestError("existing GNSS ZIP cannot be a symbolic link")
    existing_gnss_zip = existing_gnss_zip.resolve(strict=True)

    lock = _load_lock(source_lock_path)
    artifacts = {item["id"]: item for item in lock.get("artifacts", [])}
    gnss = artifacts.get("gnss_observation_archive")
    if not isinstance(gnss, dict):
        raise Phase2IngestError("GNSS observation archive is missing from source manifest")
    zip_size, zip_sha256 = sha256_file(existing_gnss_zip)
    if (
        zip_size != gnss.get("observed_byte_size")
        or zip_sha256 != gnss.get("observed_sha256")
    ):
        raise Phase2IngestError("existing GNSS ZIP differs from recorded acquisition")

    selected = set(gnss.get("selected_member_basenames", []))
    located = locate_selected_members(existing_gnss_zip, selected)
    commit = lock.get("repository_commit")
    session = lock.get("session")
    if not isinstance(commit, str) or not isinstance(session, str):
        raise Phase2IngestError("source manifest lacks version or session")
    destination_parent = data_root / "deployable" / lock["dataset"] / commit
    destination = destination_parent / session
    temporary = destination_parent / f".{session}.tmp"
    if destination.exists() or temporary.exists():
        raise Phase2IngestError("refusing to overwrite prepared data")
    temporary.mkdir(parents=True, mode=0o700)

    sources: list[dict[str, Any]] = [
        {
            "id": gnss["id"],
            "source": gnss["source_url"],
            "retrieved_at": gnss["retrieved_at"],
            "size": zip_size,
            "sha256": zip_sha256,
        }
    ]
    outputs: list[dict[str, Any]] = []
    gnss_dir = temporary / "gnss"
    gnss_dir.mkdir(mode=0o700)
    with zipfile.ZipFile(existing_gnss_zip) as archive:
        for member_name in sorted(located.values()):
            member = _safe_selected_member(member_name)
            target = gnss_dir / member.name
            with archive.open(member_name) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            target.chmod(0o600)
            size, digest = sha256_file(target)
            outputs.append(
                {
                    "path": target.relative_to(temporary).as_posix(),
                    "size": size,
                    "sha256": digest,
                }
            )

    for artifact_id in ("low_cost_imu", "extrinsic_calibration", "imu_calibration"):
        artifact = artifacts.get(artifact_id)
        if not isinstance(artifact, dict):
            raise Phase2IngestError(f"required source is missing: {artifact_id}")
        local_name = artifact["local_name"]
        reused = existing_direct_dir / local_name if existing_direct_dir is not None else None
        source_file = temporary / f".{local_name}.source"
        if reused is not None and reused.is_file():
            shutil.copyfile(reused, source_file)
        else:
            _download_once(artifact["source_url"], source_file)
        size, digest = sha256_file(source_file)
        if (
            size != artifact.get("observed_byte_size")
            or digest != artifact.get("observed_sha256")
        ):
            raise Phase2IngestError(f"source bytes differ from recorded acquisition: {artifact_id}")
        category = "imu" if artifact_id == "low_cost_imu" else "calibration"
        target_dir = temporary / category
        target_dir.mkdir(exist_ok=True, mode=0o700)
        if artifact_id == "extrinsic_calibration":
            target = target_dir / artifact["selected_output_name"]
            output_size, output_digest = _write_selected_extrinsic(
                source_file, target, artifact
            )
            source_file.unlink()
        else:
            target = target_dir / local_name
            os.replace(source_file, target)
            target.chmod(0o600)
            output_size, output_digest = size, digest
        sources.append(
            {
                "id": artifact_id,
                "source": artifact["source_url"],
                "retrieved_at": artifact["retrieved_at"],
                "size": size,
                "sha256": digest,
            }
        )
        outputs.append(
            {
                "path": target.relative_to(temporary).as_posix(),
                "size": output_size,
                "sha256": output_digest,
                **(
                    {
                        "derived_from_source_id": artifact_id,
                        "selected_key": artifact["selected_key"],
                    }
                    if artifact_id == "extrinsic_calibration"
                    else {}
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset": lock["dataset"],
        "session": session,
        "repository_commit": commit,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_status": gnss["selection_status"],
        "sources": sorted(sources, key=lambda item: item["id"]),
        "selected_outputs": sorted(outputs, key=lambda item: item["path"]),
    }
    manifest_path = temporary / "provenance_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination / "provenance_manifest.json"


def acquire_broadcast_navigation(
    source_lock_path: Path,
    data_root: Path,
) -> tuple[Path, int, str, Path]:
    """Fetch the one recorded broadcast-navigation file and extend the manifest."""

    if not data_root.is_absolute():
        raise Phase2IngestError("data root must be an absolute path")
    data_root = data_root.resolve(strict=True)
    repository_root = source_lock_path.resolve(strict=True).parents[2]
    if data_root == repository_root or repository_root in data_root.parents:
        raise Phase2IngestError("data root must be outside the source repository")

    lock = _load_lock(source_lock_path)
    artifacts = {item["id"]: item for item in lock.get("artifacts", [])}
    navigation = artifacts.get("broadcast_navigation")
    if not isinstance(navigation, dict):
        raise Phase2IngestError("broadcast-navigation source is missing")

    destination = (
        data_root
        / "deployable"
        / lock["dataset"]
        / lock["repository_commit"]
        / lock["session"]
    )
    manifest_path = destination / "provenance_manifest.json"
    manifest = _load_lock(manifest_path)
    identity = ("dataset", "session", "repository_commit")
    if any(manifest.get(field) != lock.get(field) for field in identity):
        raise Phase2IngestError("provenance manifest does not match the source record")

    if any(item.get("id") == navigation["id"] for item in manifest["sources"]):
        raise Phase2IngestError("broadcast navigation is already in the provenance manifest")

    target = destination / "gnss" / navigation["local_name"]
    relative_path = target.relative_to(destination).as_posix()
    if any(item.get("path") == relative_path for item in manifest["selected_outputs"]):
        raise Phase2IngestError("broadcast navigation output is already recorded")
    if target.exists():
        raise Phase2IngestError("refusing to overwrite broadcast navigation")
    _download_once(navigation["source_url"], target)
    target.chmod(0o600)
    size, digest = sha256_file(target)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    manifest["sources"].append(
        {
            "id": navigation["id"],
            "source": navigation["source_url"],
            "retrieved_at": retrieved_at,
            "size": size,
            "sha256": digest,
        }
    )
    manifest["sources"].sort(key=lambda item: item["id"])
    manifest["selected_outputs"].append(
        {"path": relative_path, "size": size, "sha256": digest}
    )
    manifest["selected_outputs"].sort(key=lambda item: item["path"])

    temporary_manifest = destination / ".provenance_manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return target, size, digest, manifest_path


def restrict_existing_medium_extrinsic(
    source_lock_path: Path,
    data_root: Path,
    preserved_staging_source: Path,
) -> tuple[Path, int, str, Path]:
    """Replace one existing deployable mixed file with its selected safe record."""

    if not data_root.is_absolute() or not preserved_staging_source.is_absolute():
        raise Phase2IngestError("data root and staging source must be absolute paths")
    data_root = data_root.resolve(strict=True)
    source_lock_path = source_lock_path.resolve(strict=True)
    repository_root = source_lock_path.parents[2]
    if data_root == repository_root or repository_root in data_root.parents:
        raise Phase2IngestError("data root must be outside the source repository")
    if preserved_staging_source.is_symlink():
        raise Phase2IngestError("staging source cannot be a symbolic link")
    preserved_staging_source = preserved_staging_source.resolve(strict=True)

    lock = _load_lock(source_lock_path)
    artifacts = {item["id"]: item for item in lock.get("artifacts", [])}
    artifact = artifacts.get("extrinsic_calibration")
    if not isinstance(artifact, dict):
        raise Phase2IngestError("extrinsic source is missing")
    expected = (
        artifact.get("observed_byte_size"),
        artifact.get("observed_sha256"),
    )
    if sha256_file(preserved_staging_source) != expected:
        raise Phase2IngestError("preserved staging source differs from acquisition")

    session_root = (
        data_root
        / "deployable"
        / lock["dataset"]
        / lock["repository_commit"]
        / lock["session"]
    )
    raw_path = session_root / "calibration" / artifact["local_name"]
    if raw_path.is_symlink() or sha256_file(raw_path) != expected:
        raise Phase2IngestError("deployable mixed extrinsic differs from acquisition")
    selected_path = session_root / "calibration" / artifact["selected_output_name"]
    selected_size, selected_digest = _write_selected_extrinsic(
        raw_path, selected_path, artifact
    )

    manifest_path = session_root / "provenance_manifest.json"
    manifest = _load_lock(manifest_path)
    raw_relative = raw_path.relative_to(session_root).as_posix()
    raw_entries = [
        item for item in manifest["selected_outputs"] if item.get("path") == raw_relative
    ]
    if len(raw_entries) != 1 or raw_entries[0].get("sha256") != expected[1]:
        selected_path.unlink(missing_ok=True)
        raise Phase2IngestError("unified manifest does not identify the mixed extrinsic")
    manifest["selected_outputs"] = [
        item for item in manifest["selected_outputs"] if item.get("path") != raw_relative
    ]
    manifest["selected_outputs"].append(
        {
            "path": selected_path.relative_to(session_root).as_posix(),
            "size": selected_size,
            "sha256": selected_digest,
            "derived_from_source_id": artifact["id"],
            "selected_key": artifact["selected_key"],
        }
    )
    manifest["selected_outputs"].sort(key=lambda item: item["path"])
    temporary_manifest = session_root / ".provenance_manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    raw_path.unlink()
    return selected_path, selected_size, selected_digest, manifest_path
