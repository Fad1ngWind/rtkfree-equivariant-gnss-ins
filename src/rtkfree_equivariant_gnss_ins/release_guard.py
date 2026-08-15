"""Fail-closed scanner for public repository candidates and staged Git blobs."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024
REGULAR_GIT_MODES = {"100644", "100755"}
BLOCKED_GIT_MODES = {
    "120000": "symbolic link",
    "160000": "gitlink/submodule",
}

FORBIDDEN_SUFFIXES = {
    # Navigation, trajectory, tabular, and scientific data.
    ".rnx", ".crx", ".obs", ".nav", ".sp3", ".clk", ".pos", ".ubx",
    ".hdf", ".h5", ".hdf5", ".mat", ".npy", ".npz", ".parquet",
    ".feather", ".csv", ".tsv", ".bag", ".db3", ".mcap",
    # Weights and generic serialization.
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pkl", ".pickle",
    ".joblib", ".pb", ".tflite", ".pyc", ".pyo", ".pyd",
    # Secrets, logs, backups, and archives/compressed RINEX containers.
    ".pem", ".key", ".p12", ".pfx", ".log", ".bak", ".backup", ".orig",
    ".zip", ".7z", ".tar", ".tgz", ".gz", ".z", ".bz2", ".xz", ".rar",
}

_RINEX_2_SUFFIX = re.compile(r"\.[0-9]{2}[odnmg]$", re.IGNORECASE)
_RINEX_HEADER = re.compile(b"RINEX VERSION" + b" / TYPE", re.IGNORECASE)
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

_FORBIDDEN_COMPONENTS: dict[str, set[str]] = {
    "environment/cache": {
        "venv", "env", "virtualenv", "sitepackages", "pycache", "pytestcache",
        "mypycached", "mypycache", "ruffcache", "tox", "nox",
        "ipynbcheckpoints", "jupytercache", "egginfo",
    },
    "formal data": {
        "data", "dataset", "datasets", "rawdata", "formaldata", "navigationdata",
        "rinex", "observations", "trajectorydata",
    },
    "weights/results/runtime": {
        "output", "outputs", "artifact", "artifacts", "checkpoint", "checkpoints",
        "weight", "weights", "log", "logs", "run", "runs", "wandb", "mlruns",
        "tensorboard", "result", "results",
    },
    "secret/private/unreviewed": {
        "secret", "secrets", "credential", "credentials", "private", "confidential",
        "advisorprivate", "aiconversation", "aiconversations", "chat", "chats",
        "chatexport", "chatexports", "privatechat", "privatechats", "unreviewed",
        "unreviewednotes", "privatenote", "privatenotes", "aitalkrecords",
    },
    "backup/archive/legacy": {
        "backup", "backups", "archive", "archives", "legacysnapshot",
        "legacysnapshots",
    },
}

_HIGH_PRECISION_COMPACT = {
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

SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "OpenAI-style token": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
}


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str


@dataclass(frozen=True)
class CandidateEntry:
    path: str
    mode: str


def _compact_component(component: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", component.casefold())


def _validate_repo_relative_path(path: str) -> str | None:
    """Return an error for anything other than an exact Git-style relative path."""

    if not isinstance(path, str) or not path:
        return "path must be a non-empty string"
    if "\x00" in path:
        return "path contains NUL"
    if "\\" in path:
        return "path must use Git forward-slash separators"
    if path.startswith("/") or path.startswith("//") or re.match(r"^[A-Za-z]:", path):
        return "absolute path is not a repository candidate"
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return "path contains empty, dot, or traversal component"
    if PurePosixPath(path).is_absolute():
        return "absolute path is not a repository candidate"
    return None


def _path_category_findings(path: str) -> list[Finding]:
    pure = PurePosixPath(path)
    findings: list[Finding] = []
    compact_parts = [_compact_component(part) for part in pure.parts]

    for category, forbidden in _FORBIDDEN_COMPONENTS.items():
        match = next((part for part in compact_parts if part in forbidden), None)
        if match:
            findings.append(Finding(path, f"forbidden {category} path component: {match}"))

    precision_match = next(
        (
            part
            for part in compact_parts
            if (("rtk" in part or "ppk" in part) and part not in _APPROVED_RTKFREE_COMPONENTS)
            or part in _HIGH_PRECISION_COMPACT
            or "groundtruth" in part
            or "highprecision" in part
            or "postprocessedtruth" in part
            or "referencetrajectory" in part
            or "sealedreference" in part
        ),
        None,
    )
    if precision_match:
        findings.append(Finding(path, f"high-precision/reference path marker: {precision_match}"))

    suffixes = {suffix.casefold() for suffix in pure.suffixes}
    forbidden_suffixes = sorted(suffixes & FORBIDDEN_SUFFIXES)
    if forbidden_suffixes:
        findings.append(Finding(path, f"forbidden file type: {forbidden_suffixes[-1]}"))
    if _RINEX_2_SUFFIX.search(pure.name):
        findings.append(Finding(path, "classic RINEX filename"))

    name_folded = pure.name.casefold()
    if (name_folded == ".env" or name_folded.startswith(".env.")) and pure.name != ".env.example":
        findings.append(Finding(path, "local environment/secrets file"))
    if name_folded in {".npmrc", ".pypirc", ".netrc", "credentials.json"}:
        findings.append(Finding(path, "credential-bearing configuration filename"))
    return findings


def inspect_candidate(path: str, content: bytes, mode: str = "100644") -> list[Finding]:
    """Inspect one exact repository-relative candidate without path normalization."""

    path_error = _validate_repo_relative_path(path)
    if path_error:
        return [Finding(str(path), path_error)]
    findings = _path_category_findings(path)

    if mode in BLOCKED_GIT_MODES:
        findings.append(Finding(path, f"forbidden Git mode {mode}: {BLOCKED_GIT_MODES[mode]}"))
    elif mode not in REGULAR_GIT_MODES:
        findings.append(Finding(path, f"unsupported Git mode: {mode or '<missing>'}"))

    if not isinstance(content, bytes):
        findings.append(Finding(path, "candidate content is not bytes"))
        return findings
    if len(content) > MAX_PUBLIC_FILE_BYTES:
        findings.append(Finding(path, f"file exceeds {MAX_PUBLIC_FILE_BYTES} bytes"))
    if content.startswith(_HDF5_MAGIC):
        findings.append(Finding(path, "HDF5 content signature"))
    if _RINEX_HEADER.search(content[:4096]):
        findings.append(Finding(path, "RINEX text header"))
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(content):
            findings.append(Finding(path, f"possible {label}"))
    return findings


def _run_git(repo: Path, args: list[str]) -> bytes:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True
    )
    return completed.stdout


def _index_modes(repo: Path) -> dict[str, str]:
    modes: dict[str, str] = {}
    raw = _run_git(repo, ["ls-files", "--stage", "-z"])
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            mode, _object_id, stage = metadata.decode("ascii").split()
            path = encoded_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if stage == "0":
            modes[path] = mode
    return modes


def _git_entries(repo: Path, staged: bool) -> list[CandidateEntry]:
    modes = _index_modes(repo)
    if staged:
        raw = _run_git(
            repo,
            ["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMRT"],
        )
    else:
        raw = _run_git(repo, ["ls-files", "-co", "--exclude-standard", "-z"])
    entries: list[CandidateEntry] = []
    for encoded_path in raw.split(b"\0"):
        if not encoded_path:
            continue
        path = encoded_path.decode("utf-8")
        mode = modes.get(path)
        if mode is None and not staged:
            candidate = repo / path
            mode = "120000" if candidate.is_symlink() else "100644"
        entries.append(CandidateEntry(path=path, mode=mode or ""))
    return entries


def _ignored_paths(repo: Path, paths: list[str]) -> set[str]:
    """Return ignored candidates using one NUL-safe Git query, including tracked paths."""

    if not paths:
        return set()
    encoded = b"\0".join(path.encode("utf-8") for path in paths) + b"\0"
    completed = subprocess.run(
        ["git", "check-ignore", "--no-index", "-z", "--stdin"],
        cwd=repo,
        input=encoded,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    try:
        return {
            item.decode("utf-8")
            for item in completed.stdout.split(b"\0")
            if item
        }
    except UnicodeDecodeError:
        raise


def _read_content(repo: Path, entry: CandidateEntry, staged: bool) -> bytes:
    if staged:
        return _run_git(repo, ["show", f":{entry.path}"])
    candidate = repo / entry.path
    if candidate.is_symlink():
        return os.readlink(candidate).encode("utf-8", errors="strict")
    return candidate.read_bytes()


def scan_repository(repo: Path, staged: bool) -> list[Finding]:
    """Scan all candidates; any Git enumeration or decoding failure fails closed."""

    findings: list[Finding] = []
    try:
        entries = _git_entries(repo, staged)
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError, ValueError) as exc:
        return [Finding("<repository>", f"cannot enumerate candidates: {type(exc).__name__}")]

    try:
        ignored = _ignored_paths(repo, [entry.path for entry in entries])
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError, ValueError) as exc:
        return [Finding("<repository>", f"cannot evaluate ignore policy: {type(exc).__name__}")]

    for entry in entries:
        path_error = _validate_repo_relative_path(entry.path)
        if path_error:
            findings.append(Finding(entry.path, path_error))
            continue
        if entry.path in ignored:
            findings.append(Finding(entry.path, "candidate is forbidden by repository .gitignore policy"))
        if entry.mode in BLOCKED_GIT_MODES:
            findings.append(
                Finding(entry.path, f"forbidden Git mode {entry.mode}: {BLOCKED_GIT_MODES[entry.mode]}")
            )
            continue
        if entry.mode not in REGULAR_GIT_MODES:
            findings.append(Finding(entry.path, f"unsupported Git mode: {entry.mode or '<missing>'}"))
            continue
        try:
            content = _read_content(repo, entry, staged)
        except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
            findings.append(Finding(entry.path, f"cannot inspect candidate: {type(exc).__name__}"))
            continue
        findings.extend(inspect_candidate(entry.path, content, mode=entry.mode))
    return findings


def format_findings(findings: Iterable[Finding]) -> str:
    return "\n".join(f"BLOCK {item.path}: {item.reason}" for item in findings)
