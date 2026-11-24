"""File storage for uploads, model artifacts and prediction outputs, behind a small interface
so a cloud object store can replace the local disk later without touching callers."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import BinaryIO, Protocol


class FileStore(Protocol):
    def save_stream(self, area: str, name: str, stream: BinaryIO, max_bytes: int) -> tuple[Path, int, str]: ...
    def path(self, area: str, name: str) -> Path: ...
    def delete(self, area: str, name: str) -> None: ...


class FileTooLargeError(Exception):
    pass


class LocalFileStore:
    AREAS = ("uploads", "artifacts", "outputs", "datasets")

    def __init__(self, root: Path) -> None:
        self.root = root
        for a in self.AREAS:
            (root / a).mkdir(parents=True, exist_ok=True)

    def path(self, area: str, name: str) -> Path:
        if area not in self.AREAS:
            raise ValueError(f"unknown storage area {area}")
        p = (self.root / area / name).resolve()
        if self.root.resolve() not in p.parents:  # never escape the storage root
            raise ValueError("invalid storage path")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_stream(self, area: str, name: str, stream: BinaryIO, max_bytes: int) -> tuple[Path, int, str]:
        """Copy a stream to disk in chunks, enforcing the size limit. Returns (path, size, sha256)."""
        dest = self.path(area, name)
        digest = hashlib.sha256()
        size = 0
        with open(dest, "wb") as out:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise FileTooLargeError()
                digest.update(chunk)
                out.write(chunk)
        return dest, size, digest.hexdigest()

    def delete(self, area: str, name: str) -> None:
        p = self.path(area, name)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
