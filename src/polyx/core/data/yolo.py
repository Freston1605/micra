"""The YAML box dataset declaration: one owner for ``data.yaml``.

It answers the three questions every consumer used to answer for itself: where
the dataset root is, where each split lives, and which class names it declares.
Labels, class-id policy and hashing stay with their consumers: the codec owns
the ``.txt`` format, the headless resolver owns the "consecutive ids from zero"
rule, and the fingerprint owns its keys.

A declaration is read without walking the folders. ``SplitDeclaration.images``
does the walk or reads the list on demand, so an unreadable split is reported by
the consumer that can turn it into a finding instead of aborting the load.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from polyx.core.errors import CodedFailure
from polyx.core.i18n import N_
from polyx.core.paths import find_images

#: The split keys a YOLO declaration may carry. A split that is not declared is
#: absent, not empty: ``DatasetDeclaration.split`` returns ``None`` for it.
SPLIT_NAMES = ("train", "val", "test")


class SplitKind(StrEnum):
    """How a declared split is written on disk."""

    DIRECTORY = "directory"
    """An image folder, walked recursively."""

    LIST = "list"
    """A ``.txt`` file holding image paths, read in authored order."""

    FILE = "file"
    """A file that is neither a folder nor a list; a mis-typed split."""


@dataclass(frozen=True)
class SplitDeclaration:
    """One declared split: where it lives, how it is written, what it holds."""

    name: str
    kind: SplitKind
    path: Path
    exists: bool
    outside_root: bool
    #: Resolution bases for list entries, matching the historical readers: an
    #: entry is tried against the dataset root first and the YAML's folder
    #: second. They are private because they are not facts about the split.
    _root: Path = field(repr=False, compare=False)
    _yaml_dir: Path = field(repr=False, compare=False)

    def images(self) -> tuple[Path, ...]:
        """The split's images, in stable order.

        A directory is walked recursively, filtered by the project's supported
        image extensions and sorted, through the shared `find_images`. A list
        keeps its authored order. Both drop repeated paths, keeping the first
        occurrence. An absent path, a split that is not declared, or a non-list
        file has no images.
        """
        if self.kind is SplitKind.DIRECTORY:
            if not self.exists:
                return ()
            return tuple(dict.fromkeys(find_images(self.path)))
        if self.kind is SplitKind.LIST:
            return self._list_images()
        return ()

    def files(self) -> tuple[Path, ...]:
        """Every regular file under a directory split, sorted; one walk.

        Empty for a list split or a missing path. Used to check what is present
        without re-walking, e.g. to refuse an image format membership filters out.
        """
        if self.kind is not SplitKind.DIRECTORY or not self.exists:
            return ()
        return tuple(sorted(path for path in self.path.rglob("*") if path.is_file()))

    def _list_images(self) -> tuple[Path, ...]:
        if not self.exists:
            return ()
        try:
            text = self.path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            raise CodedFailure(
                "dataset.split_list_unreadable",
                N_("cannot read the split list %(path)s: %(error)s"),
                path=str(self.path),
                error=str(exc),
            ) from exc
        found: list[Path] = []
        for line in text.splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            path = self._list_entry(entry)
            if path is not None:
                found.append(path)
        return tuple(dict.fromkeys(found))

    def _list_entry(self, entry: str) -> Path | None:
        path = Path(entry)
        if path.is_absolute():
            return path if path.exists() else None
        candidate = (self._root / path).resolve()
        if candidate.exists():
            return candidate
        fallback = (self._yaml_dir / path).resolve()
        return fallback if fallback.exists() else None


@dataclass(frozen=True)
class DatasetDeclaration:
    """The resolved YAML box dataset declaration.

    ``root`` is the declared ``path:`` when there is one, resolved against the
    YAML's folder, and the YAML's folder otherwise. ``names`` is ordered by
    class id. ``declared_nc`` is whatever ``nc`` said, or ``None``.
    """

    path: Path
    root: Path
    names: dict[int, str]
    declared_nc: int | None
    is_real_domain: bool
    _splits: dict[str, SplitDeclaration] = field(repr=False, compare=False)

    @classmethod
    def load(cls, path: str | Path) -> DatasetDeclaration:
        """Read one ``data.yaml``. A missing or unreadable file is a coded failure."""
        declaration_path = Path(path).resolve()
        try:
            text = declaration_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise CodedFailure(
                "dataset.data_yaml_missing",
                N_("data.yaml not found: %(path)s"),
                path=str(declaration_path),
            ) from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise CodedFailure(
                "dataset.data_yaml_unreadable",
                N_("could not read data.yaml %(path)s: %(error)s"),
                path=str(declaration_path),
                error=str(exc),
            ) from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise CodedFailure(
                "dataset.data_yaml_unreadable",
                N_("could not read data.yaml %(path)s: %(error)s"),
                path=str(declaration_path),
                error=str(exc),
            ) from exc
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise CodedFailure(
                "dataset.declaration_not_mapping",
                N_("%(path)s must contain a YAML mapping"),
                path=str(declaration_path),
            )
        return cls._from_mapping(declaration_path, data)

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        root: str | Path,
        names: Mapping[int, str] | Sequence[str],
        splits: Mapping[str, str | Path],
        is_real_domain: bool = False,
    ) -> DatasetDeclaration:
        """Build a declaration for writing, from concrete paths.

        Writers hand over the folder or list file they just produced; the kind
        is inferred from the path, exactly as :meth:`load` infers it.
        """
        declaration_path = Path(path).resolve()
        resolved_root = Path(root).resolve()
        parsed_names = _parse_names(names, declaration_path)
        declared_splits = {
            name: _split_declaration(
                name, Path(value).resolve(), resolved_root, declaration_path.parent
            )
            for name, value in splits.items()
        }
        return cls(
            path=declaration_path,
            root=resolved_root,
            names=parsed_names,
            declared_nc=_class_count(parsed_names),
            is_real_domain=is_real_domain,
            _splits=declared_splits,
        )

    @classmethod
    def _from_mapping(cls, path: Path, data: dict[str, Any]) -> DatasetDeclaration:
        yaml_dir = path.parent
        root = yaml_dir
        root_value = data.get("path")
        if root_value:
            candidate = Path(str(root_value))
            root = candidate if candidate.is_absolute() else (yaml_dir / candidate).resolve()
        splits = {}
        for name in SPLIT_NAMES:
            value = data.get(name)
            if value:
                splits[name] = _split_declaration(
                    name, _resolve_split(yaml_dir, root, value), root, yaml_dir
                )
        return cls(
            path=path,
            root=root,
            names=_parse_names(data.get("names"), path),
            declared_nc=_declared_nc(data.get("nc")),
            is_real_domain=declares_real_domain(data),
            _splits=splits,
        )

    def split(self, name: str) -> SplitDeclaration | None:
        """The named split, or ``None`` when the declaration does not name it."""
        return self._splits.get(name)

    def class_names(self) -> list[str]:
        """The names as a dense, id-ordered list."""
        return [self.names[index] for index in sorted(self.names)]

    @property
    def class_count(self) -> int:
        """The number of classes the ids span: ``max(id) + 1``, or zero."""
        return _class_count(self.names)


def write_declaration(declaration: DatasetDeclaration, path: str | Path | None = None) -> None:
    """Write one canonical declaration: root, splits, names and ``nc``.

    A split inside the dataset root is written relative to it, so the file stays
    portable; one that lives outside keeps its absolute path. A dense name map
    is written as a list, and a sparse one as a mapping so its class ids survive
    the round trip. `domain: real` is written when the declaration says the
    images are real-domain. What this writes, :meth:`DatasetDeclaration.load`
    reads back to the same facts. The destination is ``path`` or, by default,
    ``declaration.path``; the two cannot disagree.
    """
    payload: dict[str, Any] = {"path": str(declaration.root)}
    for name in SPLIT_NAMES:
        split = declaration.split(name)
        if split is None:
            continue
        payload[name] = _split_value(split, declaration.root)
    payload["names"] = _names_payload(declaration.names)
    payload["nc"] = declaration.class_count
    if declaration.is_real_domain:
        payload["domain"] = "real"
    destination = Path(path) if path is not None else declaration.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def declares_real_domain(metadata: Any) -> bool:
    """Whether a dataset declaration names real-domain imagery.

    Shared by the YAML declaration and the canonical COCO store, so the campaign
    cohort check has one rule for both sources.
    """
    if not isinstance(metadata, Mapping):
        return False
    for key in ("domain", "validation_domain", "image_domain"):
        if str(metadata.get(key, "")).strip().lower() in {"real", "real_domain"}:
            return True
    info = metadata.get("info")
    return declares_real_domain(info) if isinstance(info, Mapping) else False


def _resolve_split(yaml_dir: Path, root: Path, value: Any) -> Path:
    """Resolve one split value against the declared root, then the YAML's folder."""
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    under_root = (root / candidate).resolve()
    if under_root.exists():
        return under_root
    return (yaml_dir / candidate).resolve()


def _split_declaration(name: str, path: Path, root: Path, yaml_dir: Path) -> SplitDeclaration:
    return SplitDeclaration(
        name=name,
        kind=_split_kind(path),
        path=path,
        exists=path.exists(),
        outside_root=not path.is_relative_to(root),
        _root=root,
        _yaml_dir=yaml_dir,
    )


def _split_kind(path: Path) -> SplitKind:
    if path.is_dir():
        return SplitKind.DIRECTORY
    if path.suffix.lower() == ".txt":
        return SplitKind.LIST
    if path.exists():
        return SplitKind.FILE
    # A missing path without a list suffix is an intended folder that is not
    # there yet: it holds no images, and the audit reports the missing path.
    return SplitKind.DIRECTORY


def _split_value(split: SplitDeclaration, root: Path) -> str:
    if split.path.is_relative_to(root):
        return split.path.relative_to(root).as_posix()
    return str(split.path)


def _names_payload(names: dict[int, str]) -> list[str] | dict[int, str]:
    """A dense map becomes a list; a sparse one keeps its ids as a mapping."""
    if sorted(names) == list(range(len(names))):
        return [names[index] for index in sorted(names)]
    return dict(sorted(names.items()))


def _class_count(names: dict[int, str]) -> int:
    """The number of classes the ids span, which is what ``nc`` means."""
    return max(names) + 1 if names else 0


def _parse_names(raw: Any, path: Path) -> dict[int, str]:
    if not raw:
        return {}
    if isinstance(raw, (list, tuple)):
        return {index: str(value) for index, value in enumerate(raw)}
    if isinstance(raw, Mapping):
        parsed: dict[int, str] = {}
        for key, value in raw.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise CodedFailure(
                    "dataset.names_invalid",
                    N_(
                        "%(path)s must declare 'names' as a list or a mapping of class IDs to names"
                    ),
                    path=str(path),
                ) from exc
            if index < 0:
                raise CodedFailure(
                    "dataset.names_invalid",
                    N_(
                        "%(path)s must declare 'names' as a list or a mapping of class IDs to names"
                    ),
                    path=str(path),
                )
            parsed[index] = str(value)
        return dict(sorted(parsed.items()))
    raise CodedFailure(
        "dataset.names_invalid",
        N_("%(path)s must declare 'names' as a list or a mapping of class IDs to names"),
        path=str(path),
    )


def _declared_nc(raw: Any) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None
