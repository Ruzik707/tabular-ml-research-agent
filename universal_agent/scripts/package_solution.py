#!/usr/bin/env python3
"""Build a deterministic ZIP containing only the AutoML runtime allowlist."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
REQUIRED_FILES = ("solution.py", "README.md", ".env.example", "requirements-mvp.txt")


def collect_runtime_files(root: Path) -> list[tuple[str, Path]]:
    """Return safe, archive-relative runtime files in a stable order."""
    root = root.resolve()
    files: dict[str, Path] = {}

    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Обязательный файл не найден: {relative}")
        if path.is_symlink():
            raise ValueError(f"Символьная ссылка запрещена в архиве: {relative}")
        files[relative] = path

    package_dir = root / "automl_agent"
    if not package_dir.is_dir():
        raise FileNotFoundError("Обязательная директория не найдена: automl_agent")
    if package_dir.is_symlink():
        raise ValueError("Символьная ссылка запрещена: automl_agent")
    for path in package_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"В runtime-пакете найдена небезопасная ссылка: {path.relative_to(root)}")
        if path.is_file() and path.suffix == ".py":
            files[path.relative_to(root).as_posix()] = path
    if not any(name.startswith("automl_agent/") for name in files):
        raise FileNotFoundError("В automl_agent не найдено ни одного Python-модуля")

    return [(name, files[name]) for name in sorted(files)]


def build_archive(root: Path, output: Path) -> int:
    root = root.resolve()
    output = output.resolve()
    files = collect_runtime_files(root)
    included_paths = {source.resolve() for _, source in files}
    if output in included_paths:
        raise ValueError("Путь ZIP совпадает с runtime-файлом и не может быть перезаписан")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as temporary:
            temp_path = Path(temporary.name)

        with ZipFile(temp_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for relative, source in files:
                info = ZipInfo(relative, date_time=FIXED_ZIP_TIME)
                info.compress_type = ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.flag_bits |= 0x800
                archive.writestr(info, source.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)

        with temp_path.open("rb") as archive_file:
            os.fsync(archive_file.fileno())
        os.replace(temp_path, output)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="каталог проекта (по умолчанию — корень репозитория)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="путь результата (по умолчанию ROOT/dist/solution.zip)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = (args.output or root / "dist" / "solution.zip").resolve()
    try:
        count = build_archive(root, output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Создан {output} ({count} файлов runtime).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
