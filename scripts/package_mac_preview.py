from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


root = Path(__file__).resolve().parents[1]
dist = root / "dist"
dist.mkdir(exist_ok=True)
archive = dist / "ZURA-Shot-Reader-0.4.0-preview.zip"
with ZipFile(archive, "w", compression=ZIP_DEFLATED) as out:
    for name in ("__init__.py", "blockout.py", "infer.py", "character.blend"):
        source = root / "shot_reader_local" / name
        out.write(source, f"shot_reader_local/{name}")

with ZipFile(dist / "ZURA-Shot-Reader-Apple-Silicon-Preview.zip", "w", compression=ZIP_DEFLATED) as out:
    out.write(archive, archive.name)
    out.write(root / "setup-runtime-macos.sh", "setup-runtime-macos.sh")
    out.write(root / "prepare-runtime-macos.py", "prepare-runtime-macos.py")
    out.write(root / "MAC-PREVIEW.md", "START-HERE.md")
