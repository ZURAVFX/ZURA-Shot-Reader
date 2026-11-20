"""Create the Blender installation ZIP from an explicit set of source files."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


root = Path(__file__).resolve().parents[1]
package = root / "shot_reader_local"
dist = root / "dist"
dist.mkdir(exist_ok=True)
with ZipFile(dist / "ZURA-Shot-Reader-v0.3.0.zip", "w", compression=ZIP_DEFLATED) as archive:
    for name in ("__init__.py", "blockout.py", "infer.py", "character.blend"):
        archive.write(package / name, f"shot_reader_local/{name}")
