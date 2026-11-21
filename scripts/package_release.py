"""Build the portable Blender add-on without bundling checkpoints or local preferences."""

import ast
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


root = Path(__file__).resolve().parents[1]
package = root / "shot_reader_local"
dist = root / "dist"


def make_addon():
    metadata = ast.parse((package / "__init__.py").read_text(encoding="utf-8")).body[0]
    version = ".".join(str(part) for part in ast.literal_eval(metadata.value)["version"])
    dist.mkdir(exist_ok=True)
    target = dist / f"ZURA-Shot-Reader-v{version}.zip"
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for source in (*sorted(package.glob("*.py")), package / "character.blend"):
            archive.write(source, f"shot_reader_local/{source.name}")
    return target


if __name__ == "__main__":
    print(make_addon())
