from zipfile import ZIP_DEFLATED, ZipFile

from package_release import dist, make_addon, root


archive = make_addon()
preview = dist / f"{archive.stem}-Apple-Silicon-Preview.zip"
with ZipFile(preview, "w", compression=ZIP_DEFLATED) as out:
    out.write(archive, archive.name)
    out.write(root / "setup-runtime-macos.sh", "setup-runtime-macos.sh")
    out.write(root / "prepare-runtime-macos.py", "prepare-runtime-macos.py")
    out.write(root / "MAC-PREVIEW.md", "START-HERE.md")
print(preview)
