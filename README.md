# ply-to-png

Convert `.ply` files in a local input folder into `.png` images in a local output folder.

This project intentionally does **not** track the `input/` or `output/` folders in git.

## Quickstart

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Put your `.ply` files in `input/`.
3. Run the converter:

```bash
ply-to-png --input-dir input --output-dir output
```

PNG files will be written to `output/` with the same base filename.

## Notes

- Rendering is a 2D projection of the 3D vertices (point cloud view). This keeps the tool lightweight and avoids GPU requirements.
- You can choose the projection plane (XY, XZ, YZ) and control image size, background, and point size.

## CLI Options

```bash
ply-to-png --help
```
