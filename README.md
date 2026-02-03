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
ply-to-png
```

PNG files will be written to `output/` with the same base filename.

## Notes

- Rendering is a 2D projection. If faces are present in the PLY, the tool renders filled triangles; otherwise it falls back to a point cloud view.
- Input is always read from `input/` and output is always written to `output/`.
- You can choose the projection plane (XY, XZ, YZ) and control image size, background, and point size.
- Color mode defaults to `auto`: use vertex colors if present, otherwise depth-based coloring.

## CLI Options

```bash
ply-to-png --help
```

For debugging geometry differences, run:

```bash
ply-to-png --debug
```
