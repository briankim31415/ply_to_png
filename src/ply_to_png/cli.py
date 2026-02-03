from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from plyfile import PlyData

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_image_size(value: str) -> Tuple[int, int]:
    try:
        width_str, height_str = value.lower().split("x")
        width = int(width_str)
        height = int(height_str)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            "image size must be in WIDTHxHEIGHT format, e.g. 1024x1024"
        )

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("image size must be positive")

    return width, height


def iter_ply_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".ply":
            yield path


def load_vertices(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError("PLY file has no vertex element")

    vertex = ply["vertex"]
    names = vertex.data.dtype.names or ()

    if "x" not in names or "y" not in names:
        raise ValueError("PLY vertex data must include x and y")

    x = np.asarray(vertex["x"], dtype=float)
    y = np.asarray(vertex["y"], dtype=float)
    if "z" in names:
        z = np.asarray(vertex["z"], dtype=float)
    else:
        z = np.zeros_like(x, dtype=float)

    return np.column_stack((x, y, z))


def project_points(points: np.ndarray, projection: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    if projection == "xy":
        return x, y, z
    if projection == "xz":
        return x, z, y
    if projection == "yz":
        return y, z, x

    raise ValueError(f"Unsupported projection: {projection}")


def _apply_limits(ax: plt.Axes, u: np.ndarray, v: np.ndarray) -> None:
    xmin, xmax = float(np.min(u)), float(np.max(u))
    ymin, ymax = float(np.min(v)), float(np.max(v))

    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def render_png(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    output_path: Path,
    *,
    image_size: Tuple[int, int],
    point_size: float,
    color_mode: str,
    color: str,
    background: str,
    dpi: int,
) -> None:
    width, height = image_size
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(background)
    fig.patch.set_facecolor(background)

    _apply_limits(ax, u, v)

    if color_mode == "depth":
        ax.scatter(u, v, c=depth, cmap="viridis", s=point_size, linewidths=0)
    else:
        ax.scatter(u, v, c=color, s=point_size, linewidths=0)

    fig.savefig(output_path, dpi=dpi, pad_inches=0)
    plt.close(fig)


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    projection: str,
    image_size: Tuple[int, int],
    point_size: float,
    color_mode: str,
    color: str,
    background: str,
    dpi: int,
) -> None:
    points = load_vertices(input_path)
    u, v, depth = project_points(points, projection)
    render_png(
        u,
        v,
        depth,
        output_path,
        image_size=image_size,
        point_size=point_size,
        color_mode=color_mode,
        color=color,
        background=background,
        dpi=dpi,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert .ply files to .png images.",
    )
    parser.add_argument(
        "--input-dir",
        default="input",
        help="Directory containing .ply files (default: input)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write .png files (default: output)",
    )
    parser.add_argument(
        "--projection",
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Projection plane for the image",
    )
    parser.add_argument(
        "--image-size",
        type=parse_image_size,
        default="1024x1024",
        help="Output image size as WIDTHxHEIGHT",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Point size for scatter rendering",
    )
    parser.add_argument(
        "--color-mode",
        choices=["depth", "uniform"],
        default="depth",
        help="Color points by depth or use a single color",
    )
    parser.add_argument(
        "--color",
        default="white",
        help="Point color when --color-mode=uniform",
    )
    parser.add_argument(
        "--background",
        default="black",
        help="Background color (e.g. black, white, #202020)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Output DPI (affects size calculation)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .png files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    ply_files = list(iter_ply_files(input_dir))
    if not ply_files:
        print(f"No .ply files found in {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for path in ply_files:
        output_path = output_dir / f"{path.stem}.png"
        if output_path.exists() and not args.overwrite:
            print(f"Skipping (exists): {output_path.name}")
            continue

        try:
            convert_file(
                path,
                output_path,
                projection=args.projection,
                image_size=args.image_size,
                point_size=args.point_size,
                color_mode=args.color_mode,
                color=args.color,
                background=args.background,
                dpi=args.dpi,
            )
            print(f"Wrote {output_path}")
        except Exception as exc:
            failures += 1
            print(f"Failed {path.name}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
