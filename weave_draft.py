"""
NOOL - Weaving Draft & Weavability Score
-----------------------------------------
This is the missing bridge between "AI generated a pretty picture" and
"an actual weaver can put this on a loom."

Given a design image (from the AI Design Studio, or any uploaded photo),
this module:

  1. Quantizes it down to a small colour palette + a warp/weft grid,
     the way a handloom pattern actually has to be encoded (finite
     threads, finite colours per pick) - not infinite photographic detail.
  2. Renders that grid as a graph-paper "weaving draft" image the
     artisan can count off directly, with a colour legend.
  3. Scores how loom-friendly the design is (colour count, fine detail,
     horizontal repeat) so the customer/artisan see *before* committing
     to an order whether this needs simplification, or is better suited
     to embroidery/print than handloom.

Nothing here claims to be a precise textile-engineering tool (real reed
counts, ends-per-inch, etc. depend on the specific loom and yarn) - it's
a fast, honest feasibility signal for the "AI Design -> real order"
handoff described in the SIH26090 pitch, with numbers clearly labelled
as illustrative.
"""

import io
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def _quantize_grid(image_path, grid_cols, grid_rows, n_colors):
    """
    Downsamples the image to (grid_rows x grid_cols) cells - each cell's
    colour is an area-average of that patch, which is a reasonable stand-in
    for "what colour would this region of the fabric read as" - then runs
    k-means to collapse that down to n_colors actual thread colours.

    Returns:
      labels_grid: (grid_rows, grid_cols) int array, index into palette
      palette: list of (r, g, b) tuples, length == n_colors actually used
      coverage: list of float, fraction of cells using each palette colour
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read image for weaving draft")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Area-average downsample to grid resolution (INTER_AREA averages
    # patches rather than just picking nearest pixel - closer to how a
    # thread colour would visually "read" for that patch of fabric).
    small = cv2.resize(img, (grid_cols, grid_rows), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    k = max(2, min(n_colors, len(np.unique(pixels.astype(int), axis=0))))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, attempts=4, flags=cv2.KMEANS_PP_CENTERS
    )
    centers = np.clip(centers, 0, 255).astype(int)
    labels_grid = labels.reshape(grid_rows, grid_cols)

    counts = np.bincount(labels.flatten(), minlength=k)
    coverage = (counts / counts.sum()).tolist()

    palette = [tuple(c) for c in centers]
    return labels_grid, palette, coverage


def _edge_density(image_path, grid_cols, grid_rows):
    """
    Rough proxy for 'fine detail that will get lost / is hard to weave':
    Canny edge pixel ratio at grid resolution. High-frequency detail
    (fine linework, text, photographic gradients) shows up as a high
    edge ratio; flat colour blocks and simple motifs show up low.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (grid_cols, grid_rows), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 60, 150)
    return float(np.count_nonzero(edges)) / edges.size


def _horizontal_repeat_bonus(labels_grid):
    """
    Handloom patterns are much easier to produce when they repeat in a
    unit across the width (you programme the repeat once). Checks if the
    left half and right half of the grid roughly match when tiled -
    a strong match earns a feasibility bonus.
    """
    rows, cols = labels_grid.shape
    if cols < 4:
        return 0.0
    half = cols // 2
    left = labels_grid[:, :half]
    right = labels_grid[:, half:half * 2]
    if left.shape != right.shape:
        return 0.0
    match_ratio = float(np.mean(left == right))
    # Only reward genuine repetition, not coincidental partial matches.
    return max(0.0, (match_ratio - 0.5) * 2) * 15  # up to +15 points


def _score_design(palette, edge_ratio, repeat_bonus):
    n_colors = len(palette)
    # Fewer colours = fewer thread changes/shuttles = easier. Handloom
    # commonly manages 3-6 colours comfortably; beyond that, cost/time
    # rises fast.
    color_score = max(0, 100 - max(0, n_colors - 3) * 12)

    # Lower edge density = fewer fine transitions the loom has to
    # resolve. Calibrated loosely against typical photographic vs.
    # motif-style test images.
    detail_score = max(0, 100 - min(edge_ratio * 260, 100))

    overall = 0.45 * color_score + 0.45 * detail_score + min(repeat_bonus, 15)
    overall = round(max(0, min(100, overall)))

    if overall >= 75:
        label = "Loom-friendly — close to weave-ready"
    elif overall >= 50:
        label = "Moderate — will need some simplification"
    else:
        label = "Complex — likely better suited to print/embroidery than handloom"

    return overall, label, round(color_score), round(detail_score), round(min(repeat_bonus, 15), 1)


def _render_draft_image(labels_grid, palette, cell_px=14):
    rows, cols = labels_grid.shape
    swatch_h = 60
    img_w = cols * cell_px
    img_h = rows * cell_px + swatch_h
    canvas = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for r in range(rows):
        for c in range(cols):
            color = tuple(int(v) for v in palette[labels_grid[r, c]])
            x0, y0 = c * cell_px, r * cell_px
            draw.rectangle([x0, y0, x0 + cell_px, y0 + cell_px], fill=color)

    # Grid lines: thin every cell, bold every 8 (a common weaving-chart
    # counting convention so an artisan can count threads in groups).
    for c in range(cols + 1):
        x = c * cell_px
        width = 2 if c % 8 == 0 else 1
        draw.line([(x, 0), (x, rows * cell_px)], fill=(90, 90, 90), width=width)
    for r in range(rows + 1):
        y = r * cell_px
        width = 2 if r % 8 == 0 else 1
        draw.line([(0, y), (cols * cell_px, y)], fill=(90, 90, 90), width=width)

    # Legend strip along the bottom: colour swatches + hex.
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    swatch_size = 18
    x = 10
    y = rows * cell_px + 14
    for color in palette:
        color = tuple(int(v) for v in color)
        draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=color, outline=(60, 60, 60))
        hex_code = "#%02x%02x%02x" % color
        draw.text((x + swatch_size + 6, y + 2), hex_code, fill=(30, 30, 30), font=font)
        x += swatch_size + 90
        if x > img_w - 90:
            x = 10
            y += swatch_size + 8
            canvas = canvas  # (height was pre-allocated with margin; long palettes may crowd — acceptable for demo)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def generate_weave_draft(image_path, grid_cols=48, grid_rows=64, n_colors=6):
    """
    Public entry point.

    Returns dict:
      draft_image_bytes: PNG bytes of the graph-paper draft + legend
      score: 0-100 int
      score_label: human-readable feasibility summary
      breakdown: { color_score, detail_score, repeat_bonus }
      palette: [{ "hex": "#rrggbb", "coverage_pct": float }, ...]
      grid: { "cols": int, "rows": int }
      disclaimer: str
    """
    labels_grid, palette, coverage = _quantize_grid(image_path, grid_cols, grid_rows, n_colors)
    edge_ratio = _edge_density(image_path, grid_cols, grid_rows)
    repeat_bonus = _horizontal_repeat_bonus(labels_grid)

    overall, label, color_score, detail_score, repeat_pts = _score_design(
        palette, edge_ratio, repeat_bonus
    )
    draft_bytes = _render_draft_image(labels_grid, palette)

    palette_out = [
        {"hex": "#%02x%02x%02x" % tuple(int(v) for v in color), "coverage_pct": round(pct * 100, 1)}
        for color, pct in sorted(zip(palette, coverage), key=lambda x: -x[1])
    ]

    return {
        "draft_image_bytes": draft_bytes,
        "score": overall,
        "score_label": label,
        "breakdown": {
            "color_score": color_score,
            "detail_score": detail_score,
            "repeat_bonus": repeat_pts,
        },
        "palette": palette_out,
        "grid": {"cols": grid_cols, "rows": grid_rows},
        "disclaimer": (
            "Illustrative feasibility signal only — actual reed count, ends-per-inch "
            "and yarn requirements depend on the specific loom, yarn and artisan."
        ),
    }
