# ============================================================
# make_contour_masks.py
# Create contour-only masks for C and straight stimuli
# ============================================================

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


IMAGE_DIR = Path("/home/yentl/pytorch_gammanet/Images_jitter_final_2")
MASK_DIR = Path("/home/yentl/pytorch_gammanet/contour_masks_jitter_final_2")
MASK_DIR.mkdir(parents=True, exist_ok=True)

N = 512
LINE_WIDTH = 18


def bezier_curve_position(t, Ps):
    """Cubic Bezier curve."""
    t = np.asarray(t)
    P0, P1, P2, P3 = Ps

    return (
        ((1 - t) ** 3)[:, None] * P0 +
        (3 * ((1 - t) ** 2) * t)[:, None] * P1 +
        (3 * (1 - t) * (t ** 2))[:, None] * P2 +
        (t ** 3)[:, None] * P3
    )


def build_templates():
    Ps = np.array([
        [0.75, 0.9],
        [0.2,  0.9],
        [0.2,  0.1],
        [0.75, 0.1],
    ])

    scale = 0.5
    Ps_scaled = Ps * scale

    curve = bezier_curve_position(np.linspace(0, 1, 200), Ps_scaled)
    bx_base = curve[:, 0][20:-20]
    by_base = curve[:, 1][20:-20]

    x_offset = 0.07
    y_offset = 0.94

    bx_base = bx_base - np.min(bx_base) + x_offset
    by_base = by_base - np.max(by_base) + y_offset

    templates = {}

    for shift_idx in range(4):
        bx = bx_base + 0.1 * shift_idx
        by = by_base.copy()

        templates[("C", shift_idx)] = (bx, by)

        x_center = np.mean(bx)
        bx_mirror = 2 * x_center - bx
        templates[("bC", shift_idx)] = (bx_mirror, by)

        # Straight line based on same C template
        # n_points = 5
        # bx_straight = np.full(n_points, np.mean(bx))
        # by_straight = np.linspace(np.min(by), np.max(by), n_points)
        n_points = 7
        change = 0.03

        bx_straight = np.full(n_points, np.mean(bx))
        by_straight = np.linspace(np.min(by) + change, np.max(by) - change, n_points)

        templates[("straight", shift_idx)] = (bx_straight, by_straight)

    return templates


TEMPLATES = build_templates()


def parse_filename(path):
    """
    Expected:
        C_BL_0_J053_024.png
        straight_BL_0_J053_024.png
    """
    parts = path.stem.split("_")

    shape = parts[0]
    quadrant = parts[1]
    position = int(parts[2])

    if shape not in ["C", "straight"]:
        return None

    return {
        "shape": shape,
        "quadrant": quadrant,
        "position": position,
    }


def draw_template_mask(shape, position):
    bx, by = TEMPLATES[(shape, position)]

    x = bx * N
    y = by * N

    points = list(zip(x, y))

    mask = Image.new("L", (N, N), 0)
    draw = ImageDraw.Draw(mask)

    if shape == "straight":
        draw.line(points, fill=255, width=LINE_WIDTH)
    else:
        draw.line(points, fill=255, width=LINE_WIDTH, joint="curve")

    return mask


def apply_quadrant_transform(mask, original_shape, quadrant):
    """
    Reproduce the same quadrant flips as in the stimulus generator.
    """

    effective_shape = original_shape

    if quadrant == "BL":
        pass

    elif quadrant == "BR":
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if original_shape == "C":
            effective_shape = "bC"
        elif original_shape == "bC":
            effective_shape = "C"

    elif quadrant == "TL":
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    elif quadrant == "TR":
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if original_shape == "C":
            effective_shape = "bC"
        elif original_shape == "bC":
            effective_shape = "C"

    else:
        raise ValueError(f"Unknown quadrant: {quadrant}")

    return mask, effective_shape


def make_mask_for_image(image_path):
    info = parse_filename(image_path)

    if info is None:
        return False

    wanted_shape = info["shape"]
    quadrant = info["quadrant"]
    position = info["position"]

    candidate_masks = []

    if wanted_shape == "straight":
        base_shapes = ["straight"]
    else:
        base_shapes = ["C", "bC"]

    for base_shape in base_shapes:
        mask = draw_template_mask(base_shape, position)
        mask, effective_shape = apply_quadrant_transform(
            mask,
            original_shape=base_shape,
            quadrant=quadrant,
        )

        if effective_shape == wanted_shape:
            candidate_masks.append(mask)

    if len(candidate_masks) == 0:
        print("No mask found for:", image_path.name)
        return False

    final = Image.new("L", (N, N), 0)

    for mask in candidate_masks:
        final = Image.fromarray(
            np.maximum(np.asarray(final), np.asarray(mask)).astype(np.uint8)
        )

    save_path = MASK_DIR / image_path.name
    final.save(save_path)

    return True


n_done = 0

for image_path in sorted(IMAGE_DIR.glob("*.png")):
    ok = make_mask_for_image(image_path)
    if ok:
        n_done += 1

print(f"Saved {n_done} masks to: {MASK_DIR}")