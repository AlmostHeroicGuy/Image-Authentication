"""Plate extraction, insertion, masks, and visualization primitives."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .geometry import order_quad_points, quad_size, rectangle_points


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok = cv2.imwrite(str(path), image, params)
    if not ok:
        raise ValueError(f"Could not write image: {path}")


def extract_plate(image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Perspective-warp an annotated quadrilateral into a tight frontal plate."""

    src_quad = order_quad_points(vertices)
    width, height = quad_size(src_quad)
    dst_rect = rectangle_points(width, height)
    homography = cv2.getPerspectiveTransform(src_quad, dst_rect)
    return cv2.warpPerspective(
        image,
        homography,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def make_plate_mask(image_shape: tuple[int, ...], vertices: np.ndarray) -> np.ndarray:
    """Create a binary mask for exactly the destination plate quadrilateral."""

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    quad = np.round(order_quad_points(vertices)).astype(np.int32)
    cv2.fillConvexPoly(mask, quad, 255)
    return mask


def insert_plate(
    destination_image: np.ndarray,
    destination_vertices: np.ndarray,
    source_plate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a source plate into a destination quadrilateral and Poisson blend it."""

    dst_quad = order_quad_points(destination_vertices)
    width, height = quad_size(dst_quad)
    resized_plate = cv2.resize(source_plate, (width, height), interpolation=cv2.INTER_CUBIC)

    src_rect = rectangle_points(width, height)
    homography = cv2.getPerspectiveTransform(src_rect, dst_quad)
    warped_plate = cv2.warpPerspective(
        resized_plate,
        homography,
        (destination_image.shape[1], destination_image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    mask = make_plate_mask(destination_image.shape, dst_quad)
    x, y, w, h = cv2.boundingRect(np.round(dst_quad).astype(np.int32))
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(destination_image.shape[1], x + w)
    y1 = min(destination_image.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Destination quadrilateral is outside the image")

    source_patch = warped_plate[y0:y1, x0:x1]
    patch_mask = mask[y0:y1, x0:x1]
    center = (int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2)))
    blended = cv2.seamlessClone(
        source_patch,
        destination_image,
        patch_mask,
        center,
        cv2.NORMAL_CLONE,
    )
    return blended, mask


def make_visualization_grid(
    original: np.ndarray,
    extracted_plate: np.ndarray,
    mask: np.ndarray,
    manipulated: np.ndarray,
    panel_width: int = 480,
) -> np.ndarray:
    """Create a vertical visualization: original, extracted plate, mask, manipulated."""

    panels = [
        _resize_to_width(original, panel_width),
        _resize_to_width(extracted_plate, panel_width),
        _resize_to_width(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), panel_width),
        _resize_to_width(manipulated, panel_width),
    ]
    return np.vstack(panels)


def _mask_center(mask: np.ndarray) -> tuple[int, int]:
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Cannot blend with an empty destination mask")
        return int(xs.mean()), int(ys.mean())
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def _resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width:
        return image
    scale = width / max(1, w)
    height = max(1, int(round(h * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
