"""Geometry helpers for plate rectification and insertion."""

from __future__ import annotations

import numpy as np


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """Return quadrilateral points in top-left, top-right, bottom-right, bottom-left order."""

    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected points with shape (4, 2), got {pts.shape}")

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def quad_size(points: np.ndarray) -> tuple[int, int]:
    """Estimate frontal rectangle size for a quadrilateral."""

    ordered = order_quad_points(points)
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bottom = np.linalg.norm(ordered[2] - ordered[3])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    width = max(1, int(round(max(width_top, width_bottom))))
    height = max(1, int(round(max(height_right, height_left))))
    return width, height


def rectangle_points(width: int, height: int) -> np.ndarray:
    """Return a rectangle matching OpenCV perspective transform coordinates."""

    return np.asarray(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

