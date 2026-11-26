import os
import sys
from pathlib import Path

import cv2
import numpy as np


GREEN = (60, 230, 80)
RED = (65, 65, 255)
YELLOW = (0, 220, 255)
ORANGE = (0, 140, 255)
WHITE = (245, 245, 245)


def draw_pruning(image, features, result):

    canvas = image.copy()
    unknown = result.reprojection.unknown
    for i, point in enumerate(features.keypoints):
        if not np.isfinite(point).all():
            continue
        if not result.keep_mask[i]:
            color = RED
        elif not result.reprojection.has_depth[i]:
            color = YELLOW
        elif unknown[i]:
            color = ORANGE
        else:
            color = GREEN
        x, y = np.rint(point).astype(int)
        cv2.circle(canvas, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
    return canvas


def draw_matches(a, b, matches, max_lines):

    canvas = np.hstack((a.copy(), b.copy()))
    w = a.shape[1]
    order = np.argsort(-matches.scores)[:max_lines]
    for i in order:
        p = tuple(int(v) for v in np.rint(matches.kpts0[i]))
        q = tuple(int(v) for v in np.rint(matches.kpts1[i]))
        q = (q[0] + w, q[1])
        cv2.line(canvas, p, q, GREEN, 1, cv2.LINE_AA)
        cv2.circle(canvas, p, 3, GREEN, -1, cv2.LINE_AA)
        cv2.circle(canvas, q, 3, GREEN, -1, cv2.LINE_AA)
    return canvas


def panel(image, title, subtitle):

    header = np.full((66, image.shape[1], 3), (32, 32, 32), dtype=np.uint8)
    cv2.putText(header, title, (14, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, WHITE, 2, cv2.LINE_AA)
    cv2.putText(header, subtitle, (14, 51), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, WHITE, 1, cv2.LINE_AA)
    return np.vstack((header, image))


def visualize_results(image_a, image_b, fa, fb, ra, rb, full, reduced,
                      frame_a, frame_b, full_ms, prune_ms, reduced_ms,
                      max_lines=150, save_path="results/demo.png", no_show=False):

    pruning = np.hstack((draw_pruning(image_a, fa, ra), draw_pruning(image_b, fb, rb)))
    figure = np.vstack((
        panel(pruning,
              f"1. Motion pruning | A [{frame_a}] {ra.n_kept}/{ra.n_total} kept | B [{frame_b}] {rb.n_kept}/{rb.n_total} kept",
              "Green: kept | Red: pruned | Yellow: no valid depth (kept) | Orange: other unknown (kept)"),
        panel(draw_matches(image_a, image_b, full, max_lines),
              f"2. Standard LightGlue | {len(full)} matches | {full_ms:.1f} ms",
              f"All keypoints: {len(fa)} / {len(fb)} | showing {min(len(full), max_lines)} highest-confidence matches"),
        panel(draw_matches(image_a, image_b, reduced, max_lines),
              f"3. Motion-pruned LightGlue | {len(reduced)} matches | {reduced_ms:.1f} ms (+{prune_ms:.1f} ms prune)",
              f"Retained keypoints: {len(ra.features)} / {len(rb.features)} | showing {min(len(reduced), max_lines)} highest-confidence matches"),
    ))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(save_path), figure):
        raise OSError(f"Could not save {save_path}")
    print(f"\nSaved visualization to {save_path.resolve()}")

    # Saving works even on a headless Linux machine (use --no-show).
    if not no_show:
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("No graphical display detected; open the saved PNG instead.")
        else:
            try:
                scale = min(1.0, 1000 / figure.shape[0], 1600 / figure.shape[1])
                preview = cv2.resize(figure, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
               
                cv2.imshow("Motion-Informed Keypoint Pruning", preview)
                
                print("Press any key to close the demo.")
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error:
                print("OpenCV GUI unavailable; open the saved PNG instead.")

    return figure
