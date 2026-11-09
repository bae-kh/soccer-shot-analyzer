"""Assemble goal mouths from full detections or separately segmented goal posts."""
import math


def assemble_goal(parts, width, height, frame_number):
    parts = [p for p in parts if p["confidence"] >= 0.1]
    posts = [p for p in parts if p["bbox"][3] - p["bbox"][1] >= height * 0.05
             and (p["bbox"][3] - p["bbox"][1]) > 2.5 * (p["bbox"][2] - p["bbox"][0])]
    choices = []
    for left in posts:
        for right in posts:
            a, b = left["bbox"], right["bbox"]
            h1, h2 = a[3] - a[1], b[3] - b[1]
            span, mean_h = b[2] - a[0], (h1 + h2) / 2
            if b[0] <= a[2] or span < width * 0.08 or span > width * 0.95:
                continue
            if min(h1, h2) / max(h1, h2) < 0.55 or not 1.5 < span / mean_h < 5.5:
                continue
            if abs(a[1] - b[1]) > mean_h * 0.4 or abs(a[3] - b[3]) > mean_h * 0.4:
                continue
            top = min(a[1], b[1])
            for crossbar in parts:
                c = crossbar["bbox"]
                if c[2] - c[0] > span * 0.8 and c[3] - c[1] < mean_h * 0.3 and abs(c[1] - top) < mean_h * 0.3 and c[0] <= a[2] and c[2] >= b[0]:
                    top = min(top, c[1])
            confidence = (left["confidence"] + right["confidence"]) / 2
            alignment = (abs(a[1] - b[1]) + abs(a[3] - b[3])) / mean_h
            score = confidence - 0.1 * alignment - 0.05 * abs(math.log((span / mean_h) / 3.0))
            choices.append((score, [a[0], top, b[2], max(a[3], b[3])], confidence, "paired_posts"))
    if not choices:
        for part in parts:
            box = part["bbox"]
            w, h = box[2] - box[0], box[3] - box[1]
            if w >= width * 0.08 and h >= height * 0.05 and 1.4 < w / h < 5.5:
                choices.append((part["confidence"], box, part["confidence"], part.get("method", "box")))
    if not choices:
        return None
    _, box, confidence, method = max(choices, key=lambda choice: choice[0])
    x1, y1, x2, y2 = map(float, box)
    return {"bbox": [x1, y1, x2, y2], "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "confidence": float(confidence), "method": method, "frame_number": frame_number,
            "coordinate_space": "original_pixels", "origin": "top_left"}
