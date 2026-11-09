"""Conservative size evidence; never replace the detector box with a guessed circle."""
import cv2
import numpy as np


class BallSizeInspector:
    def __init__(self, frame, bbox):
        u, v, w, h = bbox
        patch = cv2.getRectSubPix(frame, (max(4, round(w)), max(4, round(h))), (float(u), float(v)))
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]]
        inner = ((xx-(patch.shape[1]-1)/2)/(patch.shape[1]*.4))**2 + ((yy-(patch.shape[0]-1)/2)/(patch.shape[0]*.4))**2 < 1
        values = hsv[:, :, 2][inner]
        saturation = hsv[:, :, 1][inner]
        # Use a seed-derived foreground only when a bright achromatic ball
        # is supported. Colored/dark balls keep the geometric checks.
        background = hsv[:, :, 2][~inner]
        foreground_value = float(np.percentile(values, 65))
        background_value = float(np.median(background))
        self.bright_ball = (float(np.mean((values > 140) & (saturation < 90))) > .5
                            and foreground_value-background_value > 35)
        self.value_threshold = max(140, min(220, (foreground_value+background_value)/2))

    def inspect(self, frame, bbox):
        x1, y1, x2, y2 = map(float, bbox)
        w, h = x2-x1, y2-y1
        result = {'usable': True, 'reason': None, 'method': 'box_shape_and_optional_seed_foreground',
                  'box_axis_ratio': round(max(w, h)/max(1, min(w, h)), 4),
                  'foreground_checked': False}
        # An elongated box can contain shoe/leg or motion smear even when YOLO
        # recognizes a ball. This rejects size, not the tracked identity.
        if max(w, h)/max(1, min(w, h)) > 1.4:
            result.update(usable=False, reason='ELONGATED_BALL_BOX')
        if not self.bright_ball or min(w, h) < 12:
            return result
        patch = cv2.getRectSubPix(frame, (max(4, round(w)), max(4, round(h))),
                                 (float((x1+x2)/2), float((y1+y2)/2)))
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 2] >= self.value_threshold) & (hsv[:, :, 1] < 100)).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return result
        # Multiple white panels can be disconnected by black panels. Their
        # combined hull describes support; the largest panel alone does not.
        contours = [c for c in contours if cv2.contourArea(c) > .015*patch.shape[0]*patch.shape[1]]
        if not contours:
            return result
        hull = cv2.convexHull(np.concatenate(contours))
        area = cv2.contourArea(hull)
        bx, by, bw, bh = cv2.boundingRect(hull)
        fraction = area/max(1, patch.shape[0]*patch.shape[1])
        # Only a substantial foreground component supplies negative evidence.
        # A faint/small/colored ball is not declared occluded from this mask.
        if fraction < .18 or fraction > .9:
            return result
        spans = [bw/patch.shape[1], bh/patch.shape[0]]
        result.update(foreground_checked=True, foreground_box_span=[round(p, 4) for p in spans],
                      foreground_area_fraction=round(fraction, 4))
        # Bright panels alone are not the full silhouette: shading/black panels
        # can shorten one span on a clean sphere. Require an asymmetric box too.
        if result['usable'] and min(spans) < .65 and max(spans) > .8 and result['box_axis_ratio'] > 1.15:
            result.update(usable=False, reason='PARTIAL_FOREGROUND_OR_BOX_CONTAMINATION')
        return result
