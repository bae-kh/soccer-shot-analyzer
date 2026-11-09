"""Original-pixel evidence for ball identity; scores are not probabilities."""
import cv2
import numpy as np
from collections import deque


def ball_patch(frame, bbox):
    """Center crop with an elliptical mask, without assuming a white ball."""
    u, v, w, h = map(float, bbox)
    if not np.isfinite([u, v, w, h]).all() or min(w, h) < 4:
        return None
    # The inner part reduces contamination from rectangle corners/background.
    crop = cv2.getRectSubPix(frame, (max(4, round(w)), max(4, round(h))), (u, v))
    patch = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[:32, :32]
    mask = (((xx - 15.5) / 14) ** 2 + ((yy - 15.5) / 14) ** 2 <= 1).astype(np.uint8)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # Hue is undefined on gray/white/black pixels: separate achromatic bins.
    colors = np.where(hsv[:, :, 1] < 55, 12, hsv[:, :, 0] // 15)
    color_hist = np.bincount(colors[mask != 0], minlength=13).astype(float)
    value_hist = np.bincount((hsv[:, :, 2] // 32)[mask != 0], minlength=8).astype(float)
    color_hist /= color_hist.sum()
    value_hist /= value_hist.sum()
    return {'color': color_hist, 'value': value_hist,
            'bright_fraction': float(np.mean(hsv[:, :, 2][mask != 0] >= 150)),
            'mean_value': float(np.mean(hsv[:, :, 2][mask != 0])), 'patch': patch}


def overlap(left, right):
    return float(np.minimum(left, right).sum())


class BallObservationGuard:
    def __init__(self, frame, bbox):
        self.reference = ball_patch(frame, bbox)
        self.seed_bbox = bbox
        self.scale = min(1.0, 480 / frame.shape[1])
        self.background = self._small(frame)
        self.background_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        mask = np.full(self.background_gray.shape, 255, dtype=np.uint8)
        u, v, w, h = bbox
        cv2.circle(mask, (round(u * self.scale), round(v * self.scale)),
                   max(4, round(max(w, h) * self.scale)), 0, -1)
        self.landmarks = cv2.goodFeaturesToTrack(self.background_gray, 160, .015, 7, mask=mask)
        self.frame_number = None
        self.current = None
        self.history = deque(maxlen=3)
        self.aligned_background = None
        self.alignment_verified = False
        self.cache = {}

    def _small(self, frame):
        return cv2.resize(frame, (round(frame.shape[1] * self.scale), round(frame.shape[0] * self.scale)),
                          interpolation=cv2.INTER_AREA)

    def begin_frame(self, frame_number, frame):
        if self.frame_number == frame_number:
            return
        if self.current is not None and frame_number == self.frame_number + 1:
            self.history.append(self.current)
        elif self.current is not None:
            self.history.clear()
        self.frame_number = frame_number
        self.current = self._small(frame)
        self.cache = {}
        self.aligned_background = None
        self.alignment_verified = False
        if self.landmarks is None or len(self.landmarks) < 12:
            return
        gray = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        points, status, _ = cv2.calcOpticalFlowPyrLK(self.background_gray, gray, self.landmarks, None,
                                                  winSize=(21, 21), maxLevel=3)
        if points is None:
            return
        valid = status.reshape(-1).astype(bool)
        if valid.sum() < 12:
            return
        old, new = self.landmarks[valid].reshape(-1, 2), points[valid].reshape(-1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC,
                                                     ransacReprojThreshold=1.5)
        if matrix is None or inliers.sum() < 12 or inliers.mean() < .6:
            return
        residual = np.linalg.norm(cv2.transform(old[:, None, :], matrix).reshape(-1, 2) - new, axis=1)
        if np.median(residual[inliers.reshape(-1).astype(bool)]) > .8:
            return
        self.aligned_background = cv2.warpAffine(self.background, matrix, (gray.shape[1], gray.shape[0]),
                                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.alignment_verified = True

    def _change(self, background, bbox):
        scaled = [float(x) * self.scale for x in bbox]
        scaled[2:] = [max(4, x) for x in scaled[2:]]
        current, old = ball_patch(self.current, scaled), ball_patch(background, scaled)
        if current is None or old is None:
            return None
        yy, xx = np.mgrid[:32, :32]
        mask = ((xx - 15.5) ** 2 + (yy - 15.5) ** 2) <= 14 ** 2
        return float(np.mean(np.abs(current['patch'].astype(float) - old['patch'].astype(float))[mask]))

    def inspect(self, frame, bbox, source='yolo', previous_center=None, previous_radius=None):
        feature = ball_patch(frame, bbox)
        if feature is None or self.reference is None:
            return {'accepted': False, 'reason': 'INVALID_APPEARANCE_PATCH'}
        reference = self.reference
        color = overlap(reference['color'], feature['color'])
        value = overlap(reference['value'], feature['value'])
        bright_drop = reference['bright_fraction'] - feature['bright_fraction']
        mean_drop = reference['mean_value'] - feature['mean_value']
        result = {'accepted': True, 'reason': None, 'color_overlap': round(color, 4),
                  'value_overlap': round(value, 4), 'bright_fraction_drop': round(bright_drop, 4),
                  'mean_value_drop': round(mean_drop, 2)}
        # Broad limits: do not require a specific color or identical panel rotation.
        if color < .28 or (bright_drop > .48 and mean_drop > 65):
            result.update(accepted=False, reason='BALL_APPEARANCE_MISMATCH')
        elif source == 'csrt' and value < .22:
            # CSRT has no detector evidence: a nearly disjoint brightness
            # distribution cannot keep correcting the ball position.
            result.update(accepted=False, reason='CSRT_APPEARANCE_MISMATCH')
        elif source == 'csrt' and value < .45 and bright_drop > .35 and mean_drop > 55:
            # With no detector confirmation, a patch dominated by a newly
            # dark occluder cannot establish an observed ball position.
            result.update(accepted=False, reason='CSRT_OCCLUSION_OR_APPEARANCE_MISMATCH')
        if result['accepted'] and source == 'csrt' and self.current is not None:
            seed_distance = float(np.linalg.norm(np.array(bbox[:2]) - np.array(self.seed_bbox[:2])))
            reference_scene = self.aligned_background if self.alignment_verified else self.background
            change = self._change(reference_scene, bbox)
            result.update(csrt_scene_change=round(change, 3) if change is not None else None)
            # CSRT can stick at the last YOLO location after the ball leaves.
            # Unlike a new YOLO candidate, its drift need not be far from that
            # last position. Preserve the seed area and a ball which changes
            # the pre-shot background, including stationary/duplicate frames.
            if seed_distance > max(self.seed_bbox[2:]) * 1.5 and change is not None and change < 3.:
                result.update(accepted=False, reason='CSRT_UNCHANGED_BACKGROUND')
        # A new location which still looks exactly like the pre-shot scene is
        # a stationary letter/icon/ground patch, not evidence of an arriving ball.
        # Preserve a ball already at that location, including slow/duplicate frames.
        if result['accepted'] and self.current is not None and previous_center is not None:
            travel = np.linalg.norm(np.array(bbox[:2]) - np.array(previous_center))
            nearby = max(5, (previous_radius or min(bbox[2:]) / 2) * 1.1)
            if travel > nearby:
                screen_change = self._change(self.background, bbox)
                scene_change = self._change(self.aligned_background, bbox) if self.alignment_verified else None
                result.update(screen_change=round(screen_change, 3) if screen_change is not None else None,
                              scene_change=round(scene_change, 3) if scene_change is not None else None)
                changes = [x for x in (screen_change, scene_change) if x is not None]
                recent = [self._change(old, bbox) for old in self.history]
                recent = [x for x in recent if x is not None]
                recent_change = float(np.mean(recent)) if len(recent) >= 2 else None
                result['recent_screen_change'] = round(recent_change, 3) if recent_change is not None else None
                if recent_change is not None:
                    changes.append(recent_change)
                # Compression/net texture can slightly change a static letter.
                # Allow that tolerance only with a second identity mismatch cue.
                static_limit = 10.0 if value < .55 else 3.0
                if changes and min(changes) < static_limit:
                    result.update(accepted=False, reason='UNCHANGED_BACKGROUND_OBJECT')
        return result
