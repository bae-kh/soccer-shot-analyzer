"""Pixel-space ball association and bounded Kalman prediction for a single shot."""
import math
import numpy as np
try:
    from .ball_observation import BallObservationGuard
except ImportError:
    from ball_observation import BallObservationGuard


def ball_radius(bbox):
    return max(1.0, min(bbox[2], bbox[3]) / 2.0)


def select_initial_ball(frames, width, height):
    """Prefer a confident ball near the shooting area, while allowing blurred boxes."""
    candidates = []
    for frame_number, detections in frames:
        for detection in detections:
            u, v, w, h = detection["bbox"]
            if detection["confidence"] < 0.1 or w <= 0 or h <= 0 or not 0.3 < w / h < 3.5:
                continue
            score = detection["confidence"] - 0.45 * abs(u / width - 0.5) - 0.65 * abs(v / height - 0.8)
            candidates.append((score, -frame_number, frame_number, detection))
    if not candidates:
        return None
    # Start from the earliest visible shooting ball, not a later high-confidence
    # airborne point or a stationary logo behind the goal.
    lower = [row for row in candidates if row[3]["bbox"][1] >= height * 0.4]
    if lower:
        candidates = lower
    first_frame = min(row[2] for row in candidates)
    candidates = [row for row in candidates if row[2] <= first_frame + 4]
    _, _, frame_number, detection = max(candidates, key=lambda row: row[:2])
    initial = {**detection, "frame_number": frame_number}
    clusters = []
    anchor = np.array(detection["bbox"][:2])
    for frame_number, detections in frames:
        for candidate in detections:
            center = np.array(candidate["bbox"][:2])
            radius = ball_radius(candidate["bbox"])
            if np.linalg.norm(center - anchor) < max(20, ball_radius(detection["bbox"]) * 2):
                continue
            cluster = next((c for c in clusters if np.linalg.norm(center - c["center"]) < max(5, radius * 0.5)), None)
            if cluster is None:
                clusters.append({"center": center, "radius": radius, "frames": {frame_number}})
            else:
                cluster["frames"].add(frame_number)
    initial["static_distractors"] = [{"center": c["center"].tolist(), "radius": c["radius"]}
                                    for c in clusters if len(c["frames"]) >= 6]
    return initial


class PixelKalman:
    """State: x, y, radius, vx, vy, radius velocity; all pixel quantities."""
    def __init__(self, bbox, fps, diagonal):
        self.dt = 1.0 / fps
        self.diagonal = diagonal
        self.state = np.array([bbox[0], bbox[1], ball_radius(bbox), 0, 0, 0], dtype=float)
        self.covariance = np.diag([16, 16, 9, diagonal ** 2, diagonal ** 2, 100.0])
        self.observation = np.zeros((3, 6))
        self.observation[:, :3] = np.eye(3)

    def predict(self):
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * self.dt
        acceleration = np.zeros((6, 3))
        acceleration[:3] = np.eye(3) * self.dt ** 2 / 2
        acceleration[3:] = np.eye(3) * self.dt
        noise = np.diag([(self.diagonal * 5) ** 2, (self.diagonal * 5) ** 2, 100.0 ** 2])
        self.state = transition @ self.state
        self.state[2] = max(1.0, self.state[2])
        self.covariance = transition @ self.covariance @ transition.T + acceleration @ noise @ acceleration.T

    def correct(self, bbox, confidence=None):
        measurement = np.array([bbox[0], bbox[1], ball_radius(bbox)])
        sigma = max(2.0, measurement[2] * 0.08)
        if confidence is None:
            sigma *= 2  # CSRT supplies no calibrated detection confidence.
        noise = np.diag([sigma ** 2, sigma ** 2, max(2.0, measurement[2] * 0.2) ** 2])
        residual = measurement - self.observation @ self.state
        innovation = self.observation @ self.covariance @ self.observation.T + noise
        gain = np.linalg.solve(innovation, self.observation @ self.covariance).T
        self.state += gain @ residual
        identity = np.eye(6) - gain @ self.observation
        self.covariance = identity @ self.covariance @ identity.T + gain @ noise @ gain.T
        self.state[2] = max(1.0, self.state[2])


class BallTracker:
    def __init__(self, initial, width, height, fps, initial_frame=None):
        self.width, self.height, self.fps = width, height, fps
        self.diagonal = math.hypot(width, height)
        self.anchor = np.array(initial["bbox"][:2], dtype=float)
        self.initial_radius = ball_radius(initial["bbox"])
        self.static_distractors = initial.get("static_distractors", [])
        self.filter = PixelKalman(initial["bbox"], fps, self.diagonal)
        self.active = False
        self.finished = False
        self.missed = 0
        self.prediction_limit = max(1, round(fps * 0.2))
        self.recovery_limit = max(3, round(fps * 0.7))
        self.kick_threshold = max(4.0, self.initial_radius * 0.6, self.diagonal * 0.002)
        self.last_observation = None
        self.last_static_frame = None
        self.shot_start_frame = None
        self.pending_motion = None
        self.pending_reacquisition = None
        self.stationary_frames = 0
        self.records = []
        self.current_frame = 0
        self.observation_guard = BallObservationGuard(initial_frame, initial['bbox']) if initial_frame is not None else None
        self.last_yolo = None
        self.rejected_observations = []
        self.frame_rejections = []

    def prepare_frame(self, frame_number, frame):
        if self.observation_guard is not None and frame is not None:
            self.observation_guard.begin_frame(frame_number, frame)

    def _validate_observation(self, frame, bbox, source):
        if self.observation_guard is None or frame is None:
            return {'accepted': True, 'reason': None, 'status': 'IMAGE_CHECK_UNAVAILABLE'}
        previous = self.last_yolo
        result = self.observation_guard.inspect(frame, bbox, source,
                     previous_center=previous[1] if previous else self.anchor,
                     previous_radius=previous[2] if previous else self.initial_radius)
        return {**result, 'status': 'IMAGE_CHECKED', 'method': 'seed_color_brightness_and_pre_shot_background'}

    def _reject(self, source, bbox, validation, confidence=None):
        event = {'frame_number': self.current_frame, 'source': source,
                 'bbox_xywh': list(map(float, bbox)), 'confidence': confidence, **validation}
        self.frame_rejections.append(event)
        self.rejected_observations.append(event)

    def _gate(self):
        if not self.active:
            return self.diagonal * 0.3
        uncertainty = math.sqrt(max(self.filter.covariance[0, 0], self.filter.covariance[1, 1]))
        travel = np.linalg.norm(self.filter.state[3:5]) / self.fps
        gate = max(self.diagonal * 0.025, self.filter.state[2] * 3, travel * 1.5, uncertainty * 3)
        if self.shot_start_frame is not None and self.current_frame - self.shot_start_frame <= round(self.fps * 0.2):
            return min(self.diagonal * 0.2, max(gate, self.diagonal * 0.15))
        return min(self.diagonal * 0.12, gate)

    def _weak_reversal(self, center, confidence):
        if not self.active or self.last_observation is None or (confidence is not None and confidence >= 0.12):
            return False
        if self.current_frame - self.shot_start_frame > round(self.fps * 0.2):
            return False
        motion = self.filter.state[3:5] / self.fps
        displacement = np.array(center) - self.last_observation[1]
        travel, distance = np.linalg.norm(motion), np.linalg.norm(displacement)
        minimum = max(4.0, self.filter.state[2] * 0.3)
        return travel > minimum and distance > minimum and np.dot(motion, displacement) < -0.2 * travel * distance

    def _weak_reacquisition(self, bbox, confidence, validation):
        """A weak distant match needs image identity or repeated detector evidence."""
        if not self.active or self.last_yolo is None or confidence >= .12:
            return False
        if (validation.get('status') == 'IMAGE_CHECKED' and validation.get('color_overlap', 0) >= .55
                and validation.get('value_overlap', 0) >= .55):
            return False
        last_frame, last_center, last_radius = self.last_yolo
        distance = np.linalg.norm(np.array(bbox[:2]) - last_center)
        if self.current_frame - last_frame <= 2 or distance <= max(last_radius * 3, self.diagonal * .02):
            return False
        pending = self.pending_reacquisition
        confirmed = (pending is not None and self.current_frame == pending[0] + 1
                     and np.linalg.norm(np.array(bbox[:2]) - pending[1]) <= max(pending[2] * 3, self.diagonal * .02))
        return not confirmed

    def _associate(self, detections, frame=None, record_rejections=False):
        position = self.filter.state[:2] if self.active else self.anchor
        radius = self.filter.state[2] if self.active else self.initial_radius
        gate = self._gate()
        candidates = []
        for detection in detections:
            u, v, w, h = detection["bbox"]
            if detection["confidence"] < 0.04 or w <= 0 or h <= 0 or not 0.2 < w / h < 4.5:
                continue
            ratio = ball_radius(detection["bbox"]) / radius
            if not 0.15 < ratio < 3.5:
                continue
            distance = np.linalg.norm(np.array([u, v]) - position)
            if distance > gate:
                continue
            if self._weak_reversal([u, v], detection["confidence"]):
                continue
            stationary_background = any(np.linalg.norm(np.array([u, v]) - np.array(d["center"])) < max(6, d["radius"])
                                        for d in self.static_distractors)
            if stationary_background and (not self.active or distance > gate * 0.35):
                continue
            validation = self._validate_observation(frame, detection['bbox'], 'yolo')
            if not validation['accepted']:
                if record_rejections:
                    self._reject('yolo', detection['bbox'], validation, float(detection['confidence']))
                continue
            if self._weak_reacquisition(detection['bbox'], detection['confidence'], validation):
                if record_rejections:
                    self._reject('yolo', detection['bbox'],
                                 {**validation, 'accepted': False, 'reason': 'WEAK_REACQUISITION_UNCONFIRMED'},
                                 float(detection['confidence']))
                continue
            early = self.active and self.shot_start_frame is not None and self.current_frame - self.shot_start_frame <= round(self.fps * 0.2)
            confidence_weight = 0.9 if early else 0.2
            cost = distance / gate + 0.12 * abs(math.log(ratio)) + confidence_weight * (1 - detection["confidence"])
            candidates.append((cost, {**detection, 'observation_validation': validation}))
        return min(candidates, key=lambda row: row[0])[1] if candidates else None

    def step(self, frame_number, detections, fallback_bbox=None, frame=None):
        self.current_frame = frame_number
        self.frame_rejections = []
        self.prepare_frame(frame_number, frame)
        self.filter.predict()
        detection = None if self.finished else self._associate(detections, frame, True)
        weak_candidates = [r for r in self.frame_rejections if r['reason'] == 'WEAK_REACQUISITION_UNCONFIRMED']
        if detection is not None:
            self.pending_reacquisition = None
        elif weak_candidates:
            candidate = max(weak_candidates, key=lambda r: r['confidence'])
            bbox = candidate['bbox_xywh']
            self.pending_reacquisition = (frame_number, np.array(bbox[:2]), ball_radius(bbox))
        elif self.pending_reacquisition is not None and frame_number > self.pending_reacquisition[0] + 1:
            self.pending_reacquisition = None
        source, confidence, measured = "missing", None, None
        validation = None
        reacquired = False
        if detection is not None:
            measured = list(map(float, detection["bbox"]))
            confidence = float(detection["confidence"])
            source = "yolo"
            validation = detection['observation_validation']
            movement = np.linalg.norm(np.array(measured[:2]) - self.anchor)
            if not self.active and movement > self.kick_threshold:
                pending = self.pending_motion
                consistent = pending is not None and frame_number - pending[0] <= 2 and np.linalg.norm(np.array(measured[:2]) - pending[1]) < self.diagonal * 0.15
                if self.last_observation is not None and (confidence >= 0.12 or consistent):
                    self.active = True
                    self.shot_start_frame = self.last_static_frame if self.last_static_frame is not None else self.last_observation[0]
                    dt = (frame_number - self.last_observation[0]) / self.fps
                    if dt > 0:
                        self.filter.state[3:5] = (np.array(measured[:2]) - self.last_observation[1]) / dt
                    self.filter.state[:3] = [measured[0], measured[1], ball_radius(measured)]
                self.pending_motion = (frame_number, np.array(measured[:2]))
            elif not self.active:
                self.last_static_frame = frame_number
                self.pending_motion = None
            reacquired = self.active and self.missed > 0
            if self.active and self.last_observation is not None:
                displacement = np.linalg.norm(np.array(measured[:2]) - self.last_observation[1])
                self.stationary_frames = self.stationary_frames + 1 if displacement < max(1.0, ball_radius(measured) * 0.08) else 0
            self.filter.correct(measured, confidence)
            if not self.active:
                self.filter.state[3:] = 0
            self.last_observation = (frame_number, np.array(measured[:2]))
            self.last_yolo = (frame_number, np.array(measured[:2]), ball_radius(measured))
            self.missed = 0
        elif self.active and not self.finished:
            self.missed += 1
            if callable(fallback_bbox):
                fallback_bbox = fallback_bbox()
            if fallback_bbox is not None and self.missed <= self.prediction_limit:
                radius_ratio = ball_radius(fallback_bbox) / self.filter.state[2]
                distance = np.linalg.norm(np.array(fallback_bbox[:2]) - self.filter.state[:2])
                if distance < self._gate() and 0.4 < radius_ratio < 2.0 and not self._weak_reversal(fallback_bbox[:2], None):
                    fallback_validation = self._validate_observation(frame, fallback_bbox, 'csrt')
                    if fallback_validation['accepted']:
                        measured = list(map(float, fallback_bbox))
                        self.filter.correct(measured)
                        source = "csrt"
                        validation = fallback_validation
                    else:
                        self._reject('csrt', fallback_bbox, fallback_validation)
            if source == "missing" and self.missed <= self.prediction_limit:
                source = "kalman_prediction"
            if self.missed > self.recovery_limit:
                self.finished = True
        if self.active and self.stationary_frames >= max(4, round(self.fps * 0.4)):
            self.finished = True
        valid = source != "missing"
        x, y, radius = map(float, self.filter.state[:3])
        valid = valid and 0 <= x < self.width and 0 <= y < self.height
        if not valid:
            source = "missing"
        record = {
            "frame_number": int(frame_number), "time_sec": frame_number / self.fps,
            "x": x if valid else None, "y": y if valid else None,
            "radius_px": radius if valid else None, "confidence": confidence,
            "source": source, "observed": source in {"yolo", "csrt"},
            "ball_bbox": [measured[0] - measured[2] / 2, measured[1] - measured[3] / 2,
                          measured[0] + measured[2] / 2, measured[1] + measured[3] / 2] if measured else None,
            "raw_center": measured[:2] if measured else None, "reacquired": reacquired,
            "detector": detection.get("detector", "full_frame") if detection is not None else None,
            "observation_validation": validation, "rejected_observations": self.frame_rejections,
        }
        self.records.append(record)
        return record, detection

    def trajectory(self):
        if self.shot_start_frame is None:
            return []
        return [r for r in self.records if r["frame_number"] >= self.shot_start_frame]

    def summary(self):
        shot = self.trajectory()
        counts = {name: sum(r["source"] == name for r in shot) for name in ("yolo", "csrt", "kalman_prediction", "missing")}
        return {"shot_start_frame": self.shot_start_frame, "kick_detected": self.active,
                "observed_points": counts["yolo"] + counts["csrt"], "source_counts": counts,
                "reacquisitions": sum(r["reacquired"] for r in shot),
                "image_validation_enabled": self.observation_guard is not None,
                "rejected_observation_count": len(self.rejected_observations),
                "rejection_reasons": {reason: sum(r['reason'] == reason for r in self.rejected_observations)
                                      for reason in sorted({r['reason'] for r in self.rejected_observations})},
                "prediction_limit_frames": self.prediction_limit, "recovery_limit_frames": self.recovery_limit}
