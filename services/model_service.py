"""Model service integrating YOLOv8-Pose and trained BiGRU-Attention.

Hybrid Architecture:
- Robust Mode (default): Confidence-aware landmark fallback, delayed kinetics reset,
  and LRU session eviction for production camera streams.
- Geometric Posture Gating: Filters out rapid arm swinging and slight leaning via
  torso inclination angles and bounding box aspect ratio checks.
- Batched Inference: Optimized tensor batching for multiple detected persons.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# --- HẰNG SỐ CẤU HÌNH HỆ THỐNG ---
MAX_LEN = 16
DECISION_WINDOW = 8
VOTE_THRESHOLD = 0.625
SESSION_TTL_SECONDS = 30 * 60
CLEANUP_INTERVAL_SECONDS = 60
MAX_ACTIVE_SESSIONS = 200
INPUT_SIZE = 102
YOLO_IMGSZ = 640
MAX_MISSED_FRAMES = 15

# Cấu hình chống nhiễu cho Robust Mode & Geometric Gate
KINETICS_RESET_AFTER_MISSED_FRAMES = 2
LANDMARK_CONF_THRESHOLD = 0.15
MIN_FALL_TORSO_ANGLE = 42.0       # Góc nghiêng thân tối thiểu để coi là ngã (độ)
MAX_FALL_BBOX_RATIO = 1.15         # Tỷ lệ H/W tối đa khi ngã (trên sàn H/W < 1.0)

COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class BiGRUAttentionModel(nn.Module):
    """Mạng BiGRU kết hợp Feed-Forward Soft Attention chuẩn xác theo báo cáo."""

    def __init__(self, input_size: int = 102, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        attention_scores = self.attention(output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(output * attention_weights, dim=1)
        return self.fc(context)


@dataclass
class PersonState:
    """Động lực học và trạng thái té ngã của từng cá nhân."""
    track_id: int
    strict_mode: bool = False
    prev_raw_flat: np.ndarray | None = None
    prev_velocity: np.ndarray | None = None
    prev_hip_center: np.ndarray | None = None
    prev_height: float | None = None
    skeleton_buffer: deque = field(default_factory=lambda: deque(maxlen=MAX_LEN))
    decision_buffer: deque = field(default_factory=lambda: deque(maxlen=DECISION_WINDOW))
    fall_probability: float = 0.0
    is_fall: bool = False
    previous_is_fall: bool = False
    fall_count: int = 0
    last_box: np.ndarray | None = None
    missed_frames: int = 0
    last_torso_angle: float = 0.0

    def reset_kinetics(self) -> None:
        """Reset bộ đệm vận tốc và gia tốc."""
        self.prev_raw_flat = None
        self.prev_velocity = None
        self.prev_hip_center = None
        self.prev_height = None

    def check_geometric_fall(self, keypoints: np.ndarray, box: np.ndarray) -> bool:
        """Bộ lọc hình học: Xác nhận tư thế cơ thể có thực sự ngã/nằm ngang hay không."""
        # 1. Tính toán góc nghiêng trục thân (Vector từ hông lên vai)
        l_sh, r_sh = keypoints[5], keypoints[6]
        l_hip, r_hip = keypoints[11], keypoints[12]

        mid_shoulder = (l_sh + r_sh) / 2.0
        mid_hip = (l_hip + r_hip) / 2.0

        dx = float(mid_shoulder[0] - mid_hip[0])
        dy = float(mid_shoulder[1] - mid_hip[1])  # Lưu ý: Trục Y hướng xuống

        # Góc lệch so với phương thẳng đứng
        angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
        self.last_torso_angle = angle

        # 2. Tính tỷ lệ khung bao H/W
        w = max(float(box[2] - box[0]), 1.0)
        h = max(float(box[3] - box[1]), 1.0)
        aspect_ratio = h / w

        # Điều kiện ngã: Thân người nằm nghiêng góc lớn HOẶC khung bao bị nén bẹp nằm ngang
        is_tilted = angle >= MIN_FALL_TORSO_ANGLE
        is_horizontal_box = aspect_ratio <= MAX_FALL_BBOX_RATIO

        return is_tilted or is_horizontal_box

    def extract_kinetics(self, keypoints: np.ndarray, confs: np.ndarray | None = None) -> np.ndarray:
        """Trích xuất đặc trưng 102 chiều (Vị trí, Vận tốc, Gia tốc)."""
        if self.strict_mode or confs is None:
            hip_center = (keypoints[11] + keypoints[12]) / 2.0
            shifted = keypoints - hip_center

            head_y = float(keypoints[0, 1])
            foot_y = float(max(keypoints[15, 1], keypoints[16, 1]))
            height = max(abs(foot_y - head_y), 1.0)
        else:
            l_hip_ok = confs[11] > LANDMARK_CONF_THRESHOLD
            r_hip_ok = confs[12] > LANDMARK_CONF_THRESHOLD

            if l_hip_ok and r_hip_ok:
                hip_center = (keypoints[11] + keypoints[12]) / 2.0
            elif l_hip_ok:
                hip_center = keypoints[11].copy()
            elif r_hip_ok:
                hip_center = keypoints[12].copy()
            elif self.prev_hip_center is not None:
                hip_center = self.prev_hip_center
            else:
                hip_center = (keypoints[11] + keypoints[12]) / 2.0

            self.prev_hip_center = hip_center
            shifted = keypoints - hip_center

            head_ok = confs[0] > LANDMARK_CONF_THRESHOLD
            foot_l_ok = confs[15] > LANDMARK_CONF_THRESHOLD
            foot_r_ok = confs[16] > LANDMARK_CONF_THRESHOLD

            if head_ok and (foot_l_ok or foot_r_ok):
                head_y = float(keypoints[0, 1])
                foot_candidates = []
                if foot_l_ok:
                    foot_candidates.append(float(keypoints[15, 1]))
                if foot_r_ok:
                    foot_candidates.append(float(keypoints[16, 1]))
                height = max(abs(max(foot_candidates) - head_y), 1.0)
                self.prev_height = height
            elif self.prev_height is not None:
                height = self.prev_height
            else:
                head_y = float(keypoints[0, 1])
                foot_y = float(max(keypoints[15, 1], keypoints[16, 1]))
                height = max(abs(foot_y - head_y), 1.0)
                self.prev_height = height

        current_raw = (shifted / height).reshape(-1).astype(np.float32)

        if self.prev_raw_flat is None:
            velocity = np.zeros(34, dtype=np.float32)
            acceleration = np.zeros(34, dtype=np.float32)
        else:
            velocity = current_raw - self.prev_raw_flat
            acceleration = (
                velocity - self.prev_velocity
                if self.prev_velocity is not None
                else np.zeros(34, dtype=np.float32)
            )

        self.prev_raw_flat = current_raw
        self.prev_velocity = velocity
        return np.concatenate((current_raw, velocity, acceleration)).astype(np.float32)


@dataclass
class CameraState:
    """Quản lý trạng thái đa đối tượng cho từng Session Camera."""
    strict_mode: bool = False
    persons: dict[int, PersonState] = field(default_factory=dict)
    next_track_id: int = 1
    processed_frames: int = 0
    last_seen: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

        return float(inter_area / float(boxA_area + boxB_area - inter_area + 1e-6))

    @staticmethod
    def _center_distance(boxA: np.ndarray, boxB: np.ndarray) -> float:
        cA = np.array([(boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0])
        cB = np.array([(boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0])
        return float(np.linalg.norm(cA - cB))

    def match_and_update_persons(
        self,
        detected_boxes: list[np.ndarray],
        detected_keypoints: list[np.ndarray],
        detected_confs: list[np.ndarray],
    ) -> list[tuple[PersonState, np.ndarray, np.ndarray, np.ndarray]]:
        matched_results = []
        assigned_person_ids = set()
        unmatched_detections = list(range(len(detected_boxes)))

        # 1. Khớp IoU
        for det_idx in list(unmatched_detections):
            box = detected_boxes[det_idx]
            best_iou = 0.10
            best_p_id = None

            for p_id, person in self.persons.items():
                if p_id in assigned_person_ids or person.last_box is None:
                    continue
                iou = self._compute_iou(person.last_box, box)
                if iou > best_iou:
                    best_iou = iou
                    best_p_id = p_id

            if best_p_id is not None:
                assigned_person_ids.add(best_p_id)
                unmatched_detections.remove(det_idx)
                person = self.persons[best_p_id]
                person.last_box = box
                person.missed_frames = 0
                matched_results.append((person, box, detected_keypoints[det_idx], detected_confs[det_idx]))

        # 2. Khớp khoảng cách tâm
        for det_idx in list(unmatched_detections):
            box = detected_boxes[det_idx]
            min_dist = 180.0
            best_p_id = None

            for p_id, person in self.persons.items():
                if p_id in assigned_person_ids or person.last_box is None:
                    continue
                dist = self._center_distance(person.last_box, box)
                if dist < min_dist:
                    min_dist = dist
                    best_p_id = p_id

            if best_p_id is not None:
                assigned_person_ids.add(best_p_id)
                unmatched_detections.remove(det_idx)
                person = self.persons[best_p_id]
                person.last_box = box
                person.missed_frames = 0
                matched_results.append((person, box, detected_keypoints[det_idx], detected_confs[det_idx]))

        # 3. Cấp ID mới
        for det_idx in unmatched_detections:
            p_id = self.next_track_id
            self.next_track_id += 1
            new_person = PersonState(
                track_id=p_id,
                last_box=detected_boxes[det_idx],
                strict_mode=self.strict_mode
            )
            self.persons[p_id] = new_person
            matched_results.append(
                (new_person, detected_boxes[det_idx], detected_keypoints[det_idx], detected_confs[det_idx]))

        # 4. Dọn dẹp đối tượng không còn trong khung hình
        matched_ids = assigned_person_ids | {p.track_id for p, _, _, _ in matched_results}
        stale_ids = []
        for p_id, person in self.persons.items():
            if p_id not in matched_ids:
                person.missed_frames += 1
                if self.strict_mode:
                    person.reset_kinetics()
                else:
                    if person.missed_frames > KINETICS_RESET_AFTER_MISSED_FRAMES:
                        person.reset_kinetics()
                if person.missed_frames > MAX_MISSED_FRAMES:
                    stale_ids.append(p_id)

        for p_id in stale_ids:
            self.persons.pop(p_id, None)

        return matched_results


class FallDetectionService:
    """Dịch vụ dự đoán té ngã thời gian thực tích hợp YOLO-Pose và BiGRU-Attention."""

    def __init__(self, strict_mode: bool = False, pose_filename: str = "yolov8n-pose.pt") -> None:
        root = Path(__file__).resolve().parents[1]
        self.model_dir = root / "ai_models"

        pose_candidates = [
            self.model_dir / pose_filename, 
            self.model_dir / "yolov8n-pose.pt", 
            self.model_dir / "yolo26n-pose.pt"
        ]
        self.pose_path = next((p for p in pose_candidates if p.exists()), self.model_dir / pose_filename)

        classifier_candidates = [
            self.model_dir / "Best_BiGRU_Attention_Model.pth", 
            self.model_dir / "Best BigRU Attention Model.pth"
        ]
        self.classifier_path = next((p for p in classifier_candidates if p.exists()), self.model_dir / "Best_BiGRU_Attention_Model.pth")

        config_candidates = [
            self.model_dir / "Best_BiGRU_Attention_Config.npy", 
            self.model_dir / "Best BigRU Attention Config.npy"
        ]
        self.config_path = next((p for p in config_candidates if p.exists()), self.model_dir / "Best_BiGRU_Attention_Config.npy")

        self.strict_mode = strict_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_model: Any | None = None
        self.classifier: BiGRUAttentionModel | None = None
        self.threshold = 0.4577
        self.hidden_size = 64
        self.num_layers = 2
        self.model_loaded = False
        self.load_error: str | None = None

        self._load_lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._states: dict[str, CameraState] = {}
        self._last_cleanup = 0.0

    def _validate_assets(self) -> None:
        missing = [p for p in (self.pose_path, self.classifier_path, self.config_path) if not p.exists()]
        if missing:
            files = "\n".join(f"- {p}" for p in missing)
            raise FileNotFoundError(f"Thiếu file model bắt buộc:\n{files}")

    def load(self) -> None:
        if self.model_loaded:
            return
        with self._load_lock:
            if self.model_loaded:
                return
            try:
                self._validate_assets()
                config = np.load(self.config_path, allow_pickle=True).item()
                self.hidden_size = int(config.get("hidden_size", 64))
                self.num_layers = int(config.get("num_layers", 2))
                self.threshold = float(config.get("threshold", 0.4577))

                classifier = BiGRUAttentionModel(
                    input_size=INPUT_SIZE,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                ).to(self.device)

                try:
                    state_dict = torch.load(self.classifier_path, map_location=self.device, weights_only=True)
                except TypeError:
                    state_dict = torch.load(self.classifier_path, map_location=self.device)

                classifier.load_state_dict(state_dict)
                classifier.eval()

                for p in classifier.parameters():
                    p.requires_grad_(False)

                from ultralytics import YOLO
                pose_model = YOLO(str(self.pose_path))

                # Warmup inference
                dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
                pose_model.predict(
                    source=dummy, verbose=False, conf=0.25, imgsz=YOLO_IMGSZ,
                    device=0 if self.device.type == "cuda" else "cpu",
                    half=self.device.type == "cuda",
                    classes=0
                )
                with torch.inference_mode():
                    classifier(torch.zeros(1, MAX_LEN, INPUT_SIZE, device=self.device))

                self.classifier = classifier
                self.pose_model = pose_model
                self.model_loaded = True
                self.load_error = None
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                raise

    def health(self, load: bool = False) -> dict[str, Any]:
        if load:
            try:
                self.load()
            except Exception:
                pass
        gpu_name = None
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "CUDA GPU"
        return {
            "ok": self.load_error is None,
            "model_loaded": self.model_loaded,
            "strict_mode": self.strict_mode,
            "load_error": self.load_error,
            "device": str(self.device),
            "gpu_name": gpu_name,
            "active_sessions": len(self._states),
        }

    def _get_state(self, session_id: str) -> CameraState:
        with self._state_lock:
            now = time.time()
            if (now - self._last_cleanup) > CLEANUP_INTERVAL_SECONDS:
                self._last_cleanup = now
                stale = [sid for sid, s in self._states.items() if now - s.last_seen > SESSION_TTL_SECONDS]
                for sid in stale:
                    self._states.pop(sid, None)

            state = self._states.get(session_id)
            if state is None:
                if len(self._states) >= MAX_ACTIVE_SESSIONS:
                    oldest_sid = min(self._states, key=lambda sid: self._states[sid].last_seen)
                    self._states.pop(oldest_sid, None)

                state = CameraState(strict_mode=self.strict_mode)
                self._states[session_id] = state

            state.last_seen = time.time()
            return state

    def reset(self, session_id: str) -> None:
        with self._state_lock:
            self._states.pop(session_id, None)

    @staticmethod
    def _extract_all_persons(result: Any) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        if result is None or getattr(result, "keypoints", None) is None:
            return [], [], []

        xy_tensor = result.keypoints.xy
        boxes_tensor = getattr(result, "boxes", None)
        conf_tensor = getattr(result.keypoints, "conf", None)

        if xy_tensor is None or len(xy_tensor) == 0:
            return [], [], []

        boxes_list, keypoints_list, confs_list = [], [], []

        for idx in range(len(xy_tensor)):
            kp = xy_tensor[idx].detach().cpu().numpy().astype(np.float32)
            if kp.shape != (17, 2) or not np.isfinite(kp).all() or float(np.abs(kp).sum()) == 0.0:
                continue

            box = np.zeros(4, dtype=np.float32)
            if boxes_tensor is not None and getattr(boxes_tensor, "xyxy", None) is not None and len(boxes_tensor.xyxy) > idx:
                box = boxes_tensor.xyxy[idx].detach().cpu().numpy().astype(np.float32)

            conf = None
            if conf_tensor is not None and len(conf_tensor) > idx:
                conf = conf_tensor[idx].detach().cpu().numpy().astype(np.float32)

            boxes_list.append(box)
            keypoints_list.append(kp)
            confs_list.append(conf if conf is not None else np.ones(17, dtype=np.float32))

        return boxes_list, keypoints_list, confs_list

    def predict(self, frame: np.ndarray, session_id: str) -> dict[str, Any]:
        self.load()
        assert self.pose_model is not None and self.classifier is not None

        state = self._get_state(session_id)
        started = time.perf_counter()
        orig_height, orig_width = frame.shape[:2]

        with self._model_lock:
            yolo_device = 0 if self.device.type == "cuda" else "cpu"
            results = self.pose_model.predict(
                source=frame,
                verbose=False,
                conf=0.25,
                imgsz=YOLO_IMGSZ,
                device=yolo_device,
                half=self.device.type == "cuda",
                classes=0,
                max_det=20,
            )
            result = results[0] if results else None
            boxes, keypoints_list, confs_list = self._extract_all_persons(result)

        persons_output: list[dict[str, Any]] = []
        overall_fall = False
        total_fall_count = 0

        with state.lock:
            state.processed_frames += 1
            matched_targets = state.match_and_update_persons(boxes, keypoints_list, confs_list)

            ready_persons: list[PersonState] = []
            ready_sequences: list[np.ndarray] = []

            for person, _, keypoints, confs in matched_targets:
                features = person.extract_kinetics(keypoints, confs)
                person.skeleton_buffer.append(features)

                if len(person.skeleton_buffer) == MAX_LEN:
                    ready_persons.append(person)
                    ready_sequences.append(np.asarray(person.skeleton_buffer, dtype=np.float32))
                else:
                    person.fall_probability = 0.0
                    person.decision_buffer.append(False)

            # Batched Inference
            if ready_sequences:
                batch_tensor = torch.from_numpy(np.stack(ready_sequences, axis=0)).to(self.device)
                with torch.inference_mode():
                    logits = self.classifier(batch_tensor)
                    probabilities = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

                if probabilities.ndim == 0:
                    probabilities = np.array([probabilities.item()])

                for p, prob in zip(ready_persons, probabilities):
                    p.fall_probability = float(prob)

            # Phân tích kết quả kết hợp Bộ lọc hình học (Geometric Gate)
            for person, box, keypoints, confs in matched_targets:
                # 1. Kiểm tra hình học: Góc nghiêng thân hoặc tỷ lệ khung bao nằm sàn
                is_posture_fallen = person.check_geometric_fall(keypoints, box)

                # 2. Gating: Chỉ chấp nhận ngã khi AI dự đoán cao VÀ tư thế cơ thể thực sự bị sụp đổ
                raw_is_fall = (person.fall_probability >= self.threshold) and is_posture_fallen
                person.decision_buffer.append(raw_is_fall)

                vote_ratio = (
                    float(sum(person.decision_buffer) / len(person.decision_buffer))
                    if person.decision_buffer else 0.0
                )
                
                # 3. Quyết định cuối cùng qua Bỏ phiếu thời gian (5/8 frame)
                person.is_fall = (vote_ratio >= VOTE_THRESHOLD) and is_posture_fallen
                if person.is_fall and not person.previous_is_fall:
                    person.fall_count += 1
                person.previous_is_fall = person.is_fall

                if person.is_fall:
                    overall_fall = True
                total_fall_count += person.fall_count

                points = [
                    {"x": float(x), "y": float(y), "conf": float(confs[i]) if confs is not None else 1.0}
                    for i, (x, y) in enumerate(keypoints)
                ]

                persons_output.append({
                    "track_id": person.track_id,
                    "status": "FALL" if person.is_fall else "NORMAL",
                    "is_fall": person.is_fall,
                    "raw_is_fall": raw_is_fall,
                    "fall_probability": round(person.fall_probability, 6),
                    "vote_ratio": round(vote_ratio, 4),
                    "fall_count": person.fall_count,
                    "bbox": [round(float(v), 1) for v in box],
                    "keypoints": points,
                })

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        max_prob = max([p["fall_probability"] for p in persons_output], default=0.0)

        return {
            "ok": True,
            "status": "FALL" if overall_fall else ("NORMAL" if len(persons_output) > 0 else "NO_PERSON"),
            "is_fall": overall_fall,
            "fall_probability": round(max_prob * 100, 2),
            "threshold": round(self.threshold * 100, 2),
            "fall_count": total_fall_count,
            "person_count": len(persons_output),
            "persons": persons_output,
            "processed_frames": state.processed_frames,
            "processing_ms": round(elapsed_ms, 1),
            "server_fps": round(1000.0 / elapsed_ms, 2) if elapsed_ms > 0 else 0.0,
            "image_width": orig_width,
            "image_height": orig_height,
            "connections": COCO_CONNECTIONS,
            "device": "CUDA" if self.device.type == "cuda" else "CPU",
        }


_service: FallDetectionService | None = None
_service_lock = threading.Lock()


def get_model_service(strict_mode: bool = False, pose_filename: str = "yolov8n-pose.pt") -> FallDetectionService:
    """Hàm khởi tạo Singleton cho Service."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = FallDetectionService(strict_mode=strict_mode, pose_filename=pose_filename)
    return _service