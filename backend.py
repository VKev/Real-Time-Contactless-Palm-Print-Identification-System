import asyncio
import base64
import heapq
import importlib
import inspect
import importlib.util
import json
import os
import queue
import threading
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from qdrant_client.models import Distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from roi_extraction import extract_palm_roi
from utils.preprocess import preprocess
from utils.qdrant import QdrantHelper
from utils.triton import TritonClient

ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = ROOT / "frontend" / "dist"

CAPTURE_WIDTH = int(os.getenv("CAPTURE_WIDTH", "1280"))
CAPTURE_HEIGHT = int(os.getenv("CAPTURE_HEIGHT", "720"))
ROI_DISPLAY_SIZE = int(os.getenv("ROI_DISPLAY_SIZE", "224"))
ROI_PROCESS_MAX_WIDTH = int(os.getenv("ROI_PROCESS_MAX_WIDTH", "0"))
CAMERA_STALE_SECONDS = float(os.getenv("CAMERA_STALE_SECONDS", "2.0"))
UPLOAD_TARGET_FPS = int(os.getenv("UPLOAD_TARGET_FPS", "24"))
OUTPUT_PNG_COMPRESSION = int(os.getenv("OUTPUT_PNG_COMPRESSION", "1"))
TRITON_GRPC_URL = os.getenv("TRITON_GRPC_URL", "localhost:8001")
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "feature_extraction")
TRITON_INPUT_NAME = os.getenv("TRITON_INPUT_NAME", "INPUT__0")
TRITON_OUTPUT_NAME = os.getenv("TRITON_OUTPUT_NAME", "OUTPUT__0")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "palm_vectors")
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "EUCLID").upper()
FEATURE_VECTOR_SIZE = int(os.getenv("FEATURE_VECTOR_SIZE", "128"))
REGISTER_KEEP_ALL_SAMPLES = os.getenv("REGISTER_KEEP_ALL_SAMPLES", "1") == "1"
REGISTER_QUEUE_MAXSIZE = int(os.getenv("REGISTER_QUEUE_MAXSIZE", "512"))
VERIFY_QUEUE_MAXSIZE = int(os.getenv("VERIFY_QUEUE_MAXSIZE", "512"))
VERIFY_TOP_K_DEFAULT = int(os.getenv("VERIFY_TOP_K_DEFAULT", "10"))
VERIFY_THRESHOLD_DEFAULT = float(os.getenv("VERIFY_THRESHOLD_DEFAULT", "35.0"))
PLOT_MAX_POINTS = int(os.getenv("PLOT_MAX_POINTS", "0"))
PLOT_QUEUE_MAXSIZE = int(os.getenv("PLOT_QUEUE_MAXSIZE", "512"))
PROJECTION_REFRESH_QUEUE_MAXSIZE = int(os.getenv("PROJECTION_REFRESH_QUEUE_MAXSIZE", "8"))
MANIFOLD_PREP_COMPONENTS = int(os.getenv("MANIFOLD_PREP_COMPONENTS", "32"))
UMAP_EPOCHS = int(os.getenv("UMAP_EPOCHS", "400"))
TSNE_ITERATIONS = int(os.getenv("TSNE_ITERATIONS", "700"))
TSNE_ANGLE = float(os.getenv("TSNE_ANGLE", "0.6"))
PLOT_PREFER_GPU = os.getenv("PLOT_PREFER_GPU", "1") == "1"
PROJECTION_GPU_ENGINE = os.getenv("PROJECTION_GPU_ENGINE", "auto").strip().lower()


@dataclass(slots=True)
class EncodedFrame:
    frame_id: int
    payload: bytes
    timestamp: float


@dataclass(slots=True)
class DecodedFrame:
    frame_id: int
    frame: np.ndarray
    timestamp: float


@dataclass(slots=True)
class RoiStageResult:
    frame_id: int
    roi_image: np.ndarray | None
    processing_ms: float


@dataclass(slots=True)
class RegistrationTask:
    frame_id: int
    target_id: int
    batch: np.ndarray


@dataclass(slots=True)
class VerifyTask:
    frame_id: int
    threshold: float
    top_k: int
    batch: np.ndarray


@dataclass(slots=True)
class PlotTask:
    point_id: str
    subject_id: str
    embedding: np.ndarray


@dataclass(slots=True)
class ProjectionRefreshTask:
    kind: str


def _load_optional_projection_class(module_name: str, class_name: str):
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, class_name, None)


def _load_optional_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


GPU_UMAP_CLASS = (
    _load_optional_projection_class("cuml", "UMAP")
    or _load_optional_projection_class("cuml.manifold", "UMAP")
)
GPU_TSNE_CLASS = (
    _load_optional_projection_class("cuml", "TSNE")
    or _load_optional_projection_class("cuml.manifold", "TSNE")
)
TORCHDR_UMAP_CLASS = _load_optional_projection_class("torchdr", "UMAP")
TORCHDR_TSNE_CLASS = _load_optional_projection_class("torchdr", "TSNE")
TORCH_MODULE = _load_optional_module("torch")


def _print_projection_backend(kind: str, backend: str, sample_count: int, detail: str | None = None):
    message = f"[projection] {kind.upper()} using {backend} for {int(sample_count)} points"
    if detail:
        message = f"{message} ({detail})"
    print(message, flush=True)


def _torch_cuda_available() -> bool:
    if TORCH_MODULE is None:
        return False
    try:
        return bool(TORCH_MODULE.cuda.is_available())
    except Exception:
        return False


def _projection_gpu_preference_order() -> list[str]:
    if PROJECTION_GPU_ENGINE in {"torchdr", "cuml", "cpu"}:
        return [PROJECTION_GPU_ENGINE]

    if os.name == "nt":
        return ["torchdr", "cuml"]

    return ["cuml", "torchdr"]


def _to_numpy_projection(coords) -> np.ndarray:
    array = coords
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "numpy"):
        array = array.numpy()
    return np.asarray(array, dtype=np.float32)


class LatestItemMailbox:
    def __init__(self):
        self._condition = threading.Condition()
        self._item = None
        self._version = 0
        self._closed = False

    def put(self, item):
        with self._condition:
            if self._closed:
                return self._version
            self._item = item
            self._version += 1
            self._condition.notify_all()
            return self._version

    def clear(self):
        with self._condition:
            self._item = None
            self._version += 1
            self._condition.notify_all()

    def get_after(self, last_version: int, shutdown_event: threading.Event, wait_timeout: float = 0.05):
        with self._condition:
            while self._version <= last_version and not self._closed and not shutdown_event.is_set():
                self._condition.wait(wait_timeout)

            if self._closed or shutdown_event.is_set():
                return None, last_version

            return self._item, self._version

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()


def _placeholder(text: str, width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        image,
        text,
        (18, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), OUTPUT_PNG_COMPRESSION])
    if not ok or buffer is None:
        raise RuntimeError("Failed to encode image")
    return buffer.tobytes()


def _resize_to_display(image: np.ndarray) -> np.ndarray:
    if image.shape[:2] == (ROI_DISPLAY_SIZE, ROI_DISPLAY_SIZE):
        return image

    height, width = image.shape[:2]
    interpolation = cv2.INTER_AREA
    if height < ROI_DISPLAY_SIZE or width < ROI_DISPLAY_SIZE:
        interpolation = cv2.INTER_CUBIC

    return cv2.resize(image, (ROI_DISPLAY_SIZE, ROI_DISPLAY_SIZE), interpolation=interpolation)


def _resize_for_processing(frame: np.ndarray) -> np.ndarray:
    if ROI_PROCESS_MAX_WIDTH <= 0:
        return frame

    height, width = frame.shape[:2]
    if width <= ROI_PROCESS_MAX_WIDTH:
        return frame

    scale = ROI_PROCESS_MAX_WIDTH / float(width)
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (ROI_PROCESS_MAX_WIDTH, resized_height), interpolation=cv2.INTER_AREA)


def _mark_latest_frame_received(app_instance: FastAPI, payload: bytes) -> int:
    with app_instance.state.frame_lock:
        app_instance.state.latest_frame_id += 1
        frame_id = app_instance.state.latest_frame_id
        frame_ts = time.time()
        app_instance.state.latest_frame_ts = frame_ts

    app_instance.state.input_mailbox.put(EncodedFrame(frame_id=frame_id, payload=payload, timestamp=frame_ts))
    return frame_id


def _clear_pipeline(app_instance: FastAPI):
    with app_instance.state.frame_lock:
        app_instance.state.latest_frame_ts = 0.0
    app_instance.state.input_mailbox.clear()
    app_instance.state.roi_mailbox.clear()
    app_instance.state.preprocess_mailbox.clear()


def _payload_to_data_url(payload: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"


def _parse_qdrant_distance(name: str) -> Distance:
    normalized = str(name).strip().upper()
    mapping = {
        "COSINE": Distance.COSINE,
        "EUCLID": Distance.EUCLID,
        "EUCLIDEAN": Distance.EUCLID,
        "EUCLEDIENT": Distance.EUCLID,
        "L2": Distance.EUCLID,
        "DOT": Distance.DOT,
        "MANHATTAN": Distance.MANHATTAN,
    }
    return mapping.get(normalized, Distance.EUCLID)


def _projection_label(kind: str) -> str:
    normalized = str(kind).strip().lower()
    if normalized == "pca":
        return "PCA"
    if normalized == "umap":
        return "UMAP"
    if normalized == "tsne":
        return "t-SNE"
    return normalized.upper()


def _construct_estimator(cls, **kwargs):
    try:
        parameters = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return cls(**kwargs)

    supported_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return cls(**supported_kwargs)


def _preprocess_roi_for_registration(roi_image: np.ndarray):
    try:
        batch, resized = preprocess(roi_image)
    except Exception:
        return None, None

    if batch is None or resized is None or batch.size == 0 or resized.size == 0:
        return None, None

    display = resized
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    return batch.astype(np.float32, copy=False), _resize_to_display(display)


def _init_backends(app_instance: FastAPI):
    app_instance.state.triton_client = None
    app_instance.state.triton_error = None
    app_instance.state.qdrant_client = None
    app_instance.state.qdrant_error = None

    try:
        triton_client = TritonClient(url=TRITON_GRPC_URL)
        if not triton_client.client.is_server_live():
            raise RuntimeError(f"Triton at '{TRITON_GRPC_URL}' is not live")
        if not triton_client.client.is_server_ready():
            raise RuntimeError(f"Triton at '{TRITON_GRPC_URL}' is not ready")
        if not triton_client.client.is_model_ready(TRITON_MODEL_NAME):
            raise RuntimeError(f"Model '{TRITON_MODEL_NAME}' is not ready")
        triton_client.client.get_model_metadata(model_name=TRITON_MODEL_NAME)
        app_instance.state.triton_client = triton_client
    except Exception as exc:
        app_instance.state.triton_error = str(exc)

    try:
        qdrant_client = QdrantHelper(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            grpc_port=QDRANT_GRPC_PORT,
            prefer_grpc=True,
        )
        qdrant_client.ensure_collection(
            collection_name=QDRANT_COLLECTION,
            vector_size=FEATURE_VECTOR_SIZE,
            distance=_parse_qdrant_distance(QDRANT_DISTANCE),
        )
        app_instance.state.qdrant_client = qdrant_client
        try:
            _load_plot_points_from_qdrant(app_instance)
        except Exception as exc:
            app_instance.state.qdrant_error = f"PCA bootstrap failed: {exc}"
    except Exception as exc:
        app_instance.state.qdrant_error = str(exc)


def _infer_embedding_from_batch(app_instance: FastAPI, batch: np.ndarray):
    if app_instance.state.triton_client is None:
        raise RuntimeError(app_instance.state.triton_error or "Triton is not connected")

    prepared_batch = np.asarray(batch, dtype=np.float32)
    outputs = app_instance.state.triton_client.infer(
        model_name=TRITON_MODEL_NAME,
        inputs={TRITON_INPUT_NAME: prepared_batch},
        outputs=[TRITON_OUTPUT_NAME],
    )
    embedding = outputs.get(TRITON_OUTPUT_NAME)
    if embedding is None or np.asarray(embedding).size == 0:
        raise RuntimeError("Triton returned an empty embedding")

    return np.asarray(embedding, dtype=np.float32).reshape(-1)


def _extract_qdrant_vector(point) -> np.ndarray | None:
    vector = getattr(point, "vector", None)
    if vector is None:
        return None

    if isinstance(vector, dict):
        if not vector:
            return None
        vector = next(iter(vector.values()))

    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _load_plot_points_from_qdrant(app_instance: FastAPI):
    helper = app_instance.state.qdrant_client
    if helper is None:
        return

    records: list[tuple[float, str, str, np.ndarray]] = []
    offset = None
    batch_size = 256

    while True:
        points, offset = helper.client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        for point in points:
            vector = _extract_qdrant_vector(point)
            if vector is None:
                continue

            payload = point.payload or {}
            subject_id = str(payload.get("subject_id", payload.get("label", point.id)))
            updated_at_raw = payload.get("updated_at", 0.0)
            try:
                updated_at = float(updated_at_raw)
            except (TypeError, ValueError):
                updated_at = 0.0

            records.append((updated_at, str(point.id), subject_id, vector.astype(np.float32, copy=True)))

        if offset is None:
            break

    records.sort(key=lambda item: (item[0], item[1]))
    if PLOT_MAX_POINTS > 0 and len(records) > PLOT_MAX_POINTS:
        records = records[-PLOT_MAX_POINTS:]

    with app_instance.state.plot_lock:
        app_instance.state.plot_point_ids = [point_id for _, point_id, _, _ in records]
        app_instance.state.plot_subject_ids = [subject_id for _, _, subject_id, _ in records]
        app_instance.state.plot_vectors = [vector for _, _, _, vector in records]
        app_instance.state.plot_latest_point_id = records[-1][1] if records else None
        app_instance.state.plot_points = []
        app_instance.state.plot_version = 0
        app_instance.state.plot_pending_count = 0

    _recompute_plot_projection(app_instance)


def _register_embedding(app_instance: FastAPI, target_id: int, embedding: np.ndarray):
    if app_instance.state.qdrant_client is None:
        raise RuntimeError(app_instance.state.qdrant_error or "Qdrant is not connected")

    vector = np.asarray(embedding, dtype=np.float32).reshape(-1).tolist()
    point_id = time.time_ns()
    if not REGISTER_KEEP_ALL_SAMPLES:
        point_id = int(target_id)

    payload = {
        "label": str(target_id),
        "subject_id": str(target_id),
        "updated_at": time.time(),
    }

    app_instance.state.qdrant_client.insert_vectors(
        collection_name=QDRANT_COLLECTION,
        vectors=[vector],
        ids=[point_id],
        payloads=[payload],
    )

    with app_instance.state.register_lock:
        app_instance.state.register_insert_count += 1
        app_instance.state.register_last_error = None

    _enqueue_plot_task(
        app_instance,
        PlotTask(
            point_id=str(point_id),
            subject_id=str(target_id),
            embedding=np.asarray(embedding, dtype=np.float32).reshape(-1).copy(),
        ),
    )

    _queue_broadcast_event(app_instance, "register")


def _set_register_error(app_instance: FastAPI, message: str):
    with app_instance.state.register_lock:
        app_instance.state.register_last_error = message
    _queue_broadcast_event(app_instance, "register")


def _clear_register_error(app_instance: FastAPI):
    with app_instance.state.register_lock:
        app_instance.state.register_last_error = None


def _set_verify_status(app_instance: FastAPI, message: str, top_matches: list[dict] | None = None):
    with app_instance.state.register_lock:
        changed = (
            app_instance.state.verify_last_result != message
            or app_instance.state.verify_last_error is not None
            or (top_matches is not None and app_instance.state.verify_top_matches != top_matches)
        )
        app_instance.state.verify_last_result = message
        app_instance.state.verify_last_error = None
        if top_matches is not None:
            app_instance.state.verify_top_matches = top_matches

    if changed:
        _queue_broadcast_event(app_instance, "verify")


def _set_verify_error(app_instance: FastAPI, message: str, top_matches: list[dict] | None = None):
    with app_instance.state.register_lock:
        changed = (
            app_instance.state.verify_last_error != message
            or app_instance.state.verify_last_result != message
            or (top_matches is not None and app_instance.state.verify_top_matches != top_matches)
        )
        app_instance.state.verify_last_error = message
        app_instance.state.verify_last_result = message
        if top_matches is not None:
            app_instance.state.verify_top_matches = top_matches

    if changed:
        _queue_broadcast_event(app_instance, "verify")


def _clear_verify_error(app_instance: FastAPI):
    with app_instance.state.register_lock:
        app_instance.state.verify_last_error = None


def _clear_registration_queue(app_instance: FastAPI):
    while True:
        try:
            app_instance.state.register_queue.get_nowait()
            with app_instance.state.register_lock:
                app_instance.state.register_pending_count = max(0, app_instance.state.register_pending_count - 1)
        except queue.Empty:
            break


def _clear_verification_queue(app_instance: FastAPI):
    while True:
        try:
            app_instance.state.verify_queue.get_nowait()
            with app_instance.state.register_lock:
                app_instance.state.verify_pending_count = max(0, app_instance.state.verify_pending_count - 1)
        except queue.Empty:
            break


def _clear_plot_queue(app_instance: FastAPI):
    while True:
        try:
            app_instance.state.plot_queue.get_nowait()
            with app_instance.state.plot_lock:
                app_instance.state.plot_pending_count = max(0, app_instance.state.plot_pending_count - 1)
        except queue.Empty:
            break


def _clear_projection_refresh_queue(app_instance: FastAPI):
    while True:
        try:
            task = app_instance.state.projection_refresh_queue.get_nowait()
            kind = str(task.kind).strip().lower()
            with app_instance.state.plot_lock:
                if kind in app_instance.state.manual_projections:
                    app_instance.state.manual_projections[kind]["pending_count"] = max(
                        0, int(app_instance.state.manual_projections[kind]["pending_count"]) - 1
                    )
        except queue.Empty:
            break


def _update_register_target(app_instance: FastAPI, target_id: int):
    if app_instance.state.triton_client is None or app_instance.state.qdrant_client is None:
        _set_register_error(app_instance, "Register unavailable: Triton or Qdrant not connected.")
        return

    _clear_registration_queue(app_instance)
    with app_instance.state.register_lock:
        app_instance.state.register_target_id = int(target_id)
        app_instance.state.register_pending_count = 0
        app_instance.state.register_last_error = None

    _queue_broadcast_event(app_instance, "register")


def _get_register_target(app_instance: FastAPI):
    with app_instance.state.register_lock:
        if not app_instance.state.register_enabled:
            return None
        target_id = app_instance.state.register_target_id

    if target_id is None:
        return None

    if app_instance.state.triton_client is None or app_instance.state.qdrant_client is None:
        _set_register_error(app_instance, "Register unavailable: Triton or Qdrant not connected.")
        return None

    return int(target_id)


def _get_verify_config(app_instance: FastAPI):
    with app_instance.state.register_lock:
        if not app_instance.state.verify_enabled:
            return None
        threshold = float(app_instance.state.verify_threshold)
        top_k = int(app_instance.state.verify_top_k)

    if app_instance.state.triton_client is None or app_instance.state.qdrant_client is None:
        _set_verify_error(app_instance, "Verify unavailable: Triton or Qdrant not connected.")
        return None

    return max(0.0, threshold), max(1, top_k)


def _enqueue_register_task(app_instance: FastAPI, task: RegistrationTask):
    inserted = False

    while not inserted:
        try:
            app_instance.state.register_queue.put_nowait(task)
            inserted = True
            with app_instance.state.register_lock:
                app_instance.state.register_pending_count += 1
                app_instance.state.register_last_error = None
        except queue.Full:
            try:
                app_instance.state.register_queue.get_nowait()
                with app_instance.state.register_lock:
                    app_instance.state.register_pending_count = max(0, app_instance.state.register_pending_count - 1)
                    app_instance.state.register_last_error = "Register queue full. Dropped the oldest pending frame."
            except queue.Empty:
                break

    _queue_broadcast_event(app_instance, "register")


def _enqueue_verify_task(app_instance: FastAPI, task: VerifyTask):
    inserted = False

    while not inserted:
        try:
            app_instance.state.verify_queue.put_nowait(task)
            inserted = True
            with app_instance.state.register_lock:
                app_instance.state.verify_pending_count += 1
                app_instance.state.verify_last_error = None
        except queue.Full:
            try:
                app_instance.state.verify_queue.get_nowait()
                with app_instance.state.register_lock:
                    app_instance.state.verify_pending_count = max(0, app_instance.state.verify_pending_count - 1)
                    app_instance.state.verify_last_error = "Verify queue full. Dropped the oldest pending frame."
                    app_instance.state.verify_last_result = app_instance.state.verify_last_error
            except queue.Empty:
                break

    _queue_broadcast_event(app_instance, "verify")


def _sanitize_plot_records(point_ids: list[str], subject_ids: list[str], vectors: list[np.ndarray]):
    if not point_ids or not subject_ids or not vectors:
        return [], [], np.empty((0, 2), dtype=np.float32)

    valid_point_ids: list[str] = []
    valid_subject_ids: list[str] = []
    valid_vectors: list[np.ndarray] = []
    dim = None

    for point_id, subject_id, vec in zip(point_ids, subject_ids, vectors):
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            continue
        if dim is None:
            dim = arr.size
        if arr.size != dim:
            continue
        valid_point_ids.append(str(point_id))
        valid_subject_ids.append(str(subject_id))
        valid_vectors.append(arr)

    if not valid_vectors:
        return [], [], np.empty((0, 2), dtype=np.float32)

    return (
        valid_point_ids,
        valid_subject_ids,
        np.vstack(valid_vectors).astype(np.float32, copy=False),
    )


def _snapshot_plot_records(app_instance: FastAPI):
    with app_instance.state.plot_lock:
        point_ids = list(app_instance.state.plot_point_ids)
        subject_ids = list(app_instance.state.plot_subject_ids)
        vectors = [vec.copy() for vec in app_instance.state.plot_vectors]
        latest_point_id = app_instance.state.plot_latest_point_id

    point_ids, subject_ids, X = _sanitize_plot_records(point_ids, subject_ids, vectors)
    return point_ids, subject_ids, X, latest_point_id


def _build_projection_points(
    point_ids: list[str], subject_ids: list[str], coords: np.ndarray, latest_point_id: str | None
) -> list[dict]:
    points: list[dict] = []
    for point_id, subject_id, coord in zip(point_ids, subject_ids, coords):
        points.append(
            {
                "id": str(point_id),
                "subject_id": str(subject_id),
                "x": float(coord[0]),
                "y": float(coord[1]),
                "latest": str(point_id) == str(latest_point_id),
            }
        )
    return points


def _canonicalize_pca_components(components: np.ndarray) -> np.ndarray:
    comps = np.asarray(components, dtype=np.float32).copy()
    if comps.ndim != 2:
        return np.empty((0, 0), dtype=np.float32)

    for i in range(comps.shape[0]):
        row = comps[i]
        if row.size == 0:
            continue
        anchor_idx = int(np.argmax(np.abs(row)))
        if row[anchor_idx] < 0:
            comps[i] = -row

    return comps


def _project_pca_canonical(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    n_components = min(2, X.shape[0], X.shape[1])
    if n_components < 1:
        return np.empty((0, 2), dtype=np.float32)

    pca = PCA(n_components=n_components, random_state=42, svd_solver="full")
    points = np.asarray(pca.fit_transform(X), dtype=np.float32)
    raw_components = np.asarray(pca.components_, dtype=np.float32)
    canonical_components = _canonicalize_pca_components(raw_components)

    for i in range(min(points.shape[1], canonical_components.shape[0], raw_components.shape[0])):
        if float(np.dot(raw_components[i], canonical_components[i])) < 0.0:
            points[:, i] *= -1.0

    if points.shape[1] == 1:
        points = np.hstack([points, np.zeros((points.shape[0], 1), dtype=np.float32)])

    if not np.all(np.isfinite(points)):
        raise ValueError("PCA produced non-finite coordinates")

    return points[:, :2].astype(np.float32, copy=False)


def _project_small_sample(X: np.ndarray) -> np.ndarray:
    if X.shape[0] <= 1:
        return np.zeros((X.shape[0], 2), dtype=np.float32)
    return _project_pca_canonical(X)


def _prepare_for_manifold(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)

    max_components = min(MANIFOLD_PREP_COMPONENTS, X.shape[0], X.shape[1])
    if max_components < 2 or X.shape[1] <= max_components:
        return X

    reducer = PCA(n_components=max_components, random_state=42, svd_solver="randomized")
    reduced = np.asarray(reducer.fit_transform(X), dtype=np.float32)
    if reduced.ndim != 2 or reduced.shape[0] != X.shape[0]:
        raise ValueError("Manifold pre-reduction produced invalid coordinates")
    return reduced


def _project_umap(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    if X.shape[0] <= 2:
        return _project_small_sample(X)
    X = _prepare_for_manifold(X)
    gpu_fallback_reasons: list[str] = []

    if PLOT_PREFER_GPU:
        for engine in _projection_gpu_preference_order():
            if engine == "cpu":
                gpu_fallback_reasons.append("GPU disabled by PROJECTION_GPU_ENGINE=cpu")
                break

            if engine == "torchdr":
                if TORCHDR_UMAP_CLASS is None:
                    gpu_fallback_reasons.append("TorchDR UMAP not installed")
                    continue
                if not _torch_cuda_available():
                    gpu_fallback_reasons.append("PyTorch CUDA not available")
                    continue

                try:
                    reducer = _construct_estimator(
                        TORCHDR_UMAP_CLASS,
                        n_components=2,
                        n_neighbors=max(2, min(15, X.shape[0] - 1)),
                        min_dist=0.18,
                        metric="euclidean",
                        random_state=42,
                        n_epochs=UMAP_EPOCHS,
                        max_iter=UMAP_EPOCHS,
                        device="cuda",
                        backend=None,
                        compile=False,
                        verbose=False,
                    )
                    coords = _to_numpy_projection(reducer.fit_transform(X))
                    if coords.ndim != 2 or coords.shape[0] != X.shape[0]:
                        raise ValueError("TorchDR UMAP produced invalid coordinates")
                    if coords.shape[1] == 1:
                        coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)])
                    if not np.all(np.isfinite(coords)):
                        raise ValueError("TorchDR UMAP produced non-finite coordinates")
                    _print_projection_backend("umap", "GPU", X.shape[0], "TorchDR / PyTorch CUDA")
                    return coords[:, :2].astype(np.float32, copy=False)
                except Exception as exc:
                    gpu_fallback_reasons.append(f"TorchDR failed: {exc.__class__.__name__}: {exc}")
                    continue

            if engine == "cuml":
                if GPU_UMAP_CLASS is None:
                    gpu_fallback_reasons.append("cuML UMAP not installed")
                    continue

                try:
                    reducer = _construct_estimator(
                        GPU_UMAP_CLASS,
                        n_components=2,
                        n_neighbors=max(2, min(15, X.shape[0] - 1)),
                        min_dist=0.18,
                        metric="euclidean",
                        n_epochs=UMAP_EPOCHS,
                        random_state=42,
                        output_type="numpy",
                    )
                    coords = np.asarray(reducer.fit_transform(X), dtype=np.float32)
                    if coords.ndim != 2 or coords.shape[0] != X.shape[0]:
                        raise ValueError("GPU UMAP produced invalid coordinates")
                    if coords.shape[1] == 1:
                        coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)])
                    if not np.all(np.isfinite(coords)):
                        raise ValueError("GPU UMAP produced non-finite coordinates")
                    _print_projection_backend("umap", "GPU", X.shape[0], "cuML")
                    return coords[:, :2].astype(np.float32, copy=False)
                except Exception as exc:
                    gpu_fallback_reasons.append(f"cuML failed: {exc.__class__.__name__}: {exc}")
                    continue

    try:
        from umap import UMAP
    except Exception as exc:
        raise RuntimeError("UMAP is not installed in the backend environment.") from exc

    n_neighbors = max(2, min(15, X.shape[0] - 1))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.18,
        metric="euclidean",
        n_epochs=UMAP_EPOCHS,
        low_memory=True,
        random_state=42,
    )
    coords = np.asarray(reducer.fit_transform(X), dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] != X.shape[0]:
        raise ValueError("UMAP produced invalid coordinates")
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)])
    if not np.all(np.isfinite(coords)):
        raise ValueError("UMAP produced non-finite coordinates")
    cpu_detail = "umap-learn"
    if gpu_fallback_reasons:
        cpu_detail = "GPU fallback: " + " | ".join(gpu_fallback_reasons)
    _print_projection_backend("umap", "CPU", X.shape[0], cpu_detail)
    return coords[:, :2].astype(np.float32, copy=False)


def _project_tsne(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    if X.shape[0] <= 2:
        return _project_small_sample(X)
    X = _prepare_for_manifold(X)
    gpu_fallback_reasons: list[str] = []

    perplexity = min(30.0, max(1.0, float(X.shape[0] - 1) / 3.0))
    if perplexity >= float(X.shape[0]):
        perplexity = max(1.0, float(X.shape[0] - 1))

    if PLOT_PREFER_GPU:
        for engine in _projection_gpu_preference_order():
            if engine == "cpu":
                gpu_fallback_reasons.append("GPU disabled by PROJECTION_GPU_ENGINE=cpu")
                break

            if engine == "torchdr":
                if TORCHDR_TSNE_CLASS is None:
                    gpu_fallback_reasons.append("TorchDR TSNE not installed")
                    continue
                if not _torch_cuda_available():
                    gpu_fallback_reasons.append("PyTorch CUDA not available")
                    continue

                try:
                    reducer = _construct_estimator(
                        TORCHDR_TSNE_CLASS,
                        n_components=2,
                        perplexity=perplexity,
                        learning_rate="auto",
                        init="pca",
                        random_state=42,
                        n_iter=TSNE_ITERATIONS,
                        max_iter=TSNE_ITERATIONS,
                        device="cuda",
                        backend=None,
                        compile=False,
                        verbose=False,
                    )
                    coords = _to_numpy_projection(reducer.fit_transform(X))
                    if coords.ndim != 2 or coords.shape[0] != X.shape[0]:
                        raise ValueError("TorchDR t-SNE produced invalid coordinates")
                    if coords.shape[1] == 1:
                        coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)])
                    if not np.all(np.isfinite(coords)):
                        raise ValueError("TorchDR t-SNE produced non-finite coordinates")
                    _print_projection_backend("tsne", "GPU", X.shape[0], "TorchDR / PyTorch CUDA")
                    return coords[:, :2].astype(np.float32, copy=False)
                except Exception as exc:
                    gpu_fallback_reasons.append(f"TorchDR failed: {exc.__class__.__name__}: {exc}")
                    continue

            if engine == "cuml":
                if GPU_TSNE_CLASS is None:
                    gpu_fallback_reasons.append("cuML t-SNE not installed")
                    continue

                try:
                    reducer = _construct_estimator(
                        GPU_TSNE_CLASS,
                        n_components=2,
                        perplexity=perplexity,
                        learning_rate="auto",
                        init="pca",
                        method="barnes_hut",
                        angle=TSNE_ANGLE,
                        random_state=42,
                        n_iter=TSNE_ITERATIONS,
                        max_iter=TSNE_ITERATIONS,
                        output_type="numpy",
                    )
                    coords = np.asarray(reducer.fit_transform(X), dtype=np.float32)
                    if coords.ndim != 2 or coords.shape[0] != X.shape[0]:
                        raise ValueError("GPU t-SNE produced invalid coordinates")
                    if coords.shape[1] == 1:
                        coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)])
                    if not np.all(np.isfinite(coords)):
                        raise ValueError("GPU t-SNE produced non-finite coordinates")
                    _print_projection_backend("tsne", "GPU", X.shape[0], "cuML")
                    return coords[:, :2].astype(np.float32, copy=False)
                except Exception as exc:
                    gpu_fallback_reasons.append(f"cuML failed: {exc.__class__.__name__}: {exc}")
                    continue

    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "learning_rate": "auto",
        "init": "pca",
        "method": "barnes_hut",
        "angle": TSNE_ANGLE,
        "random_state": 42,
    }
    tsne_signature = inspect.signature(TSNE.__init__)
    if "max_iter" in tsne_signature.parameters:
        tsne_kwargs["max_iter"] = TSNE_ITERATIONS
    else:
        tsne_kwargs["n_iter"] = TSNE_ITERATIONS

    reducer = TSNE(
        **tsne_kwargs,
    )
    coords = np.asarray(reducer.fit_transform(X), dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] != X.shape[0]:
        raise ValueError("t-SNE produced invalid coordinates")
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)])
    if not np.all(np.isfinite(coords)):
        raise ValueError("t-SNE produced non-finite coordinates")
    cpu_detail = "scikit-learn"
    if gpu_fallback_reasons:
        cpu_detail = "GPU fallback: " + " | ".join(gpu_fallback_reasons)
    _print_projection_backend("tsne", "CPU", X.shape[0], cpu_detail)
    return coords[:, :2].astype(np.float32, copy=False)


def _recompute_plot_projection(app_instance: FastAPI):
    point_ids, subject_ids, X, latest_point_id = _snapshot_plot_records(app_instance)

    if X.shape[0] == 1:
        coords = np.array([[0.0, 0.0]], dtype=np.float32)
    elif X.shape[0] > 1:
        coords = _project_pca_canonical(X)
    else:
        coords = np.empty((0, 2), dtype=np.float32)

    plot_points = _build_projection_points(point_ids, subject_ids, coords, latest_point_id)

    with app_instance.state.plot_lock:
        app_instance.state.plot_point_ids = point_ids
        app_instance.state.plot_subject_ids = subject_ids
        app_instance.state.plot_vectors = [row.astype(np.float32, copy=True) for row in X] if X.size else []
        app_instance.state.plot_points = plot_points
        app_instance.state.plot_version += 1
        app_instance.state.plot_updated_at = time.time()


def _build_plot_payload(app_instance: FastAPI):
    with app_instance.state.plot_lock:
        return {
            "kind": "pca",
            "version": int(app_instance.state.plot_version),
            "count": len(app_instance.state.plot_points),
            "max_points": PLOT_MAX_POINTS,
            "pending_count": int(app_instance.state.plot_pending_count),
            "status": "Realtime PCA updates as new registrations arrive.",
            "error": None,
            "refreshed_at": float(app_instance.state.plot_updated_at),
            "points": [dict(point) for point in app_instance.state.plot_points],
        }


def _current_plot_count(app_instance: FastAPI) -> int:
    with app_instance.state.plot_lock:
        return len(app_instance.state.plot_point_ids)


def _reset_manual_projection_cache(app_instance: FastAPI, bump_version: bool):
    point_count = _current_plot_count(app_instance)
    with app_instance.state.plot_lock:
        for kind in ("umap", "tsne"):
            projection = app_instance.state.manual_projections[kind]
            projection["points"] = []
            projection["count"] = 0
            projection["pending_count"] = 0
            projection["error"] = None
            projection["refreshed_at"] = 0.0
            projection["status"] = (
                f"Refresh to compute {_projection_label(kind)} for {point_count} registered vectors."
                if point_count > 0
                else "No registered vectors available."
            )
            if bump_version:
                projection["version"] += 1


def _build_manual_projection_payload(app_instance: FastAPI, kind: str):
    with app_instance.state.plot_lock:
        projection = dict(app_instance.state.manual_projections[kind])
        return {
            "kind": kind,
            "version": int(projection["version"]),
            "count": int(projection["count"]),
            "max_points": PLOT_MAX_POINTS,
            "pending_count": int(projection["pending_count"]),
            "status": str(projection["status"]),
            "error": projection["error"],
            "refreshed_at": float(projection["refreshed_at"]),
            "points": [dict(point) for point in projection["points"]],
        }


def _reset_plot_state(app_instance: FastAPI):
    with app_instance.state.plot_lock:
        app_instance.state.plot_point_ids = []
        app_instance.state.plot_subject_ids = []
        app_instance.state.plot_vectors = []
        app_instance.state.plot_points = []
        app_instance.state.plot_latest_point_id = None
        app_instance.state.plot_pending_count = 0
        app_instance.state.plot_version += 1
        app_instance.state.plot_updated_at = time.time()

    _reset_manual_projection_cache(app_instance, bump_version=True)


def _enqueue_plot_task(app_instance: FastAPI, task: PlotTask):
    inserted = False

    while not inserted:
        try:
            app_instance.state.plot_queue.put_nowait(task)
            inserted = True
            with app_instance.state.plot_lock:
                app_instance.state.plot_pending_count += 1
        except queue.Full:
            try:
                app_instance.state.plot_queue.get_nowait()
                with app_instance.state.plot_lock:
                    app_instance.state.plot_pending_count = max(0, app_instance.state.plot_pending_count - 1)
            except queue.Empty:
                break


def _enqueue_projection_refresh(app_instance: FastAPI, task: ProjectionRefreshTask):
    kind = str(task.kind).strip().lower()
    if kind not in {"umap", "tsne"}:
        raise ValueError("Unsupported projection kind")

    inserted = False
    while not inserted:
        try:
            app_instance.state.projection_refresh_queue.put_nowait(task)
            inserted = True
            with app_instance.state.plot_lock:
                app_instance.state.manual_projections[kind]["pending_count"] += 1
                app_instance.state.manual_projections[kind]["error"] = None
                app_instance.state.manual_projections[kind]["status"] = f"Refreshing {_projection_label(kind)}..."
        except queue.Full:
            try:
                dropped_task = app_instance.state.projection_refresh_queue.get_nowait()
                dropped_kind = str(dropped_task.kind).strip().lower()
                with app_instance.state.plot_lock:
                    if dropped_kind in app_instance.state.manual_projections:
                        app_instance.state.manual_projections[dropped_kind]["pending_count"] = max(
                            0, int(app_instance.state.manual_projections[dropped_kind]["pending_count"]) - 1
                        )
            except queue.Empty:
                break

    _queue_broadcast_event(app_instance, "projection", projection_kind=kind)


def _schedule_startup_projection_refreshes(app_instance: FastAPI):
    point_count = _current_plot_count(app_instance)
    if point_count <= 0:
        return

    for kind in ("umap", "tsne"):
        try:
            _enqueue_projection_refresh(app_instance, ProjectionRefreshTask(kind=kind))
        except Exception as exc:
            with app_instance.state.plot_lock:
                projection = app_instance.state.manual_projections[kind]
                projection["error"] = str(exc)
                projection["status"] = f"{_projection_label(kind)} startup refresh failed."
                projection["version"] += 1
            _queue_broadcast_event(app_instance, "projection", projection_kind=kind)


def _set_roi_result(app_instance: FastAPI, roi_image: np.ndarray | None, source_frame_id: int, processing_ms: float):
    preprocessed_batch = None

    if roi_image is None or roi_image.size == 0:
        roi_display = _placeholder("No ROI detected", ROI_DISPLAY_SIZE, ROI_DISPLAY_SIZE)
        preprocessed_display = _placeholder("No Preprocessed ROI", ROI_DISPLAY_SIZE, ROI_DISPLAY_SIZE)
        has_roi = False
    else:
        roi_display = _resize_to_display(roi_image)
        preprocessed_batch, preprocessed_display = _preprocess_roi_for_registration(roi_image)
        if preprocessed_batch is None or preprocessed_display is None:
            preprocessed_display = _placeholder("Preprocess failed", ROI_DISPLAY_SIZE, ROI_DISPLAY_SIZE)
        has_roi = True

    payload = _encode_png(roi_display)
    preprocessed_payload = _encode_png(preprocessed_display)
    with app_instance.state.roi_lock:
        app_instance.state.roi_jpeg = payload
        app_instance.state.preprocessed_roi_jpeg = preprocessed_payload
        app_instance.state.last_roi_detected = has_roi
        app_instance.state.last_processed_frame_id = source_frame_id
        app_instance.state.last_processed_at = time.time()
        app_instance.state.last_processing_ms = processing_ms

    _queue_broadcast_event(app_instance, "result")
    return preprocessed_batch


def _reset_results(app_instance: FastAPI):
    with app_instance.state.roi_lock:
        app_instance.state.roi_jpeg = _encode_png(_placeholder("Waiting for ROI...", ROI_DISPLAY_SIZE, ROI_DISPLAY_SIZE))
        app_instance.state.preprocessed_roi_jpeg = _encode_png(
            _placeholder("Waiting for ROI...", ROI_DISPLAY_SIZE, ROI_DISPLAY_SIZE)
        )
        app_instance.state.last_roi_detected = False
        app_instance.state.last_processed_frame_id = -1
        app_instance.state.last_processed_at = 0.0
        app_instance.state.last_processing_ms = 0.0

    _queue_broadcast_event(app_instance, "snapshot")


def _input_worker(app_instance: FastAPI):
    last_version = 0

    while not app_instance.state.shutdown_event.is_set():
        item, last_version = app_instance.state.input_mailbox.get_after(last_version, app_instance.state.shutdown_event)
        if item is None:
            continue

        try:
            encoded = np.frombuffer(item.payload, dtype=np.uint8)
            if encoded.size == 0:
                raise ValueError("invalid frame payload")

            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                raise ValueError("failed to decode frame")

            height, width = frame.shape[:2]
            with app_instance.state.frame_lock:
                app_instance.state.capture_width = int(width)
                app_instance.state.capture_height = int(height)

            app_instance.state.roi_mailbox.put(
                DecodedFrame(frame_id=item.frame_id, frame=frame, timestamp=item.timestamp)
            )
        except Exception as exc:
            app_instance.state.warmup_error = str(exc)


def _roi_worker(app_instance: FastAPI):
    dummy = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8)
    try:
        extract_palm_roi(dummy)
        app_instance.state.warmup_ready = True
        app_instance.state.warmup_error = None
    except Exception as exc:
        app_instance.state.warmup_ready = False
        app_instance.state.warmup_error = str(exc)

    last_version = 0

    while not app_instance.state.shutdown_event.is_set():
        item, last_version = app_instance.state.roi_mailbox.get_after(last_version, app_instance.state.shutdown_event)
        if item is None:
            continue

        started_at = time.perf_counter()
        try:
            process_frame = _resize_for_processing(item.frame)
            roi = extract_palm_roi(process_frame)
            duration_ms = (time.perf_counter() - started_at) * 1000.0
            app_instance.state.preprocess_mailbox.put(
                RoiStageResult(frame_id=item.frame_id, roi_image=roi, processing_ms=duration_ms)
            )
        except Exception as exc:
            app_instance.state.warmup_error = str(exc)
            duration_ms = (time.perf_counter() - started_at) * 1000.0
            app_instance.state.preprocess_mailbox.put(
                RoiStageResult(frame_id=item.frame_id, roi_image=None, processing_ms=duration_ms)
            )


def _preprocess_worker(app_instance: FastAPI):
    last_version = 0

    while not app_instance.state.shutdown_event.is_set():
        item, last_version = app_instance.state.preprocess_mailbox.get_after(
            last_version, app_instance.state.shutdown_event
        )
        if item is None:
            continue

        try:
            preprocessed_batch = _set_roi_result(app_instance, item.roi_image, item.frame_id, item.processing_ms)
            target_id = _get_register_target(app_instance)
            if target_id is not None and preprocessed_batch is not None:
                _enqueue_register_task(
                    app_instance,
                    RegistrationTask(
                        frame_id=item.frame_id,
                        target_id=target_id,
                        batch=preprocessed_batch,
                    ),
                )
            verify_config = _get_verify_config(app_instance)
            if verify_config is not None:
                threshold, top_k = verify_config
                if preprocessed_batch is not None:
                    _enqueue_verify_task(
                        app_instance,
                        VerifyTask(
                            frame_id=item.frame_id,
                            threshold=threshold,
                            top_k=top_k,
                            batch=preprocessed_batch,
                        ),
                    )
                else:
                    _set_verify_error(app_instance, "Verify failed: No palm ROI detected.")
        except Exception as exc:
            app_instance.state.warmup_error = str(exc)
            _set_roi_result(app_instance, None, item.frame_id, item.processing_ms)


def _register_worker(app_instance: FastAPI):
    while not app_instance.state.shutdown_event.is_set():
        try:
            task = app_instance.state.register_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        with app_instance.state.register_lock:
            app_instance.state.register_pending_count = max(0, app_instance.state.register_pending_count - 1)
            enabled = bool(app_instance.state.register_enabled)
            target_id = app_instance.state.register_target_id

        if not enabled or target_id is None or int(target_id) != int(task.target_id):
            _queue_broadcast_event(app_instance, "register")
            continue

        try:
            embedding = _infer_embedding_from_batch(app_instance, task.batch)
            with app_instance.state.register_lock:
                still_enabled = bool(app_instance.state.register_enabled)
                current_target_id = app_instance.state.register_target_id
            if not still_enabled or current_target_id is None or int(current_target_id) != int(task.target_id):
                _queue_broadcast_event(app_instance, "register")
                continue
            _register_embedding(app_instance, task.target_id, embedding)
        except Exception as exc:
            _set_register_error(app_instance, str(exc))


def _query_topk_euclid(app_instance: FastAPI, query_vec: np.ndarray, top_k: int) -> list[dict]:
    helper = app_instance.state.qdrant_client
    if helper is None:
        return []

    query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    if query.size == 0:
        return []

    heap: list[tuple[float, dict]] = []
    offset = None
    batch_size = 256

    while True:
        points, offset = helper.client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        batch_vectors: list[np.ndarray] = []
        batch_meta: list[tuple[str, str, str]] = []
        for point in points:
            vec = _extract_qdrant_vector(point)
            if vec is None or vec.shape[0] != query.shape[0]:
                continue

            payload = point.payload or {}
            subject_id = payload.get("subject_id", payload.get("label", point.id))
            batch_vectors.append(vec)
            batch_meta.append((str(point.id), str(subject_id), str(payload.get("label", subject_id))))

        if batch_vectors:
            stacked = np.vstack(batch_vectors).astype(np.float32, copy=False)
            distances = np.linalg.norm(stacked - query[None, :], axis=1)
            for idx, distance in enumerate(distances):
                point_id, subject_id, label = batch_meta[idx]
                candidate = {
                    "rank": 0,
                    "point_id": point_id,
                    "subject_id": subject_id,
                    "label": label,
                    "distance": float(distance),
                }
                item = (-float(distance), candidate)
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif float(distance) < -heap[0][0]:
                    heapq.heapreplace(heap, item)

        if offset is None:
            break

    ordered = [entry[1] for entry in sorted(heap, key=lambda item: -item[0])]
    for index, match in enumerate(ordered, start=1):
        match["rank"] = index
    return ordered


def _verify_embedding(app_instance: FastAPI, embedding: np.ndarray, threshold: float, top_k: int):
    matches = _query_topk_euclid(app_instance, embedding, top_k)
    if not matches:
        _set_verify_error(app_instance, "Verify failed: No vectors in database.", [])
        return

    top1 = matches[0]
    distance = float(top1["distance"])
    subject_id = str(top1["subject_id"])
    if distance < threshold:
        message = (
            f"Verified: ID={subject_id} (distance={distance:.4f}, "
            f"threshold={threshold:.4f}, top-{top_k})"
        )
        _set_verify_status(app_instance, message, matches)
        return

    message = (
        f"No match: top1 ID={subject_id} (distance={distance:.4f}, "
        f"threshold={threshold:.4f}, top-{top_k})"
    )
    _set_verify_status(app_instance, message, matches)


def _verify_worker(app_instance: FastAPI):
    while not app_instance.state.shutdown_event.is_set():
        try:
            task = app_instance.state.verify_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        with app_instance.state.register_lock:
            app_instance.state.verify_pending_count = max(0, app_instance.state.verify_pending_count - 1)
            enabled = bool(app_instance.state.verify_enabled)
            threshold = float(app_instance.state.verify_threshold)
            top_k = int(app_instance.state.verify_top_k)

        if not enabled or abs(threshold - float(task.threshold)) > 1e-6 or int(top_k) != int(task.top_k):
            _queue_broadcast_event(app_instance, "verify")
            continue

        try:
            embedding = _infer_embedding_from_batch(app_instance, task.batch)
            _verify_embedding(app_instance, embedding, float(task.threshold), int(task.top_k))
        except Exception as exc:
            _set_verify_error(app_instance, f"Verify failed: {exc}")


def _plot_worker(app_instance: FastAPI):
    while not app_instance.state.shutdown_event.is_set():
        try:
            task = app_instance.state.plot_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        with app_instance.state.plot_lock:
            app_instance.state.plot_pending_count = max(0, app_instance.state.plot_pending_count - 1)
            app_instance.state.plot_point_ids.append(str(task.point_id))
            app_instance.state.plot_subject_ids.append(str(task.subject_id))
            app_instance.state.plot_vectors.append(np.asarray(task.embedding, dtype=np.float32).reshape(-1).copy())
            app_instance.state.plot_latest_point_id = str(task.point_id)

            if PLOT_MAX_POINTS > 0:
                overflow = len(app_instance.state.plot_point_ids) - PLOT_MAX_POINTS
                if overflow > 0:
                    app_instance.state.plot_point_ids = app_instance.state.plot_point_ids[overflow:]
                    app_instance.state.plot_subject_ids = app_instance.state.plot_subject_ids[overflow:]
                    app_instance.state.plot_vectors = app_instance.state.plot_vectors[overflow:]

        try:
            _recompute_plot_projection(app_instance)
            _queue_broadcast_event(app_instance, "plot")
        except Exception as exc:
            app_instance.state.warmup_error = str(exc)


def _projection_refresh_worker(app_instance: FastAPI):
    while not app_instance.state.shutdown_event.is_set():
        try:
            task = app_instance.state.projection_refresh_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        kind = str(task.kind).strip().lower()
        label = _projection_label(kind)

        try:
            point_ids, subject_ids, X, latest_point_id = _snapshot_plot_records(app_instance)
            if X.shape[0] == 0:
                points: list[dict] = []
                count = 0
                status = "No registered vectors available."
                error = None
                refreshed_at = 0.0
            else:
                if kind == "umap":
                    coords = _project_umap(X)
                elif kind == "tsne":
                    coords = _project_tsne(X)
                else:
                    raise ValueError("Unsupported projection kind")

                points = _build_projection_points(point_ids, subject_ids, coords, latest_point_id)
                count = len(points)
                status = f"{label} refreshed from {count} registered vectors."
                error = None
                refreshed_at = time.time()
        except Exception as exc:
            with app_instance.state.plot_lock:
                projection = app_instance.state.manual_projections[kind]
                projection["pending_count"] = max(0, int(projection["pending_count"]) - 1)
                projection["error"] = str(exc)
                projection["status"] = f"{label} refresh failed."
                projection["version"] += 1
            _queue_broadcast_event(app_instance, "projection", projection_kind=kind)
            continue

        with app_instance.state.plot_lock:
            projection = app_instance.state.manual_projections[kind]
            projection["points"] = points
            projection["count"] = count
            projection["pending_count"] = max(0, int(projection["pending_count"]) - 1)
            projection["error"] = error
            projection["status"] = status
            projection["refreshed_at"] = refreshed_at
            projection["version"] += 1

        _queue_broadcast_event(app_instance, "projection", projection_kind=kind)


def _services_status(app_instance: FastAPI):
    with app_instance.state.register_lock:
        register_state = {
            "enabled": app_instance.state.register_enabled,
            "target_id": app_instance.state.register_target_id,
            "insert_count": app_instance.state.register_insert_count,
            "pending_count": app_instance.state.register_pending_count,
            "last_error": app_instance.state.register_last_error,
            "keep_all_samples": REGISTER_KEEP_ALL_SAMPLES,
            "queue_maxsize": REGISTER_QUEUE_MAXSIZE,
        }
        verify_state = {
            "enabled": app_instance.state.verify_enabled,
            "threshold": app_instance.state.verify_threshold,
            "top_k": app_instance.state.verify_top_k,
            "pending_count": app_instance.state.verify_pending_count,
            "queue_maxsize": VERIFY_QUEUE_MAXSIZE,
            "last_result": app_instance.state.verify_last_result,
            "last_error": app_instance.state.verify_last_error,
            "top_matches": [dict(match) for match in app_instance.state.verify_top_matches],
        }
    with app_instance.state.plot_lock:
        plot_state = {
            "count": len(app_instance.state.plot_points),
            "version": app_instance.state.plot_version,
            "pending_count": app_instance.state.plot_pending_count,
            "max_points": PLOT_MAX_POINTS,
        }

    return {
        "triton": {
            "grpc_url": TRITON_GRPC_URL,
            "model_name": TRITON_MODEL_NAME,
            "input_name": TRITON_INPUT_NAME,
            "output_name": TRITON_OUTPUT_NAME,
            "connected": app_instance.state.triton_client is not None,
            "error": app_instance.state.triton_error,
        },
        "qdrant": {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "grpc_port": QDRANT_GRPC_PORT,
            "collection": QDRANT_COLLECTION,
            "vector_size": FEATURE_VECTOR_SIZE,
            "distance": QDRANT_DISTANCE,
            "connected": app_instance.state.qdrant_client is not None,
            "error": app_instance.state.qdrant_error,
        },
        "register": register_state,
        "verify": verify_state,
        "plot": plot_state,
    }


def _start_register_mode(app_instance: FastAPI, target_id: int):
    if app_instance.state.triton_client is None or app_instance.state.qdrant_client is None:
        _set_register_error(app_instance, "Register unavailable: Triton or Qdrant not connected.")
        return

    _clear_registration_queue(app_instance)
    _clear_verification_queue(app_instance)
    with app_instance.state.register_lock:
        app_instance.state.register_enabled = True
        app_instance.state.register_target_id = int(target_id)
        app_instance.state.register_insert_count = 0
        app_instance.state.register_pending_count = 0
        app_instance.state.register_last_error = None
        app_instance.state.verify_enabled = False
        app_instance.state.verify_pending_count = 0
        app_instance.state.verify_last_error = None
        app_instance.state.verify_last_result = "Verify OFF (Register mode active)."
        app_instance.state.verify_top_matches = []

    _queue_broadcast_event(app_instance, "register")


def _stop_register_mode(app_instance: FastAPI):
    _clear_registration_queue(app_instance)
    with app_instance.state.register_lock:
        app_instance.state.register_enabled = False
        app_instance.state.register_pending_count = 0
        app_instance.state.register_last_error = None

    _queue_broadcast_event(app_instance, "register")


def _start_verify_mode(app_instance: FastAPI, threshold: float, top_k: int):
    if app_instance.state.triton_client is None or app_instance.state.qdrant_client is None:
        _set_verify_error(app_instance, "Verify unavailable: Triton or Qdrant not connected.")
        return

    _clear_registration_queue(app_instance)
    _clear_verification_queue(app_instance)
    with app_instance.state.register_lock:
        app_instance.state.register_enabled = False
        app_instance.state.register_pending_count = 0
        app_instance.state.register_last_error = None
        app_instance.state.verify_enabled = True
        app_instance.state.verify_threshold = float(threshold)
        app_instance.state.verify_top_k = max(1, int(top_k))
        app_instance.state.verify_pending_count = 0
        app_instance.state.verify_last_error = None
        app_instance.state.verify_last_result = (
            f"Verify ON (top1 Euclid, threshold={float(threshold):.4f}, top-{max(1, int(top_k))})"
        )
        app_instance.state.verify_top_matches = []

    _queue_broadcast_event(app_instance, "verify")


def _stop_verify_mode(app_instance: FastAPI):
    _clear_verification_queue(app_instance)
    with app_instance.state.register_lock:
        app_instance.state.verify_enabled = False
        app_instance.state.verify_pending_count = 0
        app_instance.state.verify_last_error = None
        app_instance.state.verify_last_result = "Verify OFF."
        app_instance.state.verify_top_matches = []

    _queue_broadcast_event(app_instance, "verify")


def _build_public_state(app_instance: FastAPI):
    with app_instance.state.frame_lock:
        latest_frame_id = app_instance.state.latest_frame_id
        latest_frame_ts = app_instance.state.latest_frame_ts
        capture_width = int(getattr(app_instance.state, "capture_width", CAPTURE_WIDTH))
        capture_height = int(getattr(app_instance.state, "capture_height", CAPTURE_HEIGHT))
        has_frame = latest_frame_ts > 0.0 and (
            CAMERA_STALE_SECONDS <= 0 or (time.time() - latest_frame_ts) <= CAMERA_STALE_SECONDS
        )

    with app_instance.state.roi_lock:
        has_roi = app_instance.state.last_roi_detected
        last_processed_frame_id = app_instance.state.last_processed_frame_id
        last_processed_at = app_instance.state.last_processed_at
        last_processing_ms = app_instance.state.last_processing_ms

    return {
        "camera": {
            "width": capture_width,
            "height": capture_height,
            "has_frame": has_frame,
            "latest_frame_id": latest_frame_id,
            "latest_frame_ts": latest_frame_ts,
            "target_upload_fps": UPLOAD_TARGET_FPS,
        },
        "processed": {
            "has_roi": has_roi,
            "last_processed_frame_id": last_processed_frame_id,
            "last_processed_at": last_processed_at,
            "last_processing_ms": last_processing_ms,
        },
        "backend": {
            "mode": "roi_register_verify",
            "warmup_ready": bool(app_instance.state.warmup_ready),
            "warmup_error": app_instance.state.warmup_error,
            "process_max_width": ROI_PROCESS_MAX_WIDTH,
            "roi_display_size": ROI_DISPLAY_SIZE,
        },
        "streams": {
            "roi": "/api/frame/latest/roi",
            "preprocessed": "/api/frame/latest/preprocessed",
        },
        "services": _services_status(app_instance),
    }


def _build_stream_event(app_instance: FastAPI, event_type: str, projection_kind: str | None = None):
    state = _build_public_state(app_instance)
    with app_instance.state.roi_lock:
        roi_payload = bytes(app_instance.state.roi_jpeg)
        preprocessed_payload = bytes(app_instance.state.preprocessed_roi_jpeg)

    event = {
        "type": event_type,
        "state": state,
        "images": {
            "roi": _payload_to_data_url(roi_payload),
            "preprocessed": _payload_to_data_url(preprocessed_payload),
        },
    }

    if event_type in {"snapshot", "plot"}:
        event["plot"] = _build_plot_payload(app_instance)
    if event_type == "snapshot":
        event["umap"] = _build_manual_projection_payload(app_instance, "umap")
        event["tsne"] = _build_manual_projection_payload(app_instance, "tsne")
    elif event_type == "projection" and projection_kind in {"umap", "tsne"}:
        event[projection_kind] = _build_manual_projection_payload(app_instance, projection_kind)

    return event


def _push_broadcast_event(app_instance: FastAPI, event: dict):
    queue = app_instance.state.broadcast_queue
    if queue.full():
        with suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    with suppress(asyncio.QueueFull):
        queue.put_nowait(event)


def _queue_broadcast_event(app_instance: FastAPI, event_type: str, projection_kind: str | None = None):
    loop = getattr(app_instance.state, "async_loop", None)
    if loop is None or loop.is_closed():
        return

    event = _build_stream_event(app_instance, event_type, projection_kind=projection_kind)
    loop.call_soon_threadsafe(_push_broadcast_event, app_instance, event)


def _clear_registered_vectors(app_instance: FastAPI):
    helper = app_instance.state.qdrant_client
    if helper is None:
        raise RuntimeError(app_instance.state.qdrant_error or "Qdrant is not connected")

    helper.delete_all_points(QDRANT_COLLECTION)
    _clear_registration_queue(app_instance)
    _clear_verification_queue(app_instance)
    _clear_plot_queue(app_instance)
    _clear_projection_refresh_queue(app_instance)
    _reset_plot_state(app_instance)

    with app_instance.state.register_lock:
        app_instance.state.register_insert_count = 0
        app_instance.state.register_pending_count = 0
        app_instance.state.register_last_error = None
        app_instance.state.verify_pending_count = 0
        app_instance.state.verify_last_error = None
        app_instance.state.verify_top_matches = []
        if app_instance.state.verify_enabled:
            app_instance.state.verify_last_result = "Database cleared. No vectors registered."
        else:
            app_instance.state.verify_last_result = "Verify OFF."

    _queue_broadcast_event(app_instance, "snapshot")


async def _broadcast_loop(app_instance: FastAPI):
    while True:
        event = await app_instance.state.broadcast_queue.get()
        if not app_instance.state.ws_clients:
            continue

        stale_clients: list[WebSocket] = []
        for websocket in tuple(app_instance.state.ws_clients):
            try:
                await websocket.send_json(event)
            except Exception:
                stale_clients.append(websocket)

        for websocket in stale_clients:
            app_instance.state.ws_clients.discard(websocket)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    app_instance.state.async_loop = asyncio.get_running_loop()
    app_instance.state.broadcast_queue = asyncio.Queue(maxsize=16)
    app_instance.state.ws_clients = set()
    app_instance.state.frame_lock = threading.Lock()
    app_instance.state.roi_lock = threading.Lock()
    app_instance.state.register_lock = threading.Lock()
    app_instance.state.plot_lock = threading.Lock()
    app_instance.state.input_mailbox = LatestItemMailbox()
    app_instance.state.roi_mailbox = LatestItemMailbox()
    app_instance.state.preprocess_mailbox = LatestItemMailbox()
    app_instance.state.register_queue = queue.Queue(maxsize=REGISTER_QUEUE_MAXSIZE)
    app_instance.state.verify_queue = queue.Queue(maxsize=VERIFY_QUEUE_MAXSIZE)
    app_instance.state.plot_queue = queue.Queue(maxsize=PLOT_QUEUE_MAXSIZE)
    app_instance.state.projection_refresh_queue = queue.Queue(maxsize=PROJECTION_REFRESH_QUEUE_MAXSIZE)

    app_instance.state.latest_frame_id = 0
    app_instance.state.latest_frame_ts = 0.0
    app_instance.state.capture_width = CAPTURE_WIDTH
    app_instance.state.capture_height = CAPTURE_HEIGHT

    app_instance.state.roi_jpeg = b""
    app_instance.state.preprocessed_roi_jpeg = b""
    app_instance.state.last_roi_detected = False
    app_instance.state.last_processed_frame_id = -1
    app_instance.state.last_processed_at = 0.0
    app_instance.state.last_processing_ms = 0.0

    app_instance.state.warmup_ready = False
    app_instance.state.warmup_error = None
    app_instance.state.register_enabled = False
    app_instance.state.register_target_id = None
    app_instance.state.register_insert_count = 0
    app_instance.state.register_pending_count = 0
    app_instance.state.register_last_error = None
    app_instance.state.verify_enabled = False
    app_instance.state.verify_threshold = VERIFY_THRESHOLD_DEFAULT
    app_instance.state.verify_top_k = VERIFY_TOP_K_DEFAULT
    app_instance.state.verify_pending_count = 0
    app_instance.state.verify_last_error = None
    app_instance.state.verify_last_result = "Verify OFF."
    app_instance.state.verify_top_matches = []
    app_instance.state.plot_point_ids = []
    app_instance.state.plot_subject_ids = []
    app_instance.state.plot_vectors = []
    app_instance.state.plot_points = []
    app_instance.state.plot_latest_point_id = None
    app_instance.state.plot_version = 0
    app_instance.state.plot_pending_count = 0
    app_instance.state.plot_updated_at = 0.0
    app_instance.state.manual_projections = {
        "umap": {
            "points": [],
            "count": 0,
            "version": 0,
            "pending_count": 0,
            "status": "No registered vectors available.",
            "error": None,
            "refreshed_at": 0.0,
        },
        "tsne": {
            "points": [],
            "count": 0,
            "version": 0,
            "pending_count": 0,
            "status": "No registered vectors available.",
            "error": None,
            "refreshed_at": 0.0,
        },
    }

    _init_backends(app_instance)
    _reset_manual_projection_cache(app_instance, bump_version=False)

    app_instance.state.shutdown_event = threading.Event()
    app_instance.state.worker_threads = [
        threading.Thread(target=_input_worker, args=(app_instance,), name="input-worker", daemon=True),
        threading.Thread(target=_roi_worker, args=(app_instance,), name="roi-worker", daemon=True),
        threading.Thread(target=_preprocess_worker, args=(app_instance,), name="preprocess-worker", daemon=True),
        threading.Thread(target=_register_worker, args=(app_instance,), name="register-worker", daemon=True),
        threading.Thread(target=_verify_worker, args=(app_instance,), name="verify-worker", daemon=True),
        threading.Thread(target=_plot_worker, args=(app_instance,), name="plot-worker", daemon=True),
        threading.Thread(
            target=_projection_refresh_worker,
            args=(app_instance,),
            name="projection-refresh-worker",
            daemon=True,
        ),
    ]
    app_instance.state.broadcast_task = asyncio.create_task(_broadcast_loop(app_instance))
    _reset_results(app_instance)

    for worker in app_instance.state.worker_threads:
        worker.start()

    _schedule_startup_projection_refreshes(app_instance)

    try:
        yield
    finally:
        app_instance.state.shutdown_event.set()
        app_instance.state.broadcast_task.cancel()
        for mailbox in (
            app_instance.state.input_mailbox,
            app_instance.state.roi_mailbox,
            app_instance.state.preprocess_mailbox,
        ):
            mailbox.close()

        for worker in app_instance.state.worker_threads:
            worker.join(timeout=1.0)
        with suppress(asyncio.CancelledError):
            await app_instance.state.broadcast_task


app = FastAPI(title="Palm ROI Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")


@app.get("/api/state")
async def api_state():
    return JSONResponse(_build_public_state(app))


@app.get("/api/config")
async def api_config():
    return {
        "capture_width": CAPTURE_WIDTH,
        "capture_height": CAPTURE_HEIGHT,
        "roi_size": ROI_DISPLAY_SIZE,
        "target_upload_fps": UPLOAD_TARGET_FPS,
    }


@app.post("/api/camera/frame")
async def api_camera_frame(request: Request):
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="empty frame payload")

    _mark_latest_frame_received(app, payload)
    return Response(status_code=204)


@app.post("/api/camera/stop")
async def api_camera_stop():
    _clear_pipeline(app)
    _reset_results(app)
    return Response(status_code=204)


@app.post("/api/database/clear")
async def api_database_clear():
    try:
        await asyncio.to_thread(_clear_registered_vectors, app)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return JSONResponse(
        {
            "ok": True,
            "message": "Cleared all registered vectors from Qdrant.",
            "plot": _build_plot_payload(app),
            "umap": _build_manual_projection_payload(app, "umap"),
            "tsne": _build_manual_projection_payload(app, "tsne"),
        }
    )


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    await websocket.accept()
    app.state.ws_clients.add(websocket)
    await websocket.send_json(_build_stream_event(app, "snapshot"))

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            payload = message.get("bytes")
            if payload:
                _mark_latest_frame_received(app, payload)
                continue

            text_payload = message.get("text")
            if not text_payload:
                continue

            try:
                command = json.loads(text_payload)
            except json.JSONDecodeError:
                continue

            if command.get("type") == "stop":
                _clear_pipeline(app)
                _clear_registration_queue(app)
                _clear_verification_queue(app)
                _reset_results(app)
            elif command.get("type") == "register_start":
                try:
                    target_id = int(command.get("targetId"))
                except Exception:
                    _set_register_error(app, "Register ID must be an integer.")
                    continue
                _start_register_mode(app, target_id)
            elif command.get("type") == "register_update":
                try:
                    target_id = int(command.get("targetId"))
                except Exception:
                    _set_register_error(app, "Register ID must be an integer.")
                    continue
                _update_register_target(app, target_id)
            elif command.get("type") == "register_stop":
                _stop_register_mode(app)
            elif command.get("type") == "verify_start":
                try:
                    threshold = float(command.get("threshold"))
                except Exception:
                    threshold = VERIFY_THRESHOLD_DEFAULT

                try:
                    top_k = int(command.get("topK"))
                except Exception:
                    top_k = VERIFY_TOP_K_DEFAULT

                _start_verify_mode(app, float(threshold), max(1, int(top_k)))
            elif command.get("type") == "verify_stop":
                _stop_verify_mode(app)
            elif command.get("type") == "projection_refresh":
                kind = str(command.get("kind", "")).strip().lower()
                if kind not in {"umap", "tsne"}:
                    continue
                try:
                    _enqueue_projection_refresh(app, ProjectionRefreshTask(kind=kind))
                except Exception:
                    continue
            elif command.get("type") == "snapshot":
                await websocket.send_json(_build_stream_event(app, "snapshot"))
    except WebSocketDisconnect:
        pass
    finally:
        app.state.ws_clients.discard(websocket)


@app.get("/api/frame/latest/{kind}")
async def api_frame_latest(kind: str):
    normalized = str(kind).strip().lower()
    if normalized == "roi":
        with app.state.roi_lock:
            payload = app.state.roi_jpeg
    elif normalized in {"preprocessed", "roi-preprocessed", "processed"}:
        with app.state.roi_lock:
            payload = app.state.preprocessed_roi_jpeg
    else:
        raise HTTPException(status_code=404, detail="Unknown frame kind")

    return Response(
        content=payload,
        media_type="image/png",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/", include_in_schema=False)
async def root():
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(
        {
            "ok": True,
            "message": "Frontend build not found. Run `npm install && npm run build` inside `frontend/`, then restart `backend.py`.",
            "api": "/api/state",
        }
    )


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    if full_path.startswith("api/") or full_path.startswith("healthz"):
        raise HTTPException(status_code=404, detail="Not found")

    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    raise HTTPException(status_code=404, detail="Frontend build not found")


if __name__ == "__main__":
    if importlib.util.find_spec("websockets") is None and importlib.util.find_spec("wsproto") is None:
        raise RuntimeError(
            "Realtime websocket backend requires websocket support. "
            "Install dependencies with `pip install -r requirements.txt` "
            "or `pip install 'uvicorn[standard]'`."
        )

    uvicorn.run(
        "backend:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "7001")),
        reload=False,
    )
