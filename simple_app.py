import asyncio
import colorsys
import io
import os
import threading
import time
import textwrap
import zlib
from contextlib import asynccontextmanager

import cv2
import gradio as gr
import matplotlib
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, RedirectResponse, Response, StreamingResponse
from qdrant_client.models import Distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from roi_extraction import extract_palm_roi
from utils.qdrant import QdrantHelper
from utils.preprocess import preprocess
from utils.triton import TritonClient

matplotlib.use("Agg")
from matplotlib import pyplot as plt

JPEG_QUALITY = 100
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
ROI_EVERY_N_FRAMES = 1
ROI_DISPLAY_SIZE = 224
CAMERA_DISPLAY_WIDTH = 864
CAMERA_DISPLAY_HEIGHT = 486
PLOT_MAX_POINTS = int(os.getenv("PLOT_MAX_POINTS", "0"))
PLOT_REFRESH_MS = int(os.getenv("PLOT_REFRESH_MS", "250"))
PLOT_PADDING_RATIO = float(os.getenv("PLOT_PADDING_RATIO", "0.2"))
PLOT_MIN_SPAN = float(os.getenv("PLOT_MIN_SPAN", "0.0"))
PLOT_JITTER_RATIO = float(os.getenv("PLOT_JITTER_RATIO", "0.0"))
PLOT_SYNC_SECONDS = float(os.getenv("PLOT_SYNC_SECONDS", "1.5"))
TSNE_MAX_POINTS = int(os.getenv("TSNE_MAX_POINTS", "1200"))
EMBED_VIZ_METHOD = os.getenv("EMBED_VIZ_METHOD", "umap").lower()
EMBED_VIZ_METRIC = os.getenv("EMBED_VIZ_METRIC", "euclidean")
EMBED_VIZ_N_NEIGHBORS = int(os.getenv("EMBED_VIZ_N_NEIGHBORS", "15"))
EMBED_VIZ_MIN_DIST = float(os.getenv("EMBED_VIZ_MIN_DIST", "0.15"))
PLOT_METHODS = ("pca", "umap", "tsne")
_MATPLOTLIB_RENDER_LOCK = threading.Lock()

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

try:
    import umap.umap_ as umap
except Exception:
    umap = None


def _configure_capture(cap: cv2.VideoCapture):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)


def _open_camera(idx: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    _configure_capture(cap)
    for _ in range(3):
        cap.read()
    return cap


def _capture_loop(app: FastAPI):
    while not app.state.capture_stop.is_set():
        cap = app.state.cap_map.get(app.state.cam_idx)
        if cap is None or not cap.isOpened():
            time.sleep(0.01)
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.005)
            continue

        with app.state.frame_lock:
            app.state.latest_frame = frame
            app.state.latest_frame_id += 1


def _get_latest_frame(app: FastAPI):
    with app.state.frame_lock:
        if app.state.latest_frame is None:
            return None, app.state.latest_frame_id
        return app.state.latest_frame.copy(), app.state.latest_frame_id


def _placeholder(text: str, width: int = 640, height: int = 480):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def switch_camera_sync(idx_str: str) -> str:
    idx = int(idx_str)
    prev_idx = app.state.cam_idx

    if idx == prev_idx and idx in app.state.cap_map and app.state.cap_map[idx].isOpened():
        return f"Already using camera {idx}"

    cap_new = _open_camera(idx)
    if cap_new is None:
        return f"Camera {idx} could not be opened."

    old_cap = app.state.cap_map.pop(prev_idx, None)
    app.state.cap_map[idx] = cap_new
    app.state.cam_idx = idx

    with app.state.frame_lock:
        app.state.latest_frame = None
        app.state.latest_frame_id = 0
    with app.state.register_lock:
        app.state.register_last_frame_id = -1

    if old_cap is not None and old_cap.isOpened():
        old_cap.release()

    return f"Switched to camera {idx}"


def _preprocess_roi(roi: np.ndarray):
    try:
        batch, resized = preprocess(roi)
    except Exception:
        return None, None

    if (
        batch is None
        or resized is None
        or batch.size == 0
        or resized.size == 0
    ):
        return None, None

    display = resized
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    return batch.astype(np.float32, copy=False), display


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


def _distance_to_name(distance_value) -> str:
    raw = str(distance_value).strip().upper()
    if "COS" in raw:
        return "COSINE"
    if "EUC" in raw or "L2" in raw:
        return "EUCLID"
    if "DOT" in raw:
        return "DOT"
    if "MANH" in raw:
        return "MANHATTAN"
    return raw


def _get_collection_distance_name(qdrant_helper: QdrantHelper, collection_name: str) -> str | None:
    try:
        info = qdrant_helper.client.get_collection(collection_name)
        vectors_cfg = info.config.params.vectors
        if isinstance(vectors_cfg, dict):
            if not vectors_cfg:
                return None
            vectors_cfg = next(iter(vectors_cfg.values()))
        distance_value = getattr(vectors_cfg, "distance", None)
        if distance_value is None:
            return None
        return _distance_to_name(distance_value)
    except Exception:
        return None


def _init_backends(app_instance: FastAPI):
    app_instance.state.triton_client = None
    app_instance.state.triton_error = None
    app_instance.state.qdrant_client = None
    app_instance.state.qdrant_error = None

    app_instance.state.triton_grpc_url = TRITON_GRPC_URL
    app_instance.state.triton_model_name = TRITON_MODEL_NAME
    app_instance.state.triton_input_name = TRITON_INPUT_NAME
    app_instance.state.triton_output_name = TRITON_OUTPUT_NAME

    app_instance.state.qdrant_host = QDRANT_HOST
    app_instance.state.qdrant_port = QDRANT_PORT
    app_instance.state.qdrant_grpc_port = QDRANT_GRPC_PORT
    app_instance.state.qdrant_collection = QDRANT_COLLECTION
    app_instance.state.qdrant_distance = QDRANT_DISTANCE

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
        print(
            f"[INFO] Triton connected at {TRITON_GRPC_URL}, "
            f"model='{TRITON_MODEL_NAME}'"
        )
    except Exception as exc:
        app_instance.state.triton_error = str(exc)
        print(f"[WARN] Triton initialization failed: {exc}")

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
        actual_distance = _get_collection_distance_name(qdrant_client, QDRANT_COLLECTION)
        requested_distance = _distance_to_name(QDRANT_DISTANCE)
        if (
            actual_distance is not None
            and requested_distance is not None
            and actual_distance != requested_distance
        ):
            print(
                "[WARN] Qdrant collection distance mismatch: "
                f"requested={requested_distance}, actual={actual_distance}. "
                "Delete/recreate collection to apply the requested distance."
            )
        app_instance.state.qdrant_client = qdrant_client
        _load_plot_points_from_qdrant(app_instance)
        print(
            f"[INFO] Qdrant connected at {QDRANT_HOST}:{QDRANT_PORT}, "
            f"collection='{QDRANT_COLLECTION}'"
        )
    except Exception as exc:
        app_instance.state.qdrant_error = str(exc)
        print(f"[WARN] Qdrant initialization failed: {exc}")


def _infer_embedding(app_instance: FastAPI, roi: np.ndarray):
    if app_instance.state.triton_client is None:
        return None
    batch, _ = preprocess(roi)
    return _infer_embedding_from_batch(app_instance, batch)


def _infer_embedding_from_batch(app_instance: FastAPI, batch: np.ndarray):
    if app_instance.state.triton_client is None:
        return None
    batch = batch.astype(np.float32, copy=False)
    outputs = app_instance.state.triton_client.infer(
        model_name=app_instance.state.triton_model_name,
        inputs={app_instance.state.triton_input_name: batch},
        outputs=[app_instance.state.triton_output_name],
    )
    embedding = outputs.get(app_instance.state.triton_output_name)
    if embedding is None or embedding.size == 0:
        return None
    return np.asarray(embedding).reshape(-1).astype(np.float32)


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


def _append_plot_point(app_instance: FastAPI, subject_id: str, embedding: np.ndarray):
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vec.size == 0:
        return

    need_init_pca = False
    with app_instance.state.plot_lock:
        app_instance.state.plot_subject_ids.append(str(subject_id))
        app_instance.state.plot_vectors.append(vec)
        if PLOT_MAX_POINTS > 0:
            overflow = len(app_instance.state.plot_subject_ids) - PLOT_MAX_POINTS
            if overflow > 0:
                app_instance.state.plot_subject_ids = app_instance.state.plot_subject_ids[overflow:]
                app_instance.state.plot_vectors = app_instance.state.plot_vectors[overflow:]
        app_instance.state.plot_version += 1
        need_init_pca = not app_instance.state.plot_pca_basis_ready

    if need_init_pca:
        _ensure_stable_pca_basis(app_instance)


def _load_plot_points_from_qdrant(app_instance: FastAPI):
    helper = app_instance.state.qdrant_client
    if helper is None:
        return

    loaded_records = []
    offset = None
    batch_size = 256

    while True:
        points, offset = helper.client.scroll(
            collection_name=app_instance.state.qdrant_collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        for point in points:
            vec = _extract_qdrant_vector(point)
            if vec is None:
                continue
            if vec.shape[0] != FEATURE_VECTOR_SIZE:
                continue
            payload = point.payload or {}
            sid = payload.get("subject_id", payload.get("label", "unknown"))
            updated_at = payload.get("updated_at", 0.0)
            try:
                updated_at = float(updated_at)
            except Exception:
                updated_at = 0.0

            pid = getattr(point, "id", 0)
            try:
                pid_sort = int(pid)
            except Exception:
                pid_sort = zlib.crc32(str(pid).encode("utf-8")) & 0xFFFFFFFF

            loaded_records.append((updated_at, pid_sort, str(sid), vec))

        if offset is None:
            break

    loaded_records.sort(key=lambda x: (x[0], x[1]))
    if PLOT_MAX_POINTS > 0 and len(loaded_records) > PLOT_MAX_POINTS:
        loaded_records = loaded_records[-PLOT_MAX_POINTS:]

    loaded_subject_ids = [rec[2] for rec in loaded_records]
    loaded_vectors = [rec[3] for rec in loaded_records]

    with app_instance.state.plot_lock:
        app_instance.state.plot_subject_ids = loaded_subject_ids
        app_instance.state.plot_vectors = loaded_vectors
        app_instance.state.plot_pca_mean = None
        app_instance.state.plot_pca_components = None
        app_instance.state.plot_pca_basis_ready = False
        app_instance.state.plot_version += 1

    _ensure_stable_pca_basis(app_instance)


def _color_for_subject_id(subject_id: str) -> str:
    hashed = zlib.crc32(subject_id.encode("utf-8")) & 0xFFFFFFFF
    hue = (hashed % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
    return "#{:02x}{:02x}{:02x}".format(
        int(r * 255),
        int(g * 255),
        int(b * 255),
    )


def _normalize_plot_method(method: str | None) -> str:
    if not method:
        return EMBED_VIZ_METHOD if EMBED_VIZ_METHOD in PLOT_METHODS else "pca"
    m = str(method).strip().lower()
    if m in ("t-sne", "t_sne"):
        m = "tsne"
    if m not in PLOT_METHODS:
        return EMBED_VIZ_METHOD if EMBED_VIZ_METHOD in PLOT_METHODS else "pca"
    return m


def _sanitize_plot_vectors(subject_ids: list[str], vectors: list[np.ndarray]):
    if not subject_ids or not vectors:
        return [], np.empty((0, 2), dtype=np.float32)

    valid_ids = []
    valid_vectors = []
    dim = None
    for sid, vec in zip(subject_ids, vectors):
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            continue
        if dim is None:
            dim = arr.size
        if arr.size != dim:
            continue
        valid_ids.append(str(sid))
        valid_vectors.append(arr)

    if not valid_vectors:
        return [], np.empty((0, 2), dtype=np.float32)
    return valid_ids, np.vstack(valid_vectors).astype(np.float32, copy=False)


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


def _fit_stable_pca_basis(X: np.ndarray):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] <= 0:
        return None, None
    if not np.all(np.isfinite(X)):
        return None, None

    n_components = min(2, X.shape[0], X.shape[1])
    if n_components < 1:
        return None, None

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)

    mean = np.asarray(pca.mean_, dtype=np.float32).reshape(-1)
    comps = _canonicalize_pca_components(pca.components_)
    if comps.size == 0:
        return None, None

    if comps.shape[0] == 1:
        comps = np.vstack([comps, np.zeros_like(comps[0], dtype=np.float32)])
    else:
        comps = comps[:2, :]

    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(comps)):
        return None, None
    return mean, comps


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
        # Keep PCA direction deterministic across restarts by fixing component sign.
        if float(np.dot(raw_components[i], canonical_components[i])) < 0.0:
            points[:, i] *= -1.0

    if points.shape[1] == 1:
        points = np.hstack([points, np.zeros((points.shape[0], 1), dtype=np.float32)])

    if not np.all(np.isfinite(points)):
        raise ValueError("PCA produced non-finite coordinates")
    return points[:, :2].astype(np.float32, copy=False)


def _ensure_stable_pca_basis(app_instance: FastAPI):
    with app_instance.state.plot_lock:
        if app_instance.state.plot_pca_basis_ready:
            return
        subject_ids = list(app_instance.state.plot_subject_ids)
        vectors = [v.copy() for v in app_instance.state.plot_vectors]

    _, X = _sanitize_plot_vectors(subject_ids, vectors)
    if X.shape[0] < 2:
        return

    mean, comps = _fit_stable_pca_basis(X)
    if mean is None or comps is None:
        return

    with app_instance.state.plot_lock:
        if app_instance.state.plot_pca_basis_ready:
            return
        app_instance.state.plot_pca_mean = mean
        app_instance.state.plot_pca_components = comps
        app_instance.state.plot_pca_basis_ready = True


def _project_vectors_pca_stable(app_instance: FastAPI, X: np.ndarray):
    _ = app_instance
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32), "pca"
    if X.shape[0] == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32), "pca"

    points = _project_pca_canonical(X)
    return points, "pca"


def _project_vectors_2d(X: np.ndarray, method: str | None = None):
    method = _normalize_plot_method(method)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32), method

    finite_mask = np.isfinite(X).all(axis=1)
    if not np.all(finite_mask):
        X = X[finite_mask]
    if X.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32), method

    if X.shape[0] <= 1:
        return np.array([[0.0, 0.0]], dtype=np.float32), method

    if method == "umap" and umap is not None and X.shape[0] >= 3:
        n_neighbors = min(max(2, EMBED_VIZ_N_NEIGHBORS), X.shape[0] - 1)
        try:
            reducer = umap.UMAP(
                n_components=2,
                metric=EMBED_VIZ_METRIC,
                n_neighbors=n_neighbors,
                min_dist=EMBED_VIZ_MIN_DIST,
                random_state=42,
            )
            points = reducer.fit_transform(X)
            if not np.all(np.isfinite(points)):
                raise ValueError("UMAP produced non-finite coordinates")
            return np.asarray(points, dtype=np.float32), "umap"
        except Exception:
            method = "pca"
    elif method == "umap":
        method = "pca"

    if method == "tsne" and X.shape[0] >= 3:
        try:
            if TSNE_MAX_POINTS > 0 and X.shape[0] > TSNE_MAX_POINTS:
                X = X[-TSNE_MAX_POINTS:, :]

            # Stable choice: valid for all n>=3 and avoids perplexity edge failures.
            max_perplexity = max(1.0, float(X.shape[0] - 1) / 3.0)
            perplexity = min(30.0, max_perplexity)
            perplexity = min(perplexity, float(X.shape[0]) - 1e-3)
            perplexity = max(1.0, perplexity)

            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                init="pca",
                learning_rate="auto",
            )
            points = tsne.fit_transform(X)
            if not np.all(np.isfinite(points)):
                raise ValueError("t-SNE produced non-finite coordinates")
            return np.asarray(points, dtype=np.float32), "tsne"
        except Exception:
            method = "pca"
    elif method == "tsne":
        method = "pca"

    try:
        points = _project_pca_canonical(X)
        return points, "pca"
    except Exception:
        n = int(X.shape[0])
        fallback = np.zeros((n, 2), dtype=np.float32)
        if n > 1:
            fallback[:, 0] = np.linspace(-1.0, 1.0, num=n, dtype=np.float32)
        return fallback, "pca"


def _stable_jitter(subject_ids: list[str], points_2d: np.ndarray, base_span: float) -> np.ndarray:
    if not subject_ids:
        return np.zeros((0, 2), dtype=np.float32)
    scale = max(0.0, float(base_span)) * max(0.0, PLOT_JITTER_RATIO)
    if scale <= 0:
        return np.zeros((len(subject_ids), 2), dtype=np.float32)

    jitter = np.zeros((len(subject_ids), 2), dtype=np.float32)
    for i, sid in enumerate(subject_ids):
        px = 0
        py = 0
        if i < points_2d.shape[0]:
            px = int(round(float(points_2d[i, 0]) * 1000.0))
            py = int(round(float(points_2d[i, 1]) * 1000.0))
        h = zlib.crc32(f"{sid}:{px}:{py}".encode("utf-8")) & 0xFFFFFFFF
        hx = ((h & 0xFFFF) / 65535.0) - 0.5
        hy = (((h >> 16) & 0xFFFF) / 65535.0) - 0.5
        jitter[i, 0] = hx * scale
        jitter[i, 1] = hy * scale
    return jitter


def _set_plot_viewport(ax, points_2d: np.ndarray):
    if points_2d.size == 0:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        return

    x = points_2d[:, 0]
    y = points_2d[:, 1]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_span = x_max - x_min
    y_span = y_max - y_min
    span = max(x_span, y_span, max(0.0, PLOT_MIN_SPAN))

    if span <= 0:
        # Degenerate case: all points identical.
        eps = 1e-6
        span = eps
    pad = span * max(0.0, PLOT_PADDING_RATIO)
    half = (span / 2.0) + pad
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def _build_embedding_plot_png(app_instance: FastAPI, method: str | None = None) -> bytes:
    with app_instance.state.plot_lock:
        subject_ids = list(app_instance.state.plot_subject_ids)
        vectors = [v.copy() for v in app_instance.state.plot_vectors]
    method = _normalize_plot_method(method)
    method_used = method
    points_2d_plot = None
    ids_arr = None

    if vectors:
        subject_ids, X = _sanitize_plot_vectors(subject_ids, vectors)
    else:
        X = np.empty((0, 2), dtype=np.float32)

    if X.shape[0] > 0:
        if method == "pca":
            points_2d, method_used = _project_vectors_pca_stable(app_instance, X)
        else:
            points_2d, method_used = _project_vectors_2d(X, method=method)
        if points_2d.shape[0] != len(subject_ids):
            take = min(points_2d.shape[0], len(subject_ids))
            points_2d = points_2d[-take:, :]
            subject_ids = subject_ids[-take:]
        base_span = 0.0
        if points_2d.size > 0:
            base_span = float(
                max(
                    np.ptp(points_2d[:, 0]) if points_2d.shape[0] > 1 else 0.0,
                    np.ptp(points_2d[:, 1]) if points_2d.shape[0] > 1 else 0.0,
                )
            )
        points_2d_plot = points_2d + _stable_jitter(subject_ids, points_2d, base_span)
        ids_arr = np.asarray(subject_ids, dtype=object)

    # pyplot has global mutable state; lock keeps concurrent UMAP/t-SNE renders stable.
    with _MATPLOTLIB_RENDER_LOCK:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
        ax.grid(alpha=0.25)

        if points_2d_plot is None or ids_arr is None:
            ax.set_title(f"Palm Vector DB ({method_used.upper()})")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.text(0.5, 0.5, "No vectors yet", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.set_title(f"Palm Vector DB ({method_used.upper()})")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")

            unique_ids = np.unique(ids_arr)
            for uid in unique_ids:
                mask = ids_arr == uid
                ax.scatter(
                    points_2d_plot[mask, 0],
                    points_2d_plot[mask, 1],
                    s=24,
                    c=[_color_for_subject_id(str(uid))],
                    label=f"ID {uid}",
                    alpha=0.65,
                    edgecolors="none",
                )

            _set_plot_viewport(ax, points_2d_plot)

            if unique_ids.size <= 12:
                ax.legend(loc="best", frameon=False, fontsize=8)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()


def _build_plot_error_png(method: str, message: str) -> bytes:
    title = _normalize_plot_method(method).upper()
    msg = str(message).strip() or "unknown error"
    if len(msg) > 220:
        msg = msg[:217] + "..."

    with _MATPLOTLIB_RENDER_LOCK:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
        ax.set_title(f"Palm Vector DB ({title})")
        ax.set_axis_off()
        ax.text(
            0.02,
            0.98,
            "Plot render error",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            color="#b00020",
        )
        ax.text(
            0.02,
            0.86,
            msg,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#333333",
            wrap=True,
        )
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()


def _services_status(app_instance: FastAPI):
    with app_instance.state.register_lock:
        register_state = {
            "enabled": app_instance.state.register_enabled,
            "target_id": app_instance.state.register_target_id,
            "insert_count": app_instance.state.register_insert_count,
            "last_error": app_instance.state.register_last_error,
        }
        verify_state = {
            "enabled": app_instance.state.verify_enabled,
            "threshold": app_instance.state.verify_threshold,
            "last_result": app_instance.state.verify_last_result,
            "last_error": app_instance.state.verify_last_error,
        }
    with app_instance.state.plot_lock:
        plot_state = {
            "points": len(app_instance.state.plot_subject_ids),
            "max_points": "unlimited" if PLOT_MAX_POINTS <= 0 else PLOT_MAX_POINTS,
            "method": EMBED_VIZ_METHOD,
            "methods": list(PLOT_METHODS),
            "metric": EMBED_VIZ_METRIC,
            "umap_available": umap is not None,
            "version": app_instance.state.plot_version,
            "keep_all_samples": REGISTER_KEEP_ALL_SAMPLES,
            "padding_ratio": PLOT_PADDING_RATIO,
            "min_span": PLOT_MIN_SPAN,
            "jitter_ratio": PLOT_JITTER_RATIO,
            "pca_basis_ready": app_instance.state.plot_pca_basis_ready,
        }
    return {
        "triton": {
            "grpc_url": app_instance.state.triton_grpc_url,
            "model_name": app_instance.state.triton_model_name,
            "input_name": app_instance.state.triton_input_name,
            "output_name": app_instance.state.triton_output_name,
            "connected": app_instance.state.triton_client is not None,
            "error": app_instance.state.triton_error,
        },
        "qdrant": {
            "host": app_instance.state.qdrant_host,
            "port": app_instance.state.qdrant_port,
            "grpc_port": app_instance.state.qdrant_grpc_port,
            "collection": app_instance.state.qdrant_collection,
            "vector_size": FEATURE_VECTOR_SIZE,
            "distance": QDRANT_DISTANCE,
            "connected": app_instance.state.qdrant_client is not None,
            "error": app_instance.state.qdrant_error,
        },
        "register": register_state,
        "verify": verify_state,
        "plot": plot_state,
    }


def register_start_sync(register_id_text: str) -> str:
    register_id_text = (register_id_text or "").strip()
    if not register_id_text:
        return "Please enter ID."

    try:
        register_id = int(register_id_text)
    except ValueError:
        return "ID must be an integer."

    if app.state.triton_client is None or app.state.qdrant_client is None:
        return "Register unavailable: Triton or Qdrant not connected."

    with app.state.register_lock:
        app.state.register_enabled = True
        app.state.register_target_id = register_id
        app.state.register_last_error = None
        app.state.register_last_frame_id = -1
        app.state.register_insert_count = 0
        app.state.verify_enabled = False
        app.state.verify_last_frame_id = -1
        app.state.verify_last_result = "Verify OFF (Register mode active)."

    return (
        f"Register ON for ID={register_id}. "
        f"Upserting every {ROI_EVERY_N_FRAMES} frame(s) when ROI exists. "
        f"keep_all_samples={int(REGISTER_KEEP_ALL_SAMPLES)}"
    )


def register_update_id_sync(register_id_text: str) -> str:
    register_id_text = (register_id_text or "").strip()
    if not register_id_text:
        return "Please enter ID."

    try:
        register_id = int(register_id_text)
    except ValueError:
        return "ID must be an integer."

    with app.state.register_lock:
        app.state.register_target_id = register_id
        is_running = app.state.register_enabled

    if is_running:
        return (
            f"Target ID updated to {register_id}. "
            "New upserts will use this ID immediately."
        )
    return f"Target ID set to {register_id}. Click Register to start."


def _query_top1_euclid(app_instance: FastAPI, query_vec: np.ndarray):
    helper = app_instance.state.qdrant_client
    if helper is None:
        return None

    query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    if query.size == 0:
        return None

    best = None
    offset = None
    batch_size = 256
    while True:
        points, offset = helper.client.scroll(
            collection_name=app_instance.state.qdrant_collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        for point in points:
            vec = _extract_qdrant_vector(point)
            if vec is None or vec.shape[0] != query.shape[0]:
                continue
            dist = float(np.linalg.norm(vec - query))
            if best is None or dist < best["distance"]:
                payload = point.payload or {}
                subject_id = payload.get("subject_id", payload.get("label", None))
                best = {
                    "point_id": point.id,
                    "subject_id": None if subject_id is None else str(subject_id),
                    "distance": dist,
                    "payload": payload,
                }

        if offset is None:
            break

    return best


def _set_verify_status(app_instance: FastAPI, msg: str):
    with app_instance.state.register_lock:
        app_instance.state.verify_last_result = msg
        app_instance.state.verify_last_error = None


def _set_verify_error(app_instance: FastAPI, msg: str):
    with app_instance.state.register_lock:
        app_instance.state.verify_last_error = msg
        app_instance.state.verify_last_result = msg


def _verify_embedding(app_instance: FastAPI, embedding: np.ndarray, threshold: float) -> str:
    best = _query_top1_euclid(app_instance, embedding)
    if best is None:
        msg = "Verify failed: No vectors in database."
        _set_verify_error(app_instance, msg)
        return msg

    dist = float(best["distance"])
    sid = best["subject_id"] if best["subject_id"] is not None else str(best["point_id"])
    if dist < threshold:
        msg = (
            f"Verified: ID={sid} (distance={dist:.4f}, "
            f"threshold={threshold:.4f}, rule: distance < threshold)"
        )
        _set_verify_status(app_instance, msg)
        return msg

    msg = (
        f"No match: top1 ID={sid} (distance={dist:.4f}, "
        f"threshold={threshold:.4f}, rule: distance < threshold)"
    )
    _set_verify_status(app_instance, msg)
    return msg


def verify_start_sync(threshold_input):
    try:
        threshold = float(threshold_input)
    except Exception:
        threshold = 35.0

    if app.state.triton_client is None or app.state.qdrant_client is None:
        return "Verify unavailable: Triton or Qdrant not connected.", "Register state unchanged."

    with app.state.register_lock:
        app.state.register_enabled = False
        app.state.verify_enabled = True
        app.state.verify_threshold = threshold
        app.state.verify_last_frame_id = -1
        app.state.verify_last_error = None
        app.state.verify_last_result = (
            f"Verify ON (top1 Euclid, threshold={threshold:.4f}, rule: distance < threshold)"
        )

    return (
        app.state.verify_last_result,
        "Register OFF (stopped inserting). Verify mode is running in real time.",
    )


def get_verify_status_sync() -> str:
    with app.state.register_lock:
        return app.state.verify_last_result


def _get_verify_status_text(app_instance: FastAPI) -> str:
    with app_instance.state.register_lock:
        err = app_instance.state.verify_last_error
        result = app_instance.state.verify_last_result
    if err:
        return str(err)
    return str(result)


def _claim_verify_threshold_for_frame(app_instance: FastAPI, frame_id: int):
    with app_instance.state.register_lock:
        if not app_instance.state.verify_enabled:
            return None
        threshold = float(app_instance.state.verify_threshold)
        if frame_id <= app_instance.state.verify_last_frame_id:
            return None
        app_instance.state.verify_last_frame_id = frame_id
        return threshold


def _render_verify_status_panel(text: str, width: int = 980, height: int = 86) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (210, 210, 210), 1)

    msg = (text or "").strip() or "Verify OFF."
    lines = textwrap.wrap(msg, width=120) or [msg]
    y = 28
    for line in lines[:2]:
        cv2.putText(
            panel,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return panel


async def stream_verify_status_frames():
    loop = asyncio.get_running_loop()
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    while not app.state.shutdown_event.is_set():
        text = _get_verify_status_text(app)
        panel = await loop.run_in_executor(None, _render_verify_status_panel, text)
        ok, buf = await loop.run_in_executor(
            None,
            cv2.imencode,
            ".jpg",
            panel,
            encode_params,
        )
        if ok and buf is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        await asyncio.sleep(0.12)


def _claim_register_target_for_frame(app_instance: FastAPI, frame_id: int):
    with app_instance.state.register_lock:
        if not app_instance.state.register_enabled:
            return None
        target_id = app_instance.state.register_target_id
        if target_id is None:
            return None
        if frame_id <= app_instance.state.register_last_frame_id:
            return None
        app_instance.state.register_last_frame_id = frame_id
        return target_id


def _register_embedding(app_instance: FastAPI, target_id: int, embedding: np.ndarray):
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    vector = vec.tolist()
    now_ts = time.time()
    point_id = time.time_ns()
    if not REGISTER_KEEP_ALL_SAMPLES:
        point_id = int(target_id)
    payload = {
        "label": str(target_id),
        "subject_id": str(target_id),
        "updated_at": now_ts,
    }
    app_instance.state.qdrant_client.insert_vectors(
        collection_name=app_instance.state.qdrant_collection,
        vectors=[vector],
        ids=[point_id],
        payloads=[payload],
    )
    _append_plot_point(app_instance, str(target_id), vec)

    with app_instance.state.register_lock:
        app_instance.state.register_insert_count += 1
        app_instance.state.register_last_error = None


def _set_register_error(app_instance: FastAPI, err: str):
    with app_instance.state.register_lock:
        app_instance.state.register_last_error = err


def _get_cached_plot_png(app_instance: FastAPI, method: str, force: bool = False) -> bytes:
    method = _normalize_plot_method(method)
    with app_instance.state.plot_lock:
        version = app_instance.state.plot_version
        cached = app_instance.state.plot_png_cache.get(method)
        if (not force) and cached is not None and cached[0] == version:
            return cached[1]

    render_lock = app_instance.state.plot_render_locks.get(method)
    if render_lock is None:
        render_lock = app_instance.state.plot_render_locks["pca"]

    with render_lock:
        with app_instance.state.plot_lock:
            version = app_instance.state.plot_version
            cached = app_instance.state.plot_png_cache.get(method)
            if (not force) and cached is not None and cached[0] == version:
                return cached[1]

        try:
            png = _build_embedding_plot_png(app_instance, method=method)
        except Exception as exc:
            with app_instance.state.plot_lock:
                fallback = app_instance.state.plot_png_cache.get(method)
                if fallback is not None and fallback[1]:
                    return fallback[1]
            return _build_plot_error_png(method, str(exc))

        with app_instance.state.plot_lock:
            app_instance.state.plot_png_cache[method] = (version, png)
        return png


def _manual_plot_html(method: str, force: bool = True) -> str:
    method = _normalize_plot_method(method)
    ts = time.time_ns()
    title = method.upper() if method != "tsne" else "t-SNE"
    force_q = "1" if force else "0"
    return (
        '<div style="width:100%; min-width:360px; max-width:420px;">'
        f'<img src="/plot/embeddings?method={method}&force={force_q}&t={ts}" '
        f'alt="{title} plot" '
        'style="display:block; width:100%; border:1px solid #ccc; background:#fff;" '
        f'onerror="this.outerHTML=\'<div style=&quot;padding:10px;color:#900;border:1px solid #caa;background:#fee;&quot;>{title} render failed</div>\'">'
        "</div>"
    )


def refresh_manifold_plots_sync():
    _get_cached_plot_png(app, "umap", force=True)
    _get_cached_plot_png(app, "tsne", force=True)
    return _manual_plot_html("umap", force=False), _manual_plot_html("tsne", force=False)


def disable_refresh_manifold_btn_sync():
    return gr.update(interactive=False, value="Refreshing UMAP + t-SNE...")


def enable_refresh_manifold_btn_sync():
    return gr.update(interactive=True, value="Refresh UMAP + t-SNE")


async def stream_plot_frames(method: str = "pca"):
    loop = asyncio.get_running_loop()
    last_version = -1
    cached_png = None
    method = _normalize_plot_method(method)

    while not app.state.shutdown_event.is_set():
        with app.state.plot_lock:
            version = app.state.plot_version

        if cached_png is None or version != last_version:
            cached_png = await loop.run_in_executor(
                None,
                _get_cached_plot_png,
                app,
                method,
            )
            last_version = version

        if cached_png is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/png\r\n\r\n"
                + cached_png
                + b"\r\n"
            )

        await asyncio.sleep(max(0.05, PLOT_REFRESH_MS / 1000.0))


async def stream_frames(mode: str = "raw"):
    loop = asyncio.get_running_loop()
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    last_roi = _placeholder("No ROI", ROI_DISPLAY_SIZE, ROI_DISPLAY_SIZE)
    last_preprocessed_roi = _placeholder(
        "No Preprocessed ROI", ROI_DISPLAY_SIZE, ROI_DISPLAY_SIZE
    )
    last_processed_id = -1

    while not app.state.shutdown_event.is_set():
        frame, frame_id = _get_latest_frame(app)
        if frame is None:
            if mode == "raw":
                output = _placeholder(
                    "Waiting for camera...",
                    CAMERA_DISPLAY_WIDTH,
                    CAMERA_DISPLAY_HEIGHT,
                )
            else:
                output = _placeholder(
                    "Waiting for ROI...", ROI_DISPLAY_SIZE, ROI_DISPLAY_SIZE
                )
        else:
            if mode == "raw":
                output = frame
            else:
                should_run_roi = (
                    last_processed_id < 0
                    or frame_id - last_processed_id >= ROI_EVERY_N_FRAMES
                )
                if should_run_roi:
                    verify_threshold = None
                    if mode == "preprocessed":
                        verify_threshold = _claim_verify_threshold_for_frame(app, frame_id)
                    roi = await loop.run_in_executor(None, extract_palm_roi, frame)
                    if roi is not None and roi.size > 0:
                        last_roi = cv2.resize(
                            roi,
                            (ROI_DISPLAY_SIZE, ROI_DISPLAY_SIZE),
                            interpolation=cv2.INTER_AREA,
                        )
                        if mode == "preprocessed":
                            batch, preprocessed = await loop.run_in_executor(
                                None,
                                _preprocess_roi,
                                roi,
                            )
                            if preprocessed is not None and preprocessed.size > 0:
                                last_preprocessed_roi = cv2.resize(
                                    preprocessed,
                                    (ROI_DISPLAY_SIZE, ROI_DISPLAY_SIZE),
                                    interpolation=cv2.INTER_AREA,
                                )
                            target_id = _claim_register_target_for_frame(app, frame_id)
                            needs_embedding = (
                                batch is not None
                                and (target_id is not None or verify_threshold is not None)
                            )
                            if needs_embedding:
                                try:
                                    embedding = await loop.run_in_executor(
                                        None,
                                        _infer_embedding_from_batch,
                                        app,
                                        batch,
                                    )
                                    if embedding is not None:
                                        if target_id is not None:
                                            await loop.run_in_executor(
                                                None,
                                                _register_embedding,
                                                app,
                                                target_id,
                                                embedding,
                                            )
                                        if verify_threshold is not None:
                                            await loop.run_in_executor(
                                                None,
                                                _verify_embedding,
                                                app,
                                                embedding,
                                                float(verify_threshold),
                                            )
                                    else:
                                        if target_id is not None:
                                            _set_register_error(
                                                app,
                                                "Embedding inference returned empty.",
                                            )
                                        if verify_threshold is not None:
                                            _set_verify_error(
                                                app,
                                                "Verify failed: Embedding inference returned empty.",
                                            )
                                except Exception as exc:
                                    if target_id is not None:
                                        _set_register_error(app, str(exc))
                                    if verify_threshold is not None:
                                        _set_verify_error(app, f"Verify failed: {exc}")
                    elif verify_threshold is not None:
                        _set_verify_error(app, "Verify failed: No palm ROI detected.")
                    last_processed_id = frame_id
                if mode == "preprocessed":
                    output = last_preprocessed_roi
                else:
                    output = last_roi

        ok, buf = await loop.run_in_executor(
            None,
            cv2.imencode,
            ".jpg",
            output,
            encode_params,
        )
        if ok and buf is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

        await asyncio.sleep(0.001)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    app_instance.state.cap_map = {}
    app_instance.state.cam_idx = 0
    app_instance.state.frame_lock = threading.Lock()
    app_instance.state.register_lock = threading.Lock()
    app_instance.state.plot_lock = threading.Lock()
    app_instance.state.plot_render_locks = {m: threading.Lock() for m in PLOT_METHODS}
    app_instance.state.latest_frame = None
    app_instance.state.latest_frame_id = 0
    app_instance.state.capture_stop = threading.Event()
    app_instance.state.shutdown_event = asyncio.Event()
    app_instance.state.register_enabled = False
    app_instance.state.register_target_id = None
    app_instance.state.register_insert_count = 0
    app_instance.state.register_last_error = None
    app_instance.state.register_last_frame_id = -1
    app_instance.state.verify_enabled = False
    app_instance.state.verify_threshold = 35.0
    app_instance.state.verify_last_error = None
    app_instance.state.verify_last_result = "Verify OFF."
    app_instance.state.verify_last_frame_id = -1
    app_instance.state.plot_subject_ids = []
    app_instance.state.plot_vectors = []
    app_instance.state.plot_pca_mean = None
    app_instance.state.plot_pca_components = None
    app_instance.state.plot_pca_basis_ready = False
    app_instance.state.plot_version = 0
    app_instance.state.plot_png_cache = {}

    _init_backends(app_instance)

    cap0 = _open_camera(0)
    if cap0 is not None:
        app_instance.state.cap_map[0] = cap0

    app_instance.state.capture_thread = threading.Thread(
        target=_capture_loop,
        args=(app_instance,),
        daemon=True,
    )
    app_instance.state.capture_thread.start()

    yield

    app_instance.state.shutdown_event.set()
    app_instance.state.capture_stop.set()
    if app_instance.state.capture_thread.is_alive():
        app_instance.state.capture_thread.join(timeout=1.0)

    for cap in app_instance.state.cap_map.values():
        if cap is not None and cap.isOpened():
            cap.release()


app = FastAPI(lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def go_ui():
    return RedirectResponse(url="/ui")


@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")


@app.get("/healthz/services")
async def healthz_services():
    return _services_status(app)


@app.get("/verify/status", include_in_schema=False)
async def verify_status():
    return PlainTextResponse(_get_verify_status_text(app))


@app.get("/verify/status-stream", include_in_schema=False)
async def verify_status_stream():
    return StreamingResponse(
        stream_verify_status_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/plot/embeddings", include_in_schema=False)
async def plot_embeddings(t: str = "", method: str = "pca", force: int = 0):
    _ = t
    loop = asyncio.get_running_loop()
    png = await loop.run_in_executor(None, _get_cached_plot_png, app, method, bool(force))
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/plot/stream", include_in_schema=False)
async def plot_stream(method: str = "pca"):
    method = _normalize_plot_method(method)
    return StreamingResponse(
        stream_plot_frames(method=method),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/plot/stream/{method}", include_in_schema=False)
async def plot_stream_method(method: str):
    method = _normalize_plot_method(method)
    return StreamingResponse(
        stream_plot_frames(method=method),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/plot/version", include_in_schema=False)
async def plot_version():
    with app.state.plot_lock:
        version = app.state.plot_version
    return PlainTextResponse(str(version))


@app.get("/identify/latest")
async def identify_latest(top_k: int = 5):
    if top_k <= 0:
        top_k = 1

    if app.state.triton_client is None or app.state.qdrant_client is None:
        return {
            "ok": False,
            "error": "Triton or Qdrant is not connected",
            "services": _services_status(app),
        }

    frame, _ = _get_latest_frame(app)
    if frame is None:
        return {"ok": False, "error": "No camera frame available"}

    loop = asyncio.get_running_loop()
    roi = await loop.run_in_executor(None, extract_palm_roi, frame)
    if roi is None or roi.size == 0:
        return {"ok": False, "error": "No palm ROI detected"}

    embedding = await loop.run_in_executor(None, _infer_embedding, app, roi)
    if embedding is None:
        return {"ok": False, "error": "Embedding inference failed"}

    hits = await loop.run_in_executor(
        None,
        app.state.qdrant_client.search,
        app.state.qdrant_collection,
        embedding.tolist(),
        top_k,
    )
    results = [
        {
            "id": hit.id,
            "score": float(hit.score),
            "payload": hit.payload,
        }
        for hit in hits
    ]
    return {
        "ok": True,
        "collection": app.state.qdrant_collection,
        "top_k": top_k,
        "results": results,
    }


@app.get("/video/raw", include_in_schema=False)
async def raw_feed():
    return StreamingResponse(
        stream_frames(mode="raw"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/roi", include_in_schema=False)
async def roi_feed():
    return StreamingResponse(
        stream_frames(mode="roi"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/roi-preprocessed", include_in_schema=False)
async def preprocessed_roi_feed():
    return StreamingResponse(
        stream_frames(mode="preprocessed"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


with gr.Blocks(title="Palm ROI Extraction") as ui:
    gr.Markdown("# Palm ROI Extraction (Smooth Stream)")

    with gr.Row():
        cam_dd = gr.Dropdown(
            choices=[str(i) for i in range(4)],
            value="0",
            label="Camera Index",
        )
        cam_status = gr.Textbox(label="Camera Status", interactive=False)

    cam_dd.change(fn=switch_camera_sync, inputs=[cam_dd], outputs=[cam_status], queue=False)

    with gr.Row():
        register_id_tb = gr.Textbox(
            label="Register ID",
            placeholder="Enter ID (integer)",
        )
        register_btn = gr.Button("Register", variant="primary")
        verify_btn = gr.Button("Verify", variant="secondary")
        verify_threshold_tb = gr.Number(
            label="Verify Threshold (Euclid)",
            value=35.0,
            precision=4,
        )
    with gr.Row():
        register_status = gr.Textbox(label="Register Status", interactive=False)
        verify_status = gr.Textbox(label="Verify Status", interactive=False)
    gr.HTML(
        """
<div style="font-weight:600; margin-bottom:6px;">Verify Status (Live)</div>
<img src="/verify/status-stream" alt="Verify status stream"
     style="display:block; width:100%; max-width:980px; border:1px solid #ddd; background:#f5f5f5;">
"""
    )

    register_btn.click(
        fn=register_start_sync,
        inputs=[register_id_tb],
        outputs=[register_status],
        queue=False,
    )
    register_id_tb.change(
        fn=register_update_id_sync,
        inputs=[register_id_tb],
        outputs=[register_status],
        queue=False,
    )
    verify_btn.click(
        fn=verify_start_sync,
        inputs=[verify_threshold_tb],
        outputs=[verify_status, register_status],
        queue=False,
    )

    gr.HTML(
        f"""
<div style="display:flex; gap:16px; align-items:flex-start; flex-wrap:nowrap; overflow-x:auto;">
  <div style="flex:0 0 auto;">
    <div style="font-weight:600; margin-bottom:8px;">Live Camera</div>
    <img src="/video/raw" alt="Raw stream"
         style="display:block; width:{CAMERA_DISPLAY_WIDTH}px; height:auto; border:1px solid #ccc; background:#111;">
  </div>
  <div style="flex:0 0 auto; display:flex; flex-direction:column; gap:12px;">
    <div>
      <div style="font-weight:600; margin-bottom:8px;">Palm ROI</div>
      <img src="/video/roi" alt="ROI stream"
           style="display:block; width:{ROI_DISPLAY_SIZE}px; height:{ROI_DISPLAY_SIZE}px; border:1px solid #ccc; background:#111;">
    </div>
    <div>
      <div style="font-weight:600; margin-bottom:8px;">Preprocessed ROI</div>
      <img src="/video/roi-preprocessed" alt="Preprocessed ROI stream"
           style="display:block; width:{ROI_DISPLAY_SIZE}px; height:{ROI_DISPLAY_SIZE}px; border:1px solid #ccc; background:#111;">
    </div>
  </div>
</div>
"""
    )

    gr.Markdown("### Palm Vector Database (PCA / UMAP / t-SNE)")
    with gr.Row(equal_height=False):
        with gr.Column():
            gr.Markdown("#### PCA (Live)")
            gr.HTML(
                '<img src="/plot/stream/pca" alt="PCA plot" '
                'style="display:block; width:100%; min-width:360px; max-width:420px; '
                'border:1px solid #ccc; background:#fff;">'
            )
        with gr.Column():
            gr.Markdown("#### UMAP (Manual Refresh)")
            umap_plot = gr.HTML(value=_manual_plot_html("umap"))
        with gr.Column():
            gr.Markdown("#### t-SNE (Manual Refresh)")
            tsne_plot = gr.HTML(value=_manual_plot_html("tsne"))
    refresh_manifold_btn = gr.Button(
        "Initializing UMAP + t-SNE...",
        interactive=False,
    )

    ui.load(
        fn=disable_refresh_manifold_btn_sync,
        outputs=[refresh_manifold_btn],
        queue=False,
    ).then(
        fn=refresh_manifold_plots_sync,
        outputs=[umap_plot, tsne_plot],
        queue=True,
    ).then(
        fn=enable_refresh_manifold_btn_sync,
        outputs=[refresh_manifold_btn],
        queue=False,
    )
    refresh_manifold_btn.click(
        fn=disable_refresh_manifold_btn_sync,
        outputs=[refresh_manifold_btn],
        queue=False,
    ).then(
        fn=refresh_manifold_plots_sync,
        outputs=[umap_plot, tsne_plot],
        queue=True,
    ).then(
        fn=enable_refresh_manifold_btn_sync,
        outputs=[refresh_manifold_btn],
        queue=False,
    )


gr.mount_gradio_app(app, ui, path="/ui")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7000)
