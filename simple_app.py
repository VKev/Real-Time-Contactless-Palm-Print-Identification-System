import asyncio
import threading
import time
from contextlib import asynccontextmanager

import cv2
import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, RedirectResponse, StreamingResponse

from roi_extraction import extract_palm_roi

JPEG_QUALITY = 100
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
ROI_EVERY_N_FRAMES = 1


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

    if old_cap is not None and old_cap.isOpened():
        old_cap.release()

    return f"Switched to camera {idx}"


async def stream_frames(to_roi: bool = False):
    loop = asyncio.get_running_loop()
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    last_roi = _placeholder("No ROI", 320, 320)
    last_processed_id = -1

    while not app.state.shutdown_event.is_set():
        frame, frame_id = _get_latest_frame(app)
        if frame is None:
            output = _placeholder("Waiting for camera...")
        else:
            if not to_roi:
                output = frame
            else:
                should_run_roi = (
                    last_processed_id < 0
                    or frame_id - last_processed_id >= ROI_EVERY_N_FRAMES
                )
                if should_run_roi:
                    roi = await loop.run_in_executor(None, extract_palm_roi, frame)
                    if roi is not None and roi.size > 0:
                        last_roi = roi
                    last_processed_id = frame_id
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
    app_instance.state.latest_frame = None
    app_instance.state.latest_frame_id = 0
    app_instance.state.capture_stop = threading.Event()
    app_instance.state.shutdown_event = asyncio.Event()

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


@app.get("/video/raw", include_in_schema=False)
async def raw_feed():
    return StreamingResponse(
        stream_frames(to_roi=False),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/roi", include_in_schema=False)
async def roi_feed():
    return StreamingResponse(
        stream_frames(to_roi=True),
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

    with gr.Row(equal_height=False):
        with gr.Column():
            gr.Markdown("### Live Camera")
            gr.HTML(
                '<img src="/video/raw" alt="Raw stream" '
                'style="width:100%; max-width:720px; border:1px solid #ccc; background:#111;">'
            )
        with gr.Column():
            gr.Markdown("### Palm ROI")
            gr.HTML(
                '<img src="/video/roi" alt="ROI stream" '
                'style="width:100%; max-width:360px; border:1px solid #ccc; background:#111;">'
            )


gr.mount_gradio_app(app, ui, path="/ui")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7000)
