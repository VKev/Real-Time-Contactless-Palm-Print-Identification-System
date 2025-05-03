import time
import queue
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse, PlainTextResponse
import gradio as gr
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_444

from roi_extraction.util import extract_palm_roi
from roi_extraction.gesture import OpenPalmDetector
from utils.preprocess import preprocess
from utils.triton import TritonClient
from utils.qdrant import QdrantHelper

import requests

def push_log(app: FastAPI, msg: str):
    """Update the latest log message in app state"""
    app.state.latest_log = msg

def configure_cap_full_hd(cap: cv2.VideoCapture):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


def get_cap(app: FastAPI) -> cv2.VideoCapture:
    return app.state.cap_map[app.state.cam_idx]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cap0 = cv2.VideoCapture(0)
    configure_cap_full_hd(cap0)
    app.state.cap_map = {0: cap0}
    app.state.cam_idx = 0

    app.state.triton = TritonClient("localhost:8001", verbose=False)
    app.state.gesture = OpenPalmDetector(
        "roi_extraction/gesture_recognizer/gesture_recognizer.task"
    )
    qd = QdrantHelper(host="localhost", port=6333, grpc_port=6334, prefer_grpc=True)
    qd.ensure_collection("palm_vectors", vector_size=128)
    app.state.qdrant = qd
    
    app.state.latest_log = "System initialized"

    app.state.is_register = False
    app.state.current_label = None

    yield

    for c in app.state.cap_map.values():
        c.release()


app = FastAPI(lifespan=lifespan)

doneQ: "queue.Queue[tuple[int, object]]" = queue.Queue()


def inference_callback(user_data, result, error):
    """Modified callback to update logs"""
    idx, t_in, app = user_data  # Now receives app instance
    t_out = time.time()
    latency = (t_out - t_in) * 1000

    if error:
        error_msg = f"[#{idx}] ERROR @ {t_out:.3f}s  {error}"
        push_log(app, error_msg)
    else:
        vec = result.as_numpy("OUTPUT__0").squeeze().astype(float).tolist()

        if app.state.is_register:
            label = app.state.current_label or f"palm_0"
            log_msg = (
                f"[#{idx}] REGISTER @ {t_out:.3f}s ({latency:.1f}ms) "
                f"label: {label}"
            )
            app.state.qdrant.insert_vectors(
                collection_name="palm_vectors",
                vectors=[vec],
                ids=[idx],
                payloads=[{"label": label}],
            )
        else:
            matches = app.state.qdrant.search("palm_vectors", vec, top_k=5)
            if matches:
                best = matches[0]
                # if best score is above threshold, treat as unregistered
                if best.score > 80:
                    log_msg = (
                        f"[#{idx}] VERIFY @ {t_out:.3f}s ({latency:.1f}ms) "
                        f"Unregistered"
                    )
                else:
                    log_msg = (
                        f"[#{idx}] VERIFY @ {t_out:.3f}s ({latency:.1f}ms) "
                        f"match: {best.payload.get('label')} "
                        f"({best.score:.2f})"
                    )
            else:
                log_msg = (
                    f"[#{idx}] VERIFY @ {t_out:.3f}s ({latency:.1f}ms) "
                    f"No matches"
                )
        
        push_log(app, log_msg)

    doneQ.put((idx, result))


def stream(request: Request, triton: TritonClient, to_roi: bool = False):
    """Modified stream function to pass app instance"""
    counter = 0
    while True:
        ok, frame = get_cap(request.app).read()
        if not ok:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # Clear done queue
        while True:
            try: doneQ.get_nowait()
            except queue.Empty: break

        if to_roi:
            roi = extract_palm_roi(frame)
            if roi is not None:
                batch, img = preprocess(roi)
                t_in = time.time()
                # Pass app instance in user_data
                triton.infer_async(
                    model_name="feature_extraction",
                    inputs={"INPUT__0": batch},
                    callback=inference_callback,
                    user_data=(counter, t_in, request.app),  # Added app instance
                )
                counter += 1
                frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)

        _, buf = cv2.imencode(".jpg", frame)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.get("/logs/latest")
async def get_latest_log(request: Request):
    """Endpoint to retrieve latest log"""
    return PlainTextResponse(request.app.state.latest_log)

@app.get("/", include_in_schema=False)
def go_ui():
    return RedirectResponse(url="/ui")


@app.post("/register")
async def register(request: Request, body: dict = Body(...)):
    """Switch the system to *registration* mode and remember the label."""
    label = body.get("label", f"palm_{int(time.time())}")
    request.app.state.is_register = True
    request.app.state.current_label = label
    return {"mode": "register", "label": label}


@app.post("/verify")
async def verify(request: Request):
    """Switch the system to *verification* mode (no DB insertions)."""
    request.app.state.is_register = False
    request.app.state.current_label = None
    return {"mode": "verify"}


@app.get("/video/raw", include_in_schema=False)
def raw_feed(request: Request):
    return StreamingResponse(
        stream(request, request.app.state.triton, to_roi=False),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/roi", include_in_schema=False)
def roi_feed(request: Request):
    return StreamingResponse(
        stream(request, request.app.state.triton, to_roi=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

def switch_camera(idx: int) -> str:
    idx = int(idx)
    prev_idx = app.state.cam_idx

    if idx == prev_idx:
        return f"ℹ️ Already using camera {idx}"

    cap_new = cv2.VideoCapture(idx)
    configure_cap_full_hd(cap_new)
    if not cap_new.isOpened():
        cap_new.release()
        return f"❌ Camera {idx} could not be opened."

    app.state.cap_map[idx] = cap_new
    app.state.cam_idx = idx

    if prev_idx in app.state.cap_map:
        app.state.cap_map[prev_idx].release()
        del app.state.cap_map[prev_idx]

    return f"✅ Switched to camera {idx}"

with gr.Blocks(title="Palm‑Print Identification System") as ui:
    gr.Markdown("### Palm‑Print Identification System")
    with gr.Row():
        cam_dd = gr.Dropdown(choices=[0, 1], value=0, label="Camera Selection")
        status = gr.Textbox(label="System Status", interactive=False)
        logs = gr.Textbox(label="Latest Event", interactive=False)
        
        log_timer = gr.Timer(value=0.1)
        log_timer.tick(
            fn=lambda t: app.state.latest_log,
            inputs=None,
            outputs=[logs]
        )
        
        cam_dd.change(switch_camera, cam_dd, status)

    with gr.Row():
        label_input = gr.Textbox(label="Registration Label", placeholder="Enter label…")

        reg_btn = gr.Button("Register")
        reg_resp = gr.JSON(label="Register Response")

        ver_btn = gr.Button("Verify")
        ver_resp = gr.JSON(label="Verify Response")

    def call_register(label):
        try:
            r = requests.post("http://localhost:7000/register", json={"label": label})
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def call_verify():
        try:
            r = requests.post("http://localhost:7000/verify")
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    reg_btn.click(fn=call_register, inputs=[label_input], outputs=[reg_resp])
    ver_btn.click(fn=call_verify, inputs=None,   outputs=[ver_resp])

    with gr.Row(equal_height=True):
        gr.HTML('<img src="/video/raw" style="max-width:1080px;border:1px solid #ccc;">')
        gr.HTML('<img src="/video/roi" style="max-width:480px;border:1px solid #ccc;">')

gr.mount_gradio_app(app, ui, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=7000)
