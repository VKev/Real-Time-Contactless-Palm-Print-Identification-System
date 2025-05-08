import time
import asyncio
from contextlib import asynccontextmanager
import logging

import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse, PlainTextResponse
import gradio as gr
import requests

from roi_extraction.util import extract_palm_roi
from utils.preprocess import preprocess
from utils.triton import TritonClient
from utils.qdrant import QdrantHelper
from depth_estimation.utils import resize, interpolate_np, apply_depth_mask_np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

shutdown_event = asyncio.Event()
inference_queue: asyncio.Queue | None = None
depth_queue: asyncio.Queue | None = None

def push_log(app: FastAPI, msg: str):
    app.state.latest_log = msg
    logger.info(msg)

def configure_cap_full_hd(cap: cv2.VideoCapture):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logger.info(f"Attempted to set resolution to 1280x720")

def get_cap(app: FastAPI) -> cv2.VideoCapture | None:
    return app.state.cap_map.get(app.state.cam_idx)

async def process_inference_results(app: FastAPI):
    global inference_queue
    if not inference_queue:
        logger.error("Inference queue not initialized!")
        return

    loop = asyncio.get_running_loop()
    qdrant: QdrantHelper = app.state.qdrant

    logger.info("Starting inference result processing worker.")
    while not shutdown_event.is_set():
        idx, vec, t_in, is_register, label = await inference_queue.get()
        t_out = time.time()
        latency = (t_out - t_in) * 1000

        if is_register:
            current_label = label or f"palm_{int(time.time())}"
            log_msg = (
                f"[#{idx}] REGISTER @ ({latency:.1f}ms) "
                f"label: {current_label}"
            )
            await loop.run_in_executor(
                None,
                qdrant.insert_vectors,
                "palm_vectors", [vec], [idx], [{"label": current_label}]
            )
        else:
            matches = await loop.run_in_executor(
                None, qdrant.search, "palm_vectors", vec, 1
            )

            if matches:
                best = matches[0]
                MATCH_THRESHOLD = 80
                if best.score < MATCH_THRESHOLD:
                    log_msg = (
                        f"[#{idx}] VERIFY @ ({latency:.1f}ms) "
                        f"match: {best.payload.get('label')} "
                        f"({best.score:.4f})"
                    )
                else:
                    log_msg = (
                        f"[#{idx}] VERIFY @ ({latency:.1f}ms) "
                        f"Unregistered (Score: {best.score:.4f} >= {MATCH_THRESHOLD})"
                    )
            else:
                log_msg = (
                    f"[#{idx}] VERIFY @ ({latency:.1f}ms) "
                    f"No matches found")

        push_log(app, log_msg)
        inference_queue.task_done()

    logger.info("Inference result processing worker stopped.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_queue, depth_queue
    logger.info("Application startup...")

    app.state.cap_map = {}
    app.state.cam_idx = 0
    app.state.depth_threshold = 0.6  # Initialize depth threshold

    cap0 = cv2.VideoCapture(0)
    if cap0.isOpened():
        configure_cap_full_hd(cap0)
        app.state.cap_map[0] = cap0
        logger.info("Camera 0 opened successfully.")
    else:
        logger.error("Failed to open camera 0")
        cap0.release()

    app.state.triton = TritonClient("localhost:8001", verbose=False)
    logger.info("Triton client initialized.")

    qd = QdrantHelper(host="localhost", port=6333, grpc_port=6334, prefer_grpc=True)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, qd.ensure_collection, "palm_vectors", 128)
    app.state.qdrant = qd
    logger.info("Qdrant client initialized and collection ensured.")

    app.state.latest_log = "System initialized"
    app.state.is_register = False
    app.state.current_label = None
    app.state.background_removal_enabled = False

    inference_queue = asyncio.Queue()
    depth_queue = asyncio.Queue(maxsize=1)

    worker_task = asyncio.create_task(process_inference_results(app))
    logger.info("Background worker task created.")

    yield

    logger.info("Application shutdown...")
    shutdown_event.set()

    if worker_task:
        await asyncio.wait_for(worker_task, timeout=5.0)
        logger.info("Background worker task finished.")

    for cap in app.state.cap_map.values():
        if cap and cap.isOpened():
            cap.release()
    logger.info("Camera captures released.")

    if hasattr(app.state, 'triton') and app.state.triton and hasattr(app.state.triton, 'close'):
        logger.info("Triton client closed (or managed internally).")

    if hasattr(app.state, 'qdrant') and app.state.qdrant and hasattr(app.state.qdrant, 'close'):
        logger.info("Qdrant client closed (or managed internally).")

    logger.info("Application shutdown complete.")

app = FastAPI(lifespan=lifespan)

def inference_callback(user_data, result, error):
    global inference_queue
    if not inference_queue:
        logger.warning("Inference queue not available in callback. Discarding result.")
        return

    idx, t_in, is_register, current_label = user_data

    if error:
        logger.error(f"[Triton Callback Error #{idx}] {error}")
    else:
        if result is None:
            logger.error(f"[Triton Callback Error #{idx}] Received None result.")
            return

        vec_output = result.as_numpy("OUTPUT__0")
        if vec_output is None:
            logger.error(f"[Triton Callback Error #{idx}] 'OUTPUT__0' not found in result.")
            return

        vec = vec_output.squeeze().astype(float).tolist()
        inference_queue.put_nowait((idx, vec, t_in, is_register, current_label))

def run_triton_infer(triton, model_name, inputs):
    return triton.infer(model_name=model_name, inputs=inputs)

async def stream(request: Request, triton: TritonClient | None, to_roi: bool = False):
    global depth_queue
    if not triton:
        logger.error("Triton client not available for streaming.")
        err_img = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.putText(err_img, "Triton Error", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        ok, buf = cv2.imencode(".jpg", err_img)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        return

    counter = 0
    loop = asyncio.get_running_loop()
    latest_masked_frame = None

    while not shutdown_event.is_set():
        cap = get_cap(request.app)
        if not cap or not cap.isOpened():
            logger.warning("Camera not ready or closed. Yielding placeholder.")
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera Feed", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ok = True
            await asyncio.sleep(0.5)
        else:
            ok, frame = await loop.run_in_executor(None, cap.read)

        if not ok or frame is None:
            logger.warning("Failed to read frame from camera. Using placeholder.")
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "Frame Read Error", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ok = True
            await asyncio.sleep(0.1)

        current_frame_to_process = frame
        processed_frame_for_display = current_frame_to_process
        start = time.perf_counter()
        if request.app.state.background_removal_enabled:
            resized_input, (h, w) = await loop.run_in_executor(None, resize, current_frame_to_process, 252)

            depth_result = await loop.run_in_executor(
                None,
                run_triton_infer,
                triton,
                "depth_anything_v2",
                {"INPUT__0": resized_input}
            )
            depth_out = depth_result.get("OUTPUT__0")
            
            if depth_out is not None:
                depth_norm = await loop.run_in_executor(None, interpolate_np, depth_out, h, w)

                latest_masked_frame = await loop.run_in_executor(
                    None, apply_depth_mask_np, current_frame_to_process, depth_norm, 
                    request.app.state.depth_threshold  # Use dynamic threshold
                )
                current_frame_to_process = latest_masked_frame
                processed_frame_for_display = latest_masked_frame
                
            else:
                logger.warning("Depth inference failed. Skipping background removal for this frame.")
                processed_frame_for_display = current_frame_to_process
        else:
            processed_frame_for_display = current_frame_to_process

        img_to_encode = processed_frame_for_display

        if to_roi:
            roi = await loop.run_in_executor(None, extract_palm_roi, current_frame_to_process)

            if roi is not None and roi.size > 0:
                batch, processed_img_for_display = await loop.run_in_executor(None, preprocess, roi)

                if batch is not None and processed_img_for_display is not None:
                    t_in = time.time()
                    is_reg = request.app.state.is_register
                    curr_label = request.app.state.current_label

                    triton.infer_async(
                        model_name="feature_extraction",
                        inputs={"INPUT__0": batch},
                        callback=inference_callback,
                        user_data=(counter, t_in, is_reg, curr_label),
                    )
                    counter += 1

                    img_to_encode = cv2.cvtColor(processed_img_for_display, cv2.COLOR_GRAY2BGR)
            else:
                img_to_encode = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.putText(img_to_encode, "No ROI", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

        ok_encode, buf = await loop.run_in_executor(None, cv2.imencode, ".jpg", img_to_encode)

        end = time.perf_counter()
        elapsed_s = end - start
        print(f"Inference took {elapsed_s*1000:.1f} ms")
        
        if ok_encode and buf is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        else:
            logger.warning("Failed to encode frame to JPEG.")

        await asyncio.sleep(0.0005)

@app.get("/logs/latest")
async def get_latest_log(request: Request):
    log = request.app.state.latest_log or "No logs yet."
    return PlainTextResponse(log)

@app.get("/", include_in_schema=False)
async def go_ui():
    return RedirectResponse(url="/ui")

@app.post("/register")
async def register(request: Request, body: dict = Body(...)):
    label = body.get("label")
    if not label or not isinstance(label, str) or not label.strip():
        label = f"palm_{int(time.time())}"
        logger.warning(f"No valid label provided for registration, using default: {label}")
    else:
        label = label.strip()

    request.app.state.is_register = True
    request.app.state.current_label = label
    log_msg = f"Switched to REGISTER mode. Label: {label}"
    push_log(request.app, log_msg)
    return {"mode": "register", "label": label}

@app.post("/verify")
async def verify(request: Request):
    request.app.state.is_register = False
    request.app.state.current_label = None
    log_msg = "Switched to VERIFY mode."
    push_log(request.app, log_msg)
    return {"mode": "verify"}

@app.post("/toggle_background_removal")
async def toggle_background_removal(request: Request):
    current_state = not request.app.state.background_removal_enabled
    request.app.state.background_removal_enabled = current_state
    mode = "enabled" if current_state else "disabled"
    push_log(request.app, f"Background removal {mode}.")
    return {"background_removal": mode}

@app.post("/set_depth_threshold")
async def set_depth_threshold(request: Request, body: dict = Body(...)):
    threshold = body.get("threshold")
    if not isinstance(threshold, (float, int)) or not (0 <= threshold <= 1):
        raise HTTPException(
            status_code=400,
            detail="Invalid threshold value. Must be between 0 and 1."
        )
    request.app.state.depth_threshold = float(threshold)
    return {"status": "threshold updated", "threshold": threshold}

@app.get("/video/raw", include_in_schema=False)
async def raw_feed(request: Request):
    return StreamingResponse(
        stream(request, request.app.state.triton, to_roi=False),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/video/roi", include_in_schema=False)
async def roi_feed(request: Request):
    return StreamingResponse(
        stream(request, request.app.state.triton, to_roi=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

def switch_camera_sync(idx_str: str) -> str:
    idx = int(idx_str)

    if not hasattr(app, 'state'):
        msg = "❌ Application state not found. Cannot switch camera."
        logger.error(msg)
        return msg

    prev_idx = app.state.cam_idx
    status_msg = ""

    if idx == prev_idx:
        return f"ℹ️ Already using camera {idx}"

    cap_new = None
    logger.info(f"Attempting to open camera {idx}...")
    cap_new = cv2.VideoCapture(idx)
    if not cap_new or not cap_new.isOpened():
        status_msg = f"❌ Camera {idx} could not be opened. Check connection/permissions."
        logger.error(status_msg)
        if cap_new: cap_new.release()
        return status_msg
    else:
        configure_cap_full_hd(cap_new)
        old_cap = app.state.cap_map.pop(prev_idx, None)
        app.state.cap_map[idx] = cap_new
        app.state.cam_idx = idx
        status_msg = f"✅ Switched successfully to camera {idx}"
        logger.info(status_msg)

        if old_cap and old_cap.isOpened():
            old_cap.release()
            logger.info(f"Released previous camera {prev_idx}")
        return status_msg

BASE_URL = "http://localhost:7000"

def call_register_api(label):
    payload_label = label if isinstance(label, str) and label.strip() else None
    payload = {"label": payload_label} if payload_label else {}

    response = requests.post(f"{BASE_URL}/register", json=payload, timeout=5.0)
    response.raise_for_status()
    return response.json()

def call_verify_api():
    response = requests.post(f"{BASE_URL}/verify", timeout=5.0)
    response.raise_for_status()
    return response.json()

def call_toggle_bg_api():
    response = requests.post(f"{BASE_URL}/toggle_background_removal", timeout=5.0)
    response.raise_for_status()
    status = response.json().get("background_removal", "Error")
    return f"{status}"

def call_set_depth_threshold(threshold):
    response = requests.post(
        f"{BASE_URL}/set_depth_threshold",
        json={"threshold": threshold}
    )
    response.raise_for_status()
    return response.json()

def get_latest_log_for_ui():
    if not hasattr(app, 'state') or not hasattr(app.state, 'latest_log'):
        return '''
            <div style="background-color: #f0f0f0; color: #666; padding: 15px; font-size: 16px; border-radius: 8px; transition: all 0.3s ease;">
                Log state unavailable
            </div>
        '''
    
    log = app.state.latest_log
    style = "padding: 15px; font-size: 16px; border-radius: 8px; font-weight: 500; transition: all 0.3s ease;"
    
    # Check if it's a verification result
    if "VERIFY @" in log:
        if "Unregistered" in log or "No matches" in log:
            style += "background-color: #ffebee; color: #c62828;"  # Red for unmatched
        else:
            style += "background-color: #e8f5e9; color: #2e7d32;"  # Green for matched
    elif "REGISTER @" in log:
        style += "background-color: #e3f2fd; color: #1565c0;"  # Blue for registration
    else:
        style += "background-color: #f5f5f5; color: #424242;"  # Gray for other messages
    
    return f'''
        <div style="{style}">
            {log}
        </div>
    '''

with gr.Blocks(title="Palm‑Print Identification System", css="""
    .log-container {
        transition: all 0.3s ease;
    }
    .log-container div {
        transition: all 0.3s ease;
    }
""") as ui:
    gr.Markdown("# ✋ Palm‑Print Identification System")
    with gr.Row():
        with gr.Column(scale=1):
            cam_dd = gr.Dropdown(choices=[str(i) for i in range(2)], value='0', label="Select Camera Index")
            status = gr.Textbox(label="Camera Status", interactive=False, lines=1)
            logs = gr.HTML(
                label="Latest Event",
                value='<div style="background-color: #f0f0f0; color: #666; padding: 15px; font-size: 16px; border-radius: 8px; transition: all 0.3s ease;">No logs yet.</div>',
                show_label=True,
                elem_classes=["log-container"]
            )

            log_timer = gr.Timer(0.5)  # Increased refresh interval slightly
            log_timer.tick(
                fn=get_latest_log_for_ui,
                inputs=None,
                outputs=[logs]
            )

            cam_dd.change(
                fn=switch_camera_sync,
                inputs=[cam_dd],
                outputs=[status]
            )
        with gr.Column(scale=2):
            with gr.Row():
                label_input = gr.Textbox(label="Registration Label", placeholder="Enter label (e.g., user_palm_left)")
                reg_btn = gr.Button("Register Mode", variant="primary")
                ver_btn = gr.Button("Verify Mode", variant="secondary")
            with gr.Row():
                reg_resp = gr.JSON(label="Register API Response")
                ver_resp = gr.JSON(label="Verify API Response")
            with gr.Row():
                bg_toggle_btn = gr.Button("Toggle Background Removal")
                bg_status_text = gr.Textbox(label="Background Removal Status", value="disabled", interactive=False)
                depth_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.6,
                    label="Depth Threshold",
                    interactive=True
                )

            reg_btn.click(fn=call_register_api, inputs=[label_input], outputs=[reg_resp])
            ver_btn.click(fn=call_verify_api, inputs=None, outputs=[ver_resp])
            bg_toggle_btn.click(fn=call_toggle_bg_api, inputs=None, outputs=[bg_status_text])
            depth_slider.change(
                fn=call_set_depth_threshold,
                inputs=[depth_slider],
                outputs=[bg_status_text]
            )

    with gr.Row(equal_height=False):
        with gr.Column():
            gr.Markdown("### Raw/Masked Camera Feed")
            gr.HTML(f'<img src="{BASE_URL}/video/raw" alt="Raw Feed Loading..." style="width:100%; max-width:640px; height:auto; border:1px solid #ccc; background-color:#eee;">')
        with gr.Column():
            gr.Markdown("### Processed ROI Feed")
            gr.HTML(f'<img src="{BASE_URL}/video/roi" alt="ROI Feed Loading..." style="width:100%; max-width:224px; height:auto; border:1px solid #ccc; background-color:#eee;">')

gr.mount_gradio_app(app, ui, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=7000)