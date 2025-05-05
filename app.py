import time
import asyncio
from contextlib import asynccontextmanager
import logging

import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse, PlainTextResponse
import gradio as gr
import requests

try:
    from roi_extraction.util import extract_palm_roi
    from utils.preprocess import preprocess
    from utils.triton import TritonClient
    from utils.qdrant import QdrantHelper
    from depth_estimation.utils import resize, interpolate_np, apply_depth_mask_np
except ImportError as e:
    logging.error(f"ImportError: {e}. Make sure paths are correct.")
    def extract_palm_roi(frame): return frame if frame is not None else None
    def preprocess(roi): return np.zeros((1, 3, 224, 224), dtype=np.float32), np.zeros((224, 224), dtype=np.uint8)
    def resize(frame, input_size): return np.zeros((1, 3, input_size, input_size), dtype=np.float32), (frame.shape[0], frame.shape[1]) if frame is not None else (0, 0)
    def interpolate_np(depth_out, h, w): return np.zeros((1, h, w), dtype=np.float32)
    def apply_depth_mask_np(frame, depth_norm, threshold): return frame if frame is not None else np.zeros((224, 224, 3), dtype=np.uint8)

    from utils.triton import TritonClient
    from utils.qdrant import QdrantHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

shutdown_event = asyncio.Event()
inference_queue: asyncio.Queue | None = None
depth_queue: asyncio.Queue | None = None # Queue for depth results

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
        try:
            idx, vec, t_in, is_register, label = await inference_queue.get()
            t_out = time.time()
            latency = (t_out - t_in) * 1000

            try:
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
                            f"No matches found"
                        )

                push_log(app, log_msg)

            except Exception as e:
                error_msg = f"[#{idx}] ERROR processing result @ {t_out:.3f}s: {e}"
                push_log(app, error_msg)
            finally:
                inference_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Inference result processing task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in inference result processing loop: {e}")
            await asyncio.sleep(1)

    logger.info("Inference result processing worker stopped.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_queue, depth_queue
    logger.info("Application startup...")

    app.state.cap_map = {}
    app.state.cam_idx = 0
    try:
        cap0 = cv2.VideoCapture(0)
        if cap0.isOpened():
            configure_cap_full_hd(cap0)
            app.state.cap_map[0] = cap0
            logger.info("Camera 0 opened successfully.")
        else:
            logger.error("Failed to open camera 0")
            cap0.release()
    except Exception as e:
        logger.error(f"Error initializing camera 0: {e}")

    try:
        app.state.triton = TritonClient("localhost:8001", verbose=False)
        logger.info("Triton client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Triton client: {e}")
        app.state.triton = None

    try:
        qd = QdrantHelper(host="localhost", port=6333, grpc_port=6334, prefer_grpc=True)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, qd.ensure_collection, "palm_vectors", 128)
        app.state.qdrant = qd
        logger.info("Qdrant client initialized and collection ensured.")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        app.state.qdrant = None

    app.state.latest_log = "System initialized"
    app.state.is_register = False
    app.state.current_label = None
    app.state.background_removal_enabled = False # Default off

    inference_queue = asyncio.Queue()
    depth_queue = asyncio.Queue(maxsize=1) # Keep only the latest depth result

    worker_task = asyncio.create_task(process_inference_results(app))
    logger.info("Background worker task created.")

    yield

    logger.info("Application shutdown...")
    shutdown_event.set()

    if worker_task:
        try:
            await asyncio.wait_for(worker_task, timeout=5.0)
            logger.info("Background worker task finished.")
        except asyncio.TimeoutError:
            logger.warning("Background worker task did not finish in time. Cancelling...")
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                logger.info("Background worker task cancelled successfully.")
        except Exception as e:
             logger.error(f"Error during worker task shutdown: {e}")

    for cap in app.state.cap_map.values():
        if cap and cap.isOpened():
            cap.release()
    logger.info("Camera captures released.")

    if hasattr(app.state, 'triton') and app.state.triton and hasattr(app.state.triton, 'close'):
        try:
            # Assuming TritonClient has no close method or it's handled internally
            # app.state.triton.close()
            logger.info("Triton client closed (or managed internally).")
        except Exception as e:
            logger.error(f"Error closing Triton client: {e}")

    if hasattr(app.state, 'qdrant') and app.state.qdrant and hasattr(app.state.qdrant, 'close'):
        try:
            # Assuming QdrantHelper might have a close method
            # app.state.qdrant.close()
            logger.info("Qdrant client closed (or managed internally).")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")

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
        try:
            if result is None:
                 logger.error(f"[Triton Callback Error #{idx}] Received None result.")
                 return

            vec_output = result.as_numpy("OUTPUT__0")
            if vec_output is None:
                 logger.error(f"[Triton Callback Error #{idx}] 'OUTPUT__0' not found in result.")
                 return

            vec = vec_output.squeeze().astype(float).tolist()
            try:
                inference_queue.put_nowait((idx, vec, t_in, is_register, current_label))
            except asyncio.QueueFull:
                logger.warning(f"[#{idx}] Inference queue full. Discarding result.")
            except Exception as q_err:
                 logger.error(f"[#{idx}] Error putting result into queue: {q_err}")

        except Exception as e:
            logger.error(f"[Triton Callback Exception #{idx}] Error processing result: {e}")

# Separate callback for depth inference
def depth_inference_callback(user_data, result, error):
    global depth_queue
    if not depth_queue:
        logger.warning("Depth queue not available in callback.")
        return

    frame_idx = user_data # Could pass more context if needed

    if error:
        logger.error(f"[Depth Callback Error #{frame_idx}] {error}")
        # Optionally put an error indicator in the queue
        # depth_queue.put_nowait((frame_idx, None, error))
    else:
        try:
            if result is None:
                logger.error(f"[Depth Callback Error #{frame_idx}] Received None result.")
                # Optionally put an error indicator
                # depth_queue.put_nowait((frame_idx, None, "None result received"))
                return

            depth_output = result.as_numpy("OUTPUT__0") # Adjust output tensor name if needed
            if depth_output is None:
                logger.error(f"[Depth Callback Error #{frame_idx}] Depth output tensor not found in result.")
                # Optionally put an error indicator
                # depth_queue.put_nowait((frame_idx, None, "Output tensor missing"))
                return

            # Put the result in the queue (non-blocking, overwrite if full)
            try:
                # Clear the queue first to ensure only latest result is stored
                while not depth_queue.empty():
                    depth_queue.get_nowait()
                depth_queue.put_nowait((frame_idx, depth_output, None)) # (idx, data, error)
            except asyncio.QueueFull:
                 # This should not happen with Queue(maxsize=1) after clearing, but handle just in case
                 logger.warning(f"[#{frame_idx}] Depth queue was unexpectedly full after clearing. Discarding depth result.")
            except Exception as q_err:
                 logger.error(f"[#{frame_idx}] Error putting depth result into queue: {q_err}")

        except Exception as e:
             logger.error(f"[Depth Callback Exception #{frame_idx}] Error processing depth result: {e}")
             # Optionally put an error indicator
             # depth_queue.put_nowait((frame_idx, None, str(e)))


def run_triton_infer(triton, model_name, inputs):
    # Helper to run synchronous infer in executor
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
    latest_masked_frame = None # Store the latest frame after background removal

    while not shutdown_event.is_set():
        cap = get_cap(request.app)
        if not cap or not cap.isOpened():
            logger.warning("Camera not ready or closed. Yielding placeholder.")
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera Feed", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ok = True
            await asyncio.sleep(0.5)
        else:
            try:
                ok, frame = await loop.run_in_executor(None, cap.read)
            except Exception as e:
                ok = False
                logger.error(f"OpenCV error during cap.read(): {e}", exc_info=False)

        if not ok or frame is None:
            logger.warning("Failed to read frame from camera. Using placeholder.")
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "Frame Read Error", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ok = True
            await asyncio.sleep(0.1)

        current_frame_to_process = frame.copy() # Work on a copy
        processed_frame_for_display = current_frame_to_process # Default display is raw/error frame

        if request.app.state.background_removal_enabled:
            try:
                # 1. Async Resize
                resized_input, (h, w) = await loop.run_in_executor(None, resize, current_frame_to_process, 252)

                # 2. Async Depth Inference (using sync infer in executor)
                # Check if depth model name needs adjustment
                depth_result = await loop.run_in_executor(
                    None,
                    run_triton_infer,
                    triton,
                    "depth_anything_v2", # Verify model name
                    {"INPUT__0": resized_input}
                )
                depth_out = depth_result.get("OUTPUT__0") # Verify output tensor name

                if depth_out is not None:
                    # 3. Async Interpolate
                    depth_map = await loop.run_in_executor(None, interpolate_np, depth_out, h, w)

                    # Normalize depth map (CPU bound, quick)
                    if depth_map.ndim == 3 and depth_map.shape[0] == 1:
                        depth_map = depth_map[0]
                    ptp = depth_map.ptp()
                    if ptp == 0: ptp = 1e-6
                    depth_norm = ((depth_map - depth_map.min()) / ptp * 255.0).astype("uint8")

                    # 4. Async Masking
                    if current_frame_to_process.shape[:2] != depth_norm.shape[:2]:
                        depth_norm_resized = await loop.run_in_executor(
                             None, cv2.resize, depth_norm, (current_frame_to_process.shape[1], current_frame_to_process.shape[0]), interpolation=cv2.INTER_NEAREST
                         )
                    else:
                         depth_norm_resized = depth_norm

                    latest_masked_frame = await loop.run_in_executor(
                         None, apply_depth_mask_np, current_frame_to_process, depth_norm_resized, 0.6
                    )
                    # Use the masked frame for subsequent processing
                    current_frame_to_process = latest_masked_frame
                    processed_frame_for_display = latest_masked_frame # Show masked frame in raw feed

                else:
                     logger.warning("Depth inference failed. Skipping background removal for this frame.")
                     # Keep original frame for ROI/display if depth fails
                     processed_frame_for_display = current_frame_to_process


            except Exception as bg_err:
                 logger.error(f"Error during background removal: {bg_err}", exc_info=True)
                 # Fallback to original frame if background removal fails
                 processed_frame_for_display = frame
                 current_frame_to_process = frame # Ensure ROI uses original if BG removal fails
        else:
             # If BG removal is off, the "processed" frame is just the raw frame
             processed_frame_for_display = current_frame_to_process


        img_to_encode = processed_frame_for_display # Default to showing the raw/masked frame

        if to_roi:
            try:
                # Use the potentially masked frame 'current_frame_to_process' for ROI extraction
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

                        if processed_img_for_display.ndim == 2:
                             img_to_encode = cv2.cvtColor(processed_img_for_display, cv2.COLOR_GRAY2BGR)
                        elif processed_img_for_display.ndim == 3:
                             img_to_encode = processed_img_for_display
                        else:
                             logger.warning("Processed image has unexpected dimensions. Using placeholder.")
                             img_to_encode = np.zeros((224, 224, 3), dtype=np.uint8)
                             cv2.putText(img_to_encode, "Proc Err", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    else:
                         logger.warning("Preprocessing failed. Using placeholder ROI.")
                         img_to_encode = np.zeros((224, 224, 3), dtype=np.uint8)
                         cv2.putText(img_to_encode, "Prep Fail", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                else:
                    img_to_encode = np.zeros((224, 224, 3), dtype=np.uint8)
                    cv2.putText(img_to_encode, "No ROI", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

            except Exception as e:
                logger.error(f"Error during ROI processing: {e}", exc_info=True)
                img_to_encode = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.putText(img_to_encode, "ROI Error", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        try:
            if img_to_encode is None or img_to_encode.size == 0:
                 logger.warning("img_to_encode is invalid before encoding. Skipping frame.")
                 await asyncio.sleep(0.01)
                 continue

            ok_encode, buf = await loop.run_in_executor(None, cv2.imencode, ".jpg", img_to_encode)

            if ok_encode and buf is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            else:
                logger.warning("Failed to encode frame to JPEG.")
        except cv2.error as cv_err:
             logger.error(f"OpenCV error during encoding: {cv_err}. Image shape: {img_to_encode.shape}, dtype: {img_to_encode.dtype}")
        except Exception as e:
            logger.error(f"Error encoding/yielding frame: {e}", exc_info=True)

        await asyncio.sleep(0.005)

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
    try:
        idx = int(idx_str)
    except (ValueError, TypeError):
        msg = f"❌ Invalid camera index '{idx_str}'. Please select a number."
        logger.error(msg)
        return msg

    if not hasattr(app, 'state'):
         msg = "❌ Application state not found. Cannot switch camera."
         logger.error(msg)
         return msg

    prev_idx = app.state.cam_idx
    status_msg = ""

    if idx == prev_idx:
        return f"ℹ️ Already using camera {idx}"

    cap_new = None
    try:
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
    except Exception as e:
        logger.error(f"Error switching to camera {idx}: {e}", exc_info=True)
        if cap_new: cap_new.release()
        return f"❌ Error switching to camera {idx}: {e}"

BASE_URL = "http://localhost:7000"

def call_register_api(label):
    try:
        payload_label = label if isinstance(label, str) and label.strip() else None
        payload = {"label": payload_label} if payload_label else {}

        response = requests.post(f"{BASE_URL}/register", json=payload, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error("API call to /register timed out.")
        return {"error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        logger.error(f"API call to /register failed: {e}")
        err_resp = {"error": str(e)}
        if e.response is not None:
             try:
                 err_resp["detail"] = e.response.json()
             except requests.exceptions.JSONDecodeError:
                 err_resp["detail"] = e.response.text
        return err_resp
    except Exception as e:
        logger.error(f"Error processing /register response: {e}")
        return {"error": f"Failed to process response: {e}"}

def call_verify_api():
    try:
        response = requests.post(f"{BASE_URL}/verify", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error("API call to /verify timed out.")
        return {"error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        logger.error(f"API call to /verify failed: {e}")
        err_resp = {"error": str(e)}
        if e.response is not None:
             try:
                 err_resp["detail"] = e.response.json()
             except requests.exceptions.JSONDecodeError:
                 err_resp["detail"] = e.response.text
        return err_resp
    except Exception as e:
        logger.error(f"Error processing /verify response: {e}")
        return {"error": f"Failed to process response: {e}"}

def call_toggle_bg_api():
    try:
        response = requests.post(f"{BASE_URL}/toggle_background_removal", timeout=5.0)
        response.raise_for_status()
        status = response.json().get("background_removal", "Error")
        return f"{status}"
    except requests.exceptions.Timeout:
        logger.error("API call to /toggle_background_removal timed out.")
        return "Timeout"
    except requests.exceptions.RequestException as e:
        logger.error(f"API call to /toggle_background_removal failed: {e}")
        return "API Error"
    except Exception as e:
        logger.error(f"Error processing /toggle_background_removal response: {e}")
        return "Processing Error"

def get_latest_log_for_ui():
    if hasattr(app, 'state') and hasattr(app.state, 'latest_log'):
        return app.state.latest_log
    return "Log state unavailable"

with gr.Blocks(title="Palm‑Print Identification System") as ui:
    gr.Markdown("# ✋ Palm‑Print Identification System")
    with gr.Row():
        with gr.Column(scale=1):
            cam_dd = gr.Dropdown(choices=[str(i) for i in range(2)], value='0', label="Select Camera Index")
            status = gr.Textbox(label="Camera Status", interactive=False, lines=1)
            logs = gr.Textbox(label="Latest Event", interactive=False, lines=1, max_lines=1)

            log_timer = gr.Timer(value=0.2) # Use interval instead of value
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

            reg_btn.click(fn=call_register_api, inputs=[label_input], outputs=[reg_resp])
            ver_btn.click(fn=call_verify_api, inputs=None, outputs=[ver_resp])
            bg_toggle_btn.click(fn=call_toggle_bg_api, inputs=None, outputs=[bg_status_text])


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