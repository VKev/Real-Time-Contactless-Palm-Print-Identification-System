# app.py (Optimized Version)
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
# Assuming roi_extraction and utils are structured correctly
try:
    from roi_extraction.util import extract_palm_roi
    from roi_extraction.gesture import OpenPalmDetector
    from utils.preprocess import preprocess
    from utils.triton import TritonClient
    from utils.qdrant import QdrantHelper
except ImportError as e:
    logging.error(f"ImportError: {e}. Make sure paths are correct.")
    # Fallback or dummy implementations if needed for testing structure
    def extract_palm_roi(frame): return frame if frame is not None else None
    class OpenPalmDetector: pass
    def preprocess(roi): return np.zeros((1, 3, 224, 224), dtype=np.float32), roi
    from utils.triton import TritonClient # Assume this imports
    from utils.qdrant import QdrantHelper # Assume this imports

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global State (managed by lifespan) ---
# Using asyncio primitives for thread-safety and async compatibility
shutdown_event = asyncio.Event()
inference_queue: asyncio.Queue | None = None

# --- Helper Functions ---

def push_log(app: FastAPI, msg: str):
    """Update the latest log message in app state (thread-safe update if needed)"""
    # In a real-world scenario with multiple workers potentially logging,
    # consider using a thread-safe mechanism if app.state itself isn't inherently safe
    # For this structure, direct assignment within the Qdrant worker task is likely okay.
    app.state.latest_log = msg
    logger.info(msg) # Also log to console/file

def configure_cap_full_hd(cap: cv2.VideoCapture):
    # Configure desired resolution - check return values if specific resolutions are critical
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logger.info(f"Attempted to set resolution to 1280x720")
    # Optional: Log actual resolution
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # logger.info(f"Actual resolution: {width}x{height}")


def get_cap(app: FastAPI) -> cv2.VideoCapture | None:
    """Safely get the current camera capture object."""
    return app.state.cap_map.get(app.state.cam_idx)


async def process_inference_results(app: FastAPI):
    """Worker task to process results from the inference queue."""
    global inference_queue
    if not inference_queue:
        logger.error("Inference queue not initialized!")
        return

    loop = asyncio.get_running_loop()
    qdrant: QdrantHelper = app.state.qdrant

    logger.info("Starting inference result processing worker.")
    while not shutdown_event.is_set():
        try:
            # Wait for an item from the queue
            idx, vec, t_in, is_register, label = await inference_queue.get()
            t_out = time.time()
            latency = (t_out - t_in) * 1000

            try:
                if is_register:
                    current_label = label or f"palm_{int(time.time())}" # Ensure label exists
                    log_msg = (
                        f"[#{idx}] REGISTER @ ({latency:.1f}ms) "
                        f"label: {current_label}"
                    )
                    # Run blocking Qdrant insert in thread pool
                    await loop.run_in_executor(
                        None, # Default executor (thread pool)
                        qdrant.insert_vectors,
                        "palm_vectors", [vec], [idx], [{"label": current_label}]
                    )
                else:
                    # Run blocking Qdrant search in thread pool
                    matches = await loop.run_in_executor(
                        None, qdrant.search, "palm_vectors", vec, 1 # Search only top 1 for verification
                    )

                    if matches:
                        best = matches[0]
                        # Adjusted threshold example - tune based on observed scores
                        # Higher score means MORE SIMILAR in EUCLID distance in this qdrant setup
                        # Let's assume lower score is better match
                        # Check qdrant distance metric used (EUCLID usually means lower = better)
                        # Assuming Distance.EUCLID was used (lower is better)
                        MATCH_THRESHOLD =80 # Example threshold - TUNE THIS VALUE!
                        if best.score < MATCH_THRESHOLD:
                             log_msg = (
                                f"[#{idx}] VERIFY @ ({latency:.1f}ms) "
                                f"match: {best.payload.get('label')} "
                                f"({best.score:.4f})" # More precision for scores
                            )
                        else:
                            log_msg = (
                                f"[#{idx}] VERIFY @ ({latency:.1f}ms) "
                                f"Unregistered (Score: {best.score:.4f} > {MATCH_THRESHOLD})"
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
                inference_queue.task_done() # Notify queue that task is complete

        except asyncio.CancelledError:
            logger.info("Inference result processing task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in inference result processing loop: {e}")
            await asyncio.sleep(1) # Avoid tight loop on unexpected errors

    logger.info("Inference result processing worker stopped.")


# --- Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_queue
    logger.info("Application startup...")

    # Initialize resources
    app.state.cap_map = {}
    app.state.cam_idx = 0
    try:
        cap0 = cv2.VideoCapture(0)
        if cap0.isOpened():
            configure_cap_full_hd(cap0)
            app.state.cap_map[0] = cap0
        else:
            logger.error("Failed to open camera 0")
            cap0.release() # Release if opened but failed config or check
    except Exception as e:
        logger.error(f"Error initializing camera 0: {e}")

    # Initialize Triton client
    try:
        # Ensure TritonClient doesn't block excessively on init
        app.state.triton = TritonClient("localhost:8001", verbose=False)
        # Optional: Add a readiness check/ping to Triton here
        logger.info("Triton client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Triton client: {e}")
        app.state.triton = None # Mark as unavailable

    # Initialize Gesture Detector (assuming it's lightweight)
    try:
        app.state.gesture = OpenPalmDetector(
            "roi_extraction/gesture_recognizer/gesture_recognizer.task"
        )
        logger.info("Gesture detector initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize gesture detector: {e}")
        app.state.gesture = None

    # Initialize Qdrant client
    try:
        # Qdrant client init might involve network I/O - consider async if available
        # or ensure it's reasonably fast.
        qd = QdrantHelper(host="localhost", port=6333, grpc_port=6334, prefer_grpc=True)
        # Run ensure_collection in executor as it might block
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

    # Initialize asyncio Queue
    inference_queue = asyncio.Queue()

    # Start the background worker task
    worker_task = asyncio.create_task(process_inference_results(app))
    logger.info("Background worker task created.")

    yield # Application runs here

    # --- Cleanup ---
    logger.info("Application shutdown...")
    shutdown_event.set() # Signal tasks to stop

    # Wait for the worker task to finish
    if worker_task:
        try:
            await asyncio.wait_for(worker_task, timeout=5.0)
            logger.info("Background worker task finished.")
        except asyncio.TimeoutError:
            logger.warning("Background worker task did not finish in time.")
            worker_task.cancel()
        except Exception as e:
             logger.error(f"Error during worker task shutdown: {e}")


    # Release camera captures
    for cap in app.state.cap_map.values():
        if cap and cap.isOpened():
            cap.release()
    logger.info("Camera captures released.")

    # Cleanup other resources if necessary (e.g., close client connections if they have close methods)
    if hasattr(app.state.triton, 'close'):
        app.state.triton.close()
        logger.info("Triton client closed.")
    if hasattr(app.state.qdrant, 'close'):
        # Assuming qdrant client might have a close method
        app.state.qdrant.close()
        logger.info("Qdrant client closed.")

    logger.info("Application shutdown complete.")


# --- FastAPI App ---
app = FastAPI(lifespan=lifespan)

# --- Triton Callback ---

def inference_callback(user_data, result, error):
    """Callback from Triton: Puts data onto the asyncio queue."""
    global inference_queue
    if not inference_queue: return # Queue might not be ready on startup/shutdown

    idx, t_in, is_register, current_label = user_data

    if error:
        # Log error immediately, don't put on queue? Or put error marker?
        # For now, log and skip queue. Can adjust if error processing needed downstream.
        logger.error(f"[Triton Callback Error #{idx}] {error}")
        # Optionally push a log update immediately if critical
        # loop = asyncio.get_running_loop()
        # asyncio.run_coroutine_threadsafe(push_log_async(app, f"[#{idx}] INFERENCE ERROR"), loop)
    else:
        try:
            # Ensure accessing result is safe (check result object type if needed)
            vec = result.as_numpy("OUTPUT__0").squeeze().astype(float).tolist()
            # Put data onto the queue for the async worker task
            try:
                 inference_queue.put_nowait((idx, vec, t_in, is_register, current_label))
            except asyncio.QueueFull:
                 logger.warning(f"[#{idx}] Inference queue full. Discarding result.")

        except Exception as e:
            logger.error(f"[Triton Callback Exception #{idx}] Error processing result: {e}")


# --- Video Streaming ---

async def stream(request: Request, triton: TritonClient | None, to_roi: bool = False):
    """Async generator for video streaming, offloading blocking calls."""
    if not triton:
        logger.error("Triton client not available for streaming.")
        # Optionally yield a placeholder/error frame
        return

    counter = 0
    loop = asyncio.get_running_loop()

    while not shutdown_event.is_set():
        cap = get_cap(request.app)
        if not cap or not cap.isOpened():
            logger.warning("Camera not ready or closed.")
            # Yield a placeholder frame or break/sleep
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            ok = False
            await asyncio.sleep(0.5) # Wait before retrying
            # continue # Or break if camera is essential
        else:
           # Run blocking cv2.read() in thread pool
            ok, frame = await loop.run_in_executor(None, cap.read)

        if not ok:
            logger.warning("Failed to read frame from camera.")
            # Reuse last known frame or placeholder? For now, placeholder.
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            await asyncio.sleep(0.1) # Avoid busy-looping on read errors
            # continue # Or break loop

        img_to_encode = frame # Default to sending the raw frame

        if to_roi:
            try:
                # Run potentially blocking ROI extraction in thread pool
                # Note: extract_palm_roi might need access to app.state.gesture if it uses it
                # Pass necessary state or make gesture detector thread-safe if required.
                # For simplicity, assuming it's self-contained for now.
                roi = await loop.run_in_executor(None, extract_palm_roi, frame)

                if roi is not None:
                    # Run potentially blocking preprocessing in thread pool
                    # Preprocess might return batch and processed image for display
                    batch, processed_img_for_display = await loop.run_in_executor(None, preprocess, roi)

                    t_in = time.time()
                    # Get current registration state AT THE TIME OF SENDING
                    # This is slightly racy if state changes exactly between read and infer,
                    # but generally acceptable for this kind of UI feedback.
                    is_reg = request.app.state.is_register
                    curr_label = request.app.state.current_label

                    # Schedule async inference (non-blocking)
                    triton.infer_async(
                        model_name="feature_extraction",
                        inputs={"INPUT__0": batch},
                        callback=inference_callback,
                        # Pass necessary context for the callback
                        user_data=(counter, t_in, is_reg, curr_label),
                    )
                    counter += 1
                    # Update frame to show the processed ROI
                    if processed_img_for_display.ndim == 2: # Grayscale
                        img_to_encode = cv2.cvtColor(processed_img_for_display, cv2.COLOR_GRAY2BGR)
                    else: # Assume already BGR-like
                         img_to_encode = processed_img_for_display

                else:
                    # No ROI found, send placeholder for ROI feed
                    img_to_encode = np.zeros((224, 224, 3), dtype=np.uint8)

            except Exception as e:
                logger.error(f"Error during ROI processing: {e}")
                img_to_encode = np.zeros((224, 224, 3), dtype=np.uint8) # Fallback frame


        # Encode the frame (can also be moved to executor if it's slow)
        try:
            ok, buf = await loop.run_in_executor(None, cv2.imencode, ".jpg", img_to_encode)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            else:
                logger.warning("Failed to encode frame to JPEG.")
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")

        # Small sleep to prevent extremely tight loop, allowing other tasks to run
        await asyncio.sleep(0.005) # Adjust as needed


# --- API Endpoints ---

@app.get("/logs/latest")
async def get_latest_log(request: Request):
    """Endpoint to retrieve latest log"""
    return PlainTextResponse(request.app.state.latest_log)

@app.get("/", include_in_schema=False)
async def go_ui(): # Make async
    return RedirectResponse(url="/ui")


@app.post("/register")
async def register(request: Request, body: dict = Body(...)):
    """Switch the system to *registration* mode and remember the label."""
    label = body.get("label") # Get label, handle if None
    if not label:
        label = f"palm_{int(time.time())}"
        logger.warning(f"No label provided for registration, using default: {label}")
    # Update state (assuming direct assignment is okay here, check thread safety if needed)
    request.app.state.is_register = True
    request.app.state.current_label = label
    log_msg = f"Switched to REGISTER mode. Label: {label}"
    push_log(request.app, log_msg)
    return {"mode": "register", "label": label}


@app.post("/verify")
async def verify(request: Request):
    """Switch the system to *verification* mode (no DB insertions)."""
    # Update state
    request.app.state.is_register = False
    request.app.state.current_label = None
    log_msg = "Switched to VERIFY mode."
    push_log(request.app, log_msg)
    return {"mode": "verify"}


@app.get("/video/raw", include_in_schema=False)
async def raw_feed(request: Request):
    """Raw video feed stream."""
    return StreamingResponse(
        stream(request, request.app.state.triton, to_roi=False),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/roi", include_in_schema=False)
async def roi_feed(request: Request):
    """ROI video feed stream with inference triggering."""
    return StreamingResponse(
        stream(request, request.app.state.triton, to_roi=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

# --- Gradio UI ---

# Define Gradio UI interaction functions (run synchronously for Gradio)
# These interact with the running FastAPI app via HTTP requests or direct state manipulation (carefully)

def switch_camera_sync(idx: int) -> str:
    """Synchronous wrapper for Gradio to switch camera."""
    # Note: This directly manipulates state. Consider thread safety if multiple
    # UI interactions could happen concurrently. For Gradio, usually single-threaded.
    idx = int(idx)
    prev_idx = app.state.cam_idx
    status_msg = ""

    if idx == prev_idx:
        return f"ℹ️ Already using camera {idx}"

    # Blockingly open the new camera - this might hang the UI thread briefly
    cap_new = cv2.VideoCapture(idx)
    if not cap_new.isOpened():
        cap_new.release()
        status_msg = f"❌ Camera {idx} could not be opened."
        logger.error(status_msg)
        return status_msg
    else:
        configure_cap_full_hd(cap_new) # Configure the new camera
        # Safely update the shared state
        old_cap = app.state.cap_map.pop(prev_idx, None) # Remove old first
        app.state.cap_map[idx] = cap_new
        app.state.cam_idx = idx
        status_msg = f"✅ Switched to camera {idx}"
        logger.info(status_msg)
        # Release the old camera if it existed
        if old_cap and old_cap.isOpened():
            old_cap.release()
            logger.info(f"Released previous camera {prev_idx}")
        return status_msg


# Use requests for interacting with the app's API from Gradio callbacks
# This avoids potential threading issues with direct state manipulation from Gradio
# Ensure the host/port match where uvicorn runs
BASE_URL = "http://localhost:7000" # Make this configurable if needed

def call_register_api(label):
    try:
        # Add a timeout to requests
        response = requests.post(f"{BASE_URL}/register", json={"label": label}, timeout=5.0)
        response.raise_for_status() # Raise exception for bad status codes (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API call to /register failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error processing /register response: {e}")
        return {"error": "Failed to process response"}


def call_verify_api():
    try:
        response = requests.post(f"{BASE_URL}/verify", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API call to /verify failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error processing /verify response: {e}")
        return {"error": "Failed to process response"}

def get_latest_log_for_ui():
     # Direct state access okay here as it's just reading for UI update
     # If latest_log updates were complex, might need a lock or async fetch
     return app.state.latest_log

# Define Gradio UI Layout
with gr.Blocks(title="Palm‑Print Identification System") as ui:
    gr.Markdown("### Palm‑Print Identification System")
    with gr.Row():
        cam_dd = gr.Dropdown(choices=[0, 1, 2, 3], value=0, label="Camera Selection") # Added more options potentially
        status = gr.Textbox(label="System Status", interactive=False)
        logs = gr.Textbox(label="Latest Event", interactive=False, lines=1) # Single line might be better

        # Gradio Timer to poll for the latest log message from app state
        # This runs in Gradio's loop, not FastAPI's
        log_timer = gr.Timer(value=0.2) # Poll every 500ms
        log_timer.tick(
            fn=get_latest_log_for_ui, # Function to get log from app state
            inputs=None,
            outputs=[logs]
        )

        # Connect camera dropdown change to the sync function
        cam_dd.change(switch_camera_sync, cam_dd, status)

    with gr.Row():
        label_input = gr.Textbox(label="Registration Label", placeholder="Enter label (e.g., user_palm)")

        reg_btn = gr.Button("Register Mode")
        reg_resp = gr.JSON(label="Register API Response")

        ver_btn = gr.Button("Verify Mode")
        ver_resp = gr.JSON(label="Verify API Response")

    # Connect buttons to API call functions
    reg_btn.click(fn=call_register_api, inputs=[label_input], outputs=[reg_resp])
    ver_btn.click(fn=call_verify_api, inputs=None, outputs=[ver_resp])

    with gr.Row(equal_height=True):
        # Use Gradio Image components that update based on the stream endpoint
        # Note: Standard HTML img src="/video/..." might be simpler if Gradio integration is complex
        # For simplicity, using HTML img tag pointing to the FastAPI stream endpoints
        gr.HTML(f'<img src="{BASE_URL}/video/raw" alt="Raw Feed" style="max-width:640px; height:auto; border:1px solid #ccc;">')
        gr.HTML(f'<img src="{BASE_URL}/video/roi" alt="ROI Feed" style="max-width:224px; height:auto; border:1px solid #ccc;">')


# Mount Gradio app
gr.mount_gradio_app(app, ui, path="/ui")


if __name__ == "__main__":
    # Consider configuration for host/port
    uvicorn.run(app, host="localhost", port=7000) # Use 0.0.0.0 to allow access from network