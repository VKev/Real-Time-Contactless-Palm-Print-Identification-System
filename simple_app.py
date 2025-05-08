import time
import queue
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse, PlainTextResponse
import gradio as gr

from roi_extraction.util import extract_palm_roi
from utils.preprocess import preprocess
from utils.triton import TritonClient
from utils.qdrant import QdrantHelper

from depth_estimation.utils import resize, interpolate_np, apply_depth_mask_np

import requests

# --- Functions from the original code ---

def push_log(app: FastAPI, msg: str):
    """Update the latest log message in app state"""
    app.state.latest_log = msg

def configure_cap_full_hd(cap: cv2.VideoCapture):
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def get_cap(app: FastAPI) -> cv2.VideoCapture:
    return app.state.cap_map[app.state.cam_idx]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cap0 = cv2.VideoCapture(0)
    configure_cap_full_hd(cap0)
    app.state.cap_map = {0: cap0}
    app.state.cam_idx = 0

    app.state.triton = TritonClient("localhost:8001", verbose=False)
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


def inference_callback_depth(user_data, result, error):
    if error:
        print("Async error:", error)
    else:
        print("Async feature extraction result shape:", result.as_numpy("OUTPUT__0").shape)

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
                # NOTE: Lower score is better match for Euclidean distance
                # Let's adjust the threshold logic if needed, assuming lower is better.
                # Example threshold: if best.score < 0.8: # Assuming normalized vectors or specific distance range
                if best.score > 80: # Keeping original logic - assuming higher score means *less* similar
                    log_msg = (
                        f"[#{idx}] VERIFY @ {t_out:.3f}s ({latency:.1f}ms) "
                        f"Unregistered (Score: {best.score:.2f} > 80)"
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


# --- MODIFIED STREAM FUNCTION ---
def stream(request: Request, triton: TritonClient, to_roi: bool = False):
    """Modified stream function to pass app instance and measure timings."""
    counter = 0
    frame_count = 0 # For printing timings less frequently if needed
    while True:
        t_frame_start = time.perf_counter() # Start timing the whole frame processing

        ok, frame = get_cap(request.app).read()
        if not ok:
            print("Warning: Failed to read frame from camera.")
            # Optional: Add a small delay before retrying
            time.sleep(0.1)
            # Provide a placeholder frame to avoid errors downstream
            frame = np.zeros((int(get_cap(request.app).get(cv2.CAP_PROP_FRAME_HEIGHT)),
                              int(get_cap(request.app).get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)
            # If still no frame after placeholder, maybe break or continue
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                print("Error: Placeholder frame is invalid. Skipping frame.")
                continue


        # Clear done queue
        while True:
            try: doneQ.get_nowait()
            except queue.Empty: break

        # --- Timing Resize ---
        t_start_resize = time.perf_counter()
        input_np, (h,w) = resize(frame, input_size=252)
        t_end_resize = time.perf_counter()
        resize_time_ms = (t_end_resize - t_start_resize) * 1000
        # --- End Timing Resize ---

        # --- Timing Triton Inference for Depth ---
        t_start_infer_depth = time.perf_counter()
        depth_out = triton.infer(
            model_name="depth_anything_v2",
            inputs={"INPUT__0": input_np},
            )["OUTPUT__0"]
        t_end_infer_depth = time.perf_counter()
        infer_depth_time_ms = (t_end_infer_depth - t_start_infer_depth) * 1000
        # --- End Timing Triton Inference ---

        # --- Timing Interpolation ---
        t_start_interp = time.perf_counter()
        depth_map = interpolate_np(depth_out, h, w)
        t_end_interp = time.perf_counter()
        interp_time_ms = (t_end_interp - t_start_interp) * 1000
        # --- End Timing Interpolation ---

        # Normalize depth map for masking
        if depth_map.ndim == 3 and depth_map.shape[0] == 1:
            depth_map = depth_map[0]
        # Add a small epsilon to prevent division by zero if depth range is zero
        ptp = depth_map.ptp()
        if ptp == 0: ptp = 1e-6 # Avoid division by zero
        depth_norm = ((depth_map - depth_map.min()) / ptp * 255.0).astype("uint8")

        # --- Timing Masking ---
        t_start_mask = time.perf_counter()
        # Ensure frame and depth_norm have compatible dimensions for masking
        if frame.shape[:2] != depth_norm.shape[:2]:
             # If shapes don't match (shouldn't happen with correct interpolate), resize depth_norm
             print(f"Warning: Resizing depth_norm ({depth_norm.shape}) to match frame ({frame.shape[:2]}) for masking.")
             depth_norm_resized = cv2.resize(depth_norm, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
             masked_frame = apply_depth_mask_np(frame, depth_norm_resized, threshold=0.6)
        else:
             masked_frame = apply_depth_mask_np(frame, depth_norm, threshold=0.6)
        t_end_mask = time.perf_counter()
        mask_time_ms = (t_end_mask - t_start_mask) * 1000
        # --- End Timing Masking ---

        # Use the masked frame for further processing
        output_frame = masked_frame # Assign the result of masking

        # --- Optional: Print timings (e.g., every 30 frames) ---
        frame_count += 1
        if frame_count % 30 == 0: # Print every 30 frames
             print(f"Frame {frame_count} Timings (ms): Resize={resize_time_ms:.2f}, InferDepth={infer_depth_time_ms:.2f}, Interp={interp_time_ms:.2f}, Mask={mask_time_ms:.2f}")


        if to_roi:
            # Use the 'masked_frame' which has depth applied
            roi = extract_palm_roi(masked_frame)
            if roi is not None:
                batch, img = preprocess(roi)
                t_in = time.time() # Using time.time() here as it matches the existing callback logic
                # Pass app instance in user_data
                triton.infer_async(
                    model_name="feature_extraction",
                    inputs={"INPUT__0": batch},
                    callback=inference_callback,
                    user_data=(counter, t_in, request.app),  # Added app instance
                )
                counter += 1
                # Display the extracted grayscale ROI in the ROI feed
                output_frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                # If no ROI found, display a placeholder in the ROI feed
                output_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        # If not 'to_roi', output_frame remains the depth-masked frame

        # Encode the frame to be sent in the stream
        ok_encode, buf = cv2.imencode(".jpg", output_frame)
        if not ok_encode:
            print("Error: Failed to encode frame to JPEG.")
            continue # Skip this frame

        # Yield the frame for the streaming response
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

        t_frame_end = time.perf_counter()
        frame_time_ms = (t_frame_end - t_frame_start) * 1000
        # Optional: print total frame time / FPS
        # if frame_count % 30 == 0:
        #    print(f"Frame {frame_count} Total Time: {frame_time_ms:.2f} ms, Approx FPS: {1000 / frame_time_ms if frame_time_ms > 0 else float('inf'):.1f}")

# --- Rest of the original code ---

@app.get("/logs/latest")
async def get_latest_log(request: Request):
    """Endpoint to retrieve latest log"""
    # Ensure latest_log exists and is a string
    log_message = getattr(request.app.state, 'latest_log', "No logs yet.")
    if not isinstance(log_message, str):
        log_message = str(log_message) # Convert if it's not a string
    return PlainTextResponse(log_message)

@app.get("/", include_in_schema=False)
def go_ui():
    return RedirectResponse(url="/ui")


@app.post("/register")
async def register(request: Request, body: dict = Body(...)):
    """Switch the system to *registration* mode and remember the label."""
    label = body.get("label", f"palm_{int(time.time())}")
    if not label: # Handle empty label input
         label = f"palm_{int(time.time())}"
    request.app.state.is_register = True
    request.app.state.current_label = label
    push_log(request.app, f"Mode switched to REGISTER. Label: '{label}'") # Log mode switch
    return {"mode": "register", "label": label}


@app.post("/verify")
async def verify(request: Request):
    """Switch the system to *verification* mode (no DB insertions)."""
    request.app.state.is_register = False
    request.app.state.current_label = None
    push_log(request.app, "Mode switched to VERIFY.") # Log mode switch
    return {"mode": "verify"}


@app.get("/video/raw", include_in_schema=False)
def raw_feed(request: Request):
    # This feed now shows the depth-masked video
    return StreamingResponse(
        stream(request, request.app.state.triton, to_roi=False),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/roi", include_in_schema=False)
def roi_feed(request: Request):
    # This feed shows the extracted ROI (or placeholder) after depth masking
    return StreamingResponse(
        stream(request, request.app.state.triton, to_roi=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

def switch_camera(idx: int, request: Request) -> str: # Pass request to access app state
    # Safely get current app state
    app_state = request.app.state
    try:
        idx = int(idx)
    except (ValueError, TypeError):
        return f"❌ Invalid camera index: {idx}"

    prev_idx = app_state.cam_idx

    if idx == prev_idx:
        return f"ℹ️ Already using camera {idx}"

    # Check if the camera index is already managed but maybe closed
    if idx in app_state.cap_map and not app_state.cap_map[idx].isOpened():
         print(f"Re-opening camera {idx}")
         app_state.cap_map[idx].release() # Ensure it's released before trying again
         del app_state.cap_map[idx] # Remove entry to force reinitialization

    if idx not in app_state.cap_map:
        print(f"Attempting to open camera {idx}...")
        cap_new = cv2.VideoCapture(idx)
        if not cap_new.isOpened():
            cap_new.release()
            print(f"❌ Camera {idx} could not be opened.")
            return f"❌ Camera {idx} could not be opened."
        print(f"Camera {idx} opened successfully. Configuring...")
        configure_cap_full_hd(cap_new)
        app_state.cap_map[idx] = cap_new
    else:
         # Camera already exists and is presumably open
         print(f"Camera {idx} already initialized.")
         cap_new = app_state.cap_map[idx] # Use existing capture object

    # Update the current camera index
    app_state.cam_idx = idx
    print(f"✅ Switched active camera index to {idx}")

    # Release the *previous* camera if it's different and exists
    if prev_idx != idx and prev_idx in app_state.cap_map:
        print(f"Releasing previous camera {prev_idx}...")
        app_state.cap_map[prev_idx].release()
        del app_state.cap_map[prev_idx]
        print(f"Previous camera {prev_idx} released.")

    return f"✅ Switched to camera {idx}"


# --- Gradio UI Setup ---
# Need to wrap API calls in functions that can be called by Gradio event handlers
def call_register_api(label, request: gr.Request):
    """Calls the /register endpoint."""
    # Use the request object to find the base URL if running behind a proxy,
    # otherwise default to localhost.
    base_url = "http://localhost:7000" # Default
    if request:
        # This might give the Gradio URL, adjust if needed
        # For simplicity, assuming direct access to localhost:7000
        pass # print(f"Gradio request host: {request.client.host}")

    url = f"{base_url}/register"
    try:
        payload = {"label": label if label else f"palm_{int(time.time())}"}
        r = requests.post(url, json=payload, timeout=5) # Add timeout
        r.raise_for_status() # Raise exception for bad status codes
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling /register: {e}")
        return {"error": f"Failed to connect or call API: {e}"}
    except Exception as e:
        print(f"Unexpected error during register call: {e}")
        return {"error": str(e)}

def call_verify_api(request: gr.Request):
    """Calls the /verify endpoint."""
    base_url = "http://localhost:7000" # Default
    if request:
         pass # print(f"Gradio request host: {request.client.host}")

    url = f"{base_url}/verify"
    try:
        r = requests.post(url, timeout=5) # Add timeout
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling /verify: {e}")
        return {"error": f"Failed to connect or call API: {e}"}
    except Exception as e:
        print(f"Unexpected error during verify call: {e}")
        return {"error": str(e)}

# Function to fetch latest log for Gradio
# Needs access to app state - requires passing request or using global (less ideal)
# A simple way for Gradio: call the /logs/latest endpoint
def get_latest_log_api(request: gr.Request):
     base_url = "http://localhost:7000" # Default
     if request:
         pass
     url = f"{base_url}/logs/latest"
     try:
         r = requests.get(url, timeout=1)
         r.raise_for_status()
         return r.text # Return plain text log
     except requests.exceptions.RequestException as e:
         # Don't spam console for log fetch errors unless debugging
         # print(f"Error fetching logs: {e}")
         return "Log service unavailable"
     except Exception as e:
         print(f"Unexpected error fetching logs: {e}")
         return "Error fetching logs"


with gr.Blocks(title="Palm‑Print Identification System") as ui:
    gr.Markdown("### Palm‑Print Identification System")
    with gr.Row():
        cam_dd = gr.Dropdown(choices=[0, 1], value=0, label="Camera Selection")
        status = gr.Textbox(label="System Status", interactive=False, scale=1) # Use scale for layout
        logs = gr.Textbox(label="Latest Event", interactive=False, scale=2) # Give logs more space

        log_timer = gr.Timer(value=0.2)
        log_timer.tick(
            fn=get_latest_log_api,
            inputs=None, # No direct input needed for the API call function
            outputs=[logs],
            api_name=False # Don't expose as Gradio API
        )

        # Use the request object available in Gradio event handlers
        cam_dd.change(switch_camera, inputs=[cam_dd], outputs=[status], api_name=False)


    with gr.Row():
        label_input = gr.Textbox(label="Registration Label", placeholder="Enter label…", scale=2)

        reg_btn = gr.Button("Register", scale=1)
        ver_btn = gr.Button("Verify", scale=1)


    with gr.Row():
         reg_resp = gr.JSON(label="Register Response", scale=1)
         ver_resp = gr.JSON(label="Verify Response", scale=1)


    # Connect buttons to the API calling functions
    reg_btn.click(fn=call_register_api, inputs=[label_input], outputs=[reg_resp], api_name=False)
    ver_btn.click(fn=call_verify_api, inputs=None,  outputs=[ver_resp], api_name=False)

    with gr.Row(equal_height=True):
        # Use absolute paths for image sources if running locally
        gr.HTML('<img src="/video/raw" style="max-width:100%; height:auto; border:1px solid #ccc; object-fit: contain;">')
        gr.HTML('<img src="/video/roi" style="max-width:100%; height:auto; border:1px solid #ccc; object-fit: contain;">')


# Mount Gradio app correctly
app = gr.mount_gradio_app(app, ui, path="/ui")

if __name__ == "__main__":
    # It's generally recommended to run Uvicorn programmatically for more control
    # or use the command line: uvicorn your_script_name:app --host localhost --port 7000 --reload
    uvicorn.run(app, host="localhost", port=7000)