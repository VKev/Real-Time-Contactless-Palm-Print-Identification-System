import time
import cv2
import uvicorn
import queue
import numpy as np

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

import gradio as gr
from importlib.resources import files

from roi_extraction.util import extract_palm_roi
from utils.preprocess import preprocess
from utils.triton import TritonClient
from utils.qdrant import QdrantHelper
from roi_extraction.gesture import OpenPalmDetector

from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_444

jpeg = TurboJPEG()

def encode_jpeg_np(arr: np.ndarray) -> bytes:
    is_gray = arr.ndim == 2 or arr.shape[2] == 1
    return jpeg.encode(
        arr,
        quality=95,
        jpeg_subsample=TJSAMP_444,
        pixel_format=TJPF_GRAY if is_gray else None
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    cap = cv2.VideoCapture(0)
    app.state.cap = cap
    app.state.triton = TritonClient("localhost:8001", verbose=False)
    app.state.gesture = OpenPalmDetector("roi_extraction/gesture_recognizer/gesture_recognizer.task")
    
    qd = QdrantHelper(host = "localhost",port = 6333, grpc_port = 6334, prefer_grpc = True)
    qd.ensure_collection('palm_vectors', vector_size=128)
    app.state.qdrant = qd
    yield
    cap.release()

app = FastAPI(lifespan=lifespan)

@app.get("/", include_in_schema=False)
def go_ui():
    return RedirectResponse(url="/ui")

doneQ = queue.Queue()

def inference_callback(user_data, result, error):
    idx, t_in = user_data
    t_out = time.time()
    latency_ms = (t_out - t_in) * 1000

    if error:
        print(f"[#{idx}] ERROR @ {t_out:.3f}s  {error}")
    else:
        vec = result.as_numpy("OUTPUT__0").squeeze().astype(float).tolist()
        print(
            f"[#{idx}] OUT @ {t_out:.3f}s  "
            f"latency={latency_ms:.1f} ms  len={len(vec)}"
        )
        app.state.qdrant.insert_vectors(
            collection_name="palm_vectors",
            vectors=[vec],
            ids=[idx],
            payloads=[{"timestamp": t_out}],
        )

    doneQ.put((idx, result))

def stream(cap, triton: TritonClient, to_roi=False):
    counter = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        try:
            while True:
                idx, res = doneQ.get_nowait()
        except queue.Empty:
            pass

        if to_roi:
            roi = extract_palm_roi(frame)
            if roi is not None:
                batch, img = preprocess(roi)
                t_in = time.time()
                print(f"[#{counter}] IN  @ {t_in:.3f}s")
                triton.infer_async(
                    model_name="feature_extraction",
                    inputs={"INPUT__0": batch},
                    callback=inference_callback,
                    user_data=(counter, t_in)
                )
                counter += 1
                frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                frame = np.zeros((224,224,3), dtype=np.uint8)

        _, buf = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buf.tobytes() +
            b"\r\n"
        )


@app.get("/video/raw", include_in_schema=False)
def raw_feed(request: Request):
    return StreamingResponse(
        stream(request.app.state.cap, request.app.state.triton, to_roi=False),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/video/roi", include_in_schema=False)
def roi_feed(request: Request):
    return StreamingResponse(
        stream(request.app.state.cap, request.app.state.triton, to_roi=True),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

with gr.Blocks(title="Palm-Print Identification System") as ui:
    with gr.Row():
        gr.Markdown("### Palm-Print Identification System")
    with gr.Row(equal_height=True):
        gr.HTML('<img src="/video/raw"  style="max-width:720px; border:1px solid #ccc;">')
        gr.HTML('<img src="/video/roi"  style="max-width:480px; border:1px solid #ccc;">')

gr.mount_gradio_app(app, ui, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=7000)
