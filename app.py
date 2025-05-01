import cv2, gradio as gr, uvicorn, importlib.resources as pkg
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from roi_extraction.util import extract_palm_roi
from utils.preprocess import preprocess
@asynccontextmanager
async def lifespan(_: FastAPI):
    cap = cv2.VideoCapture(0) 
    _.state.cap = cap
    yield
    cap.release()

app = FastAPI(lifespan=lifespan)

def stream(cap, to_roi=False, to_gray=False):
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if to_roi:
            roi = extract_palm_roi(frame)
            if roi is not None:
                pre = preprocess(roi)
                frame = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)

        elif to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        
@app.get("/video/raw")
def raw_feed(request: Request):
    return StreamingResponse(stream(request.app.state.cap),
           media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video/roi")
def roi_feed(request: Request):
    return StreamingResponse(
        stream(request.app.state.cap, to_roi=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

asset_dir = pkg.files("gradio") / "static"
if asset_dir.is_dir():
    app.mount("/static", StaticFiles(directory=asset_dir), name="static")

with gr.Blocks(title="Palm-Print Identification System") as ui:
    with gr.Row():
        gr.Markdown("### Palm-Print Identification System")
    with gr.Row(equal_height=True):
        gr.HTML('<img src="/video/raw"  style="width:100%;max-width:720px;border:1px solid #ccc;">')
        gr.HTML('<img src="/video/roi" style="width:100%;max-width:480px;border:1px solid #ccc;">')

gr.mount_gradio_app(app, ui, path="/ui")        

@app.get("/")
def go_ui():
    return RedirectResponse(url="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
