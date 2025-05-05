import glob
import gradio as gr
import matplotlib
import numpy as np
from PIL import Image
import torch
import tempfile
from gradio_imageslider import ImageSlider
from utils import apply_depth_mask, image2tensor, interpolate
from depth_anything_v2.dpt import DepthAnythingV2

css = """
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 80vh;
}
#img-display-output {
    max-height: 80vh;
}
#download {
    height: 62px;
}
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder = 'vits'
model = DepthAnythingV2(**model_configs[encoder])
state_dict = torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

def predict_depth(image):
    image, (h, w) = image2tensor(raw_image = image, input_size=518)
    result = model.forward(image).detach()
    result = interpolate(result, h, w)
    return result

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Background Removal demo")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
    submit = gr.Button(value="Background remove")
    gray_depth_file = gr.File(label="Grayscale depth map", elem_id="download",)
    raw_file = gr.File(label="16-bit raw output (can be considered as disparity)", elem_id="download",)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    def on_submit(image):
        original_image = image.copy()

        h, w = image.shape[:2]

        
        depth = predict_depth(image[:, :, ::-1])
        raw_depth = Image.fromarray(depth.astype('uint16'))
        tmp_raw_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth.save(tmp_raw_depth.name)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)

        gray_depth = Image.fromarray(depth)
        tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        gray_depth.save(tmp_gray_depth.name)

        masked_image = apply_depth_mask(original_image, depth, threshold=0.5)
        return [(original_image, masked_image), tmp_gray_depth.name, tmp_raw_depth.name]

    submit.click(on_submit, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file, raw_file])

    # example_files = glob.glob('assets/examples/*')
    # examples = gr.Examples(examples=example_files, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file, raw_file], fn=on_submit)


if __name__ == '__main__':
    demo.queue().launch()