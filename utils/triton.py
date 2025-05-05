import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from functools import partial
import os, sys
try:
    here = os.path.dirname(__file__)
    root = os.path.dirname(here)
    sys.path.insert(0, root)
    from depth_estimation.utils import interpolate_np, apply_depth_mask, image2tensor
except ImportError:
    print()
    
dtype_map = {
    np.dtype('float32'): "FP32",
    np.dtype('float64'): "FP64",
    np.dtype('int32'):   "INT32",
    np.dtype('int64'):   "INT64",
    np.dtype('uint8'):   "UINT8",
    np.dtype('bool'):    "BOOL",
}

class TritonClient:
    def __init__(self, url: str = "localhost:8001", verbose: bool = False):
        self.client = InferenceServerClient(url=url, verbose=verbose)

    def infer(
        self,
        model_name: str,
        inputs: dict,
        outputs: list = None,
        model_version: str = "",
        timeout: float = None
    ) -> dict:
        return self._do_infer(model_name, inputs, outputs, model_version, timeout, async_mode=False)

    def infer_async(
        self,
        model_name: str,
        inputs: dict,
        outputs: list = None,
        model_version: str = "",
        timeout: float = None,
        callback=None,
        user_data=None
    ):
        return self._do_infer(model_name, inputs, outputs, model_version, timeout, async_mode=True,
                              callback=callback, user_data=user_data)

    def _do_infer(
        self,
        model_name: str,
        inputs: dict,
        outputs: list,
        model_version: str,
        timeout: float,
        async_mode: bool,
        callback=None,
        user_data=None
    ):
        if outputs is None:
            meta = self.client.get_model_metadata(model_name=model_name,
                                                  model_version=model_version)
            outputs = [o.name for o in meta.outputs]

        infer_inputs = []
        for name, array in inputs.items():
            arr = np.ascontiguousarray(array)
            triton_dtype = dtype_map.get(arr.dtype)
            if triton_dtype is None:
                raise ValueError(f"Unsupported numpy dtype: {arr.dtype} for input '{name}'")
            infer_in = InferInput(name, arr.shape, triton_dtype)
            infer_in.set_data_from_numpy(arr)
            infer_inputs.append(infer_in)

        infer_outputs = [InferRequestedOutput(o) for o in outputs]

        if async_mode:
            cb = callback or (lambda ud, result, error: None)
            return self.client.async_infer(
                model_name=model_name,
                inputs=infer_inputs,
                outputs=infer_outputs,
                callback=partial(cb, user_data),
                model_version=model_version,
                timeout=timeout
            )
        else:
            resp = self.client.infer(
                model_name=model_name,
                inputs=infer_inputs,
                outputs=infer_outputs,
                model_version=model_version,
                timeout=timeout
            )
            return {o: resp.as_numpy(o) for o in outputs}

import os
import glob
from PIL import Image
import cv2
if __name__ == "__main__":
    client = TritonClient("localhost:8001")
    dummy = np.random.random((1, 3, 224, 224)).astype(np.float32)
    out = client.infer(model_name="feature_extraction", inputs={"INPUT__0": dummy})
    print("Sync inference result shapes:", {k: v.shape for k, v in out.items()})
    
    dummy_depth = np.random.random((1, 3, 252, 252)).astype(np.float32)
    depth_out = client.infer(model_name="depth_anything_v2", inputs={"INPUT__0": dummy_depth})
    print("Sync inference result shapes:", {k: v.shape for k, v in depth_out.items()})
    
    import queue, time
    doneQ = queue.Queue()
    def cb(user_data, result, error):
        if error:
            print("Async error:", error)
        else:
            print("Async feature extraction result shape:", result.as_numpy("OUTPUT__0").shape)

    fut = client.infer_async("feature_extraction", {"INPUT__0": dummy}, callback=cb, user_data=None)
    fut = client.infer_async("depth_anything_v2", {"INPUT__0": dummy_depth}, callback=cb, user_data=None)
    time.sleep(1)


    def remove_background_and_save(
    client: TritonClient,
    img_path: str,
    out_dir: str,
    threshold: float = 0.5,
    depth_size: int = 252,
    ):
        """
        Load an image, infer depth via Triton, apply mask, and save results.
        - Original image:      saved as <basename>_orig.png
        - Grayscale depth map: <basename>_depth_gray.png
        - Masked image:        <basename>_masked.png
        """
        # 1) Load input
        img = np.array(Image.open(img_path).convert("RGB"))
        print(img.shape, img.dtype)
        h, w = img.shape[:2]

        # 2) Prepare and infer depth
        tensor, (h, w) = image2tensor(img, input_size=depth_size)  # (1,3,depth_size,depth_size)
        depth_out = client.infer(
            model_name="depth_anything_v2",
            inputs={"INPUT__0": tensor.cpu().numpy().astype(np.float32)},
        )["OUTPUT__0"]  # shape: (1,1,depth_size,depth_size) or (1,depth_size,depth_size)

        # 3) Upsample & normalize
        depth_map = interpolate_np(depth_out, h, w)  # returns np.ndarray (h,w) or (1,h,w)
        # if shape is (1,h,w), squeeze
        if depth_map.ndim == 3 and depth_map.shape[0] == 1:
            depth_map = depth_map[0]

        # convert raw depth to 16-bit
        raw16 = Image.fromarray(depth_map.astype("uint16"))
        # normalize for gray (0–255)
        depth_norm = ((depth_map - depth_map.min()) / (depth_map.ptp()) * 255.0).astype("uint8")
        gray8 = Image.fromarray(depth_norm)

        # 4) Apply mask
        masked = apply_depth_mask(img, depth_norm, threshold=threshold)

        # 5) Build output filenames
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_orig   = os.path.join(out_dir, f"{base}_orig.png")
        out_depth  = os.path.join(out_dir, f"{base}_depth_gray.png")
        out_raw16  = os.path.join(out_dir, f"{base}_depth_raw16.png")
        out_masked = os.path.join(out_dir, f"{base}_masked.png")

        # 6) Ensure output dir exists
        os.makedirs(out_dir, exist_ok=True)

        # 7) Save everything
        Image.fromarray(img).save(out_orig)
        gray8.save(out_depth)
        raw16.save(out_raw16)
        masked.save(out_masked)

        print(f"Processed {img_path}:")
        print(f"  • original → {out_orig}")
        print(f"  • gray depth → {out_depth}")
        print(f"  • raw16 depth → {out_raw16}")
        print(f"  • masked → {out_masked}")

    input = r'C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\RealisticSet\Roi\video_frames_2\frame_1.jpg'
    output = r'C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\RealisticSet\Roi\video_frames_2\test'
    # Expand input glob or directory
    if os.path.isdir(input):
        pattern = os.path.join(input, "*.*")
    else:
        pattern = input
    img_paths = sorted(glob.glob(pattern))

    if not img_paths:
        print(f"No images found with pattern: {pattern}")
        exit(1)

    # Process each
    for img_path in img_paths:
        try:
            remove_background_and_save(
                client=client,
                img_path=img_path,
                out_dir=output,
                threshold=0.5,
                depth_size=252,
            )
        except Exception as e:
            print(f"❌ Failed on {img_path}: {e}")