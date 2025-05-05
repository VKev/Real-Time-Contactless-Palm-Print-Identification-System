import os
import cv2
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)

def load_and_preprocess(path):
    """For feature_extraction: resize to 224×224, normalize, channel-first."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    # channel-first and add batch dim
    return np.transpose(img, (2, 0, 1))[None, ...]

def load_and_preprocess_depth(path, target_width=518):
    """For depthanythingv2: resize width to 518, keep aspect ratio, normalize."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    new_h = int(h * target_width / w)
    # ensure new_h is a multiple of 14 for your model (if needed, you can round up)
    new_h = ((new_h + 13) // 14) * 14
    img = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[None, ...]  # shape (1,3,H,518)

def make_input(name, array):
    inp = httpclient.InferInput(name, array.shape, "FP32")
    inp.set_data_from_numpy(array)
    return inp

def test_feature_dist(img_path1, img_path2):
    # preprocess
    img1 = load_and_preprocess(img_path1)
    img2 = load_and_preprocess(img_path2)
    # wrap as Triton inputs
    input1 = make_input("INPUT__0", img1)
    input2 = make_input("INPUT__0", img2)
    output = httpclient.InferRequestedOutput("OUTPUT__0")

    # infer both
    resp1 = client.infer(model_name="feature_extraction",
                         inputs=[input1],
                         outputs=[output])
    resp2 = client.infer(model_name="feature_extraction",
                         inputs=[input2],
                         outputs=[output])

    feat1 = resp1.as_numpy("OUTPUT__0").reshape(-1)
    feat2 = resp2.as_numpy("OUTPUT__0").reshape(-1)
    dist = np.linalg.norm(feat1 - feat2)

    print("=== Feature Extraction ===")
    print(f"Feature 1 norm: {np.linalg.norm(feat1):.4f}")
    print(f"Feature 2 norm: {np.linalg.norm(feat2):.4f}")
    print(f"Euclidean distance : {dist:.4f}\n")

def test_depth(img_path, output_dir="depth_outputs"):
    # preprocess
    depth_in = load_and_preprocess_depth(img_path)
    input_depth = make_input("INPUT__0", depth_in)
    output = httpclient.InferRequestedOutput("OUTPUT__0")

    # infer
    resp = client.infer(model_name="depth_anything_v2",
                        inputs=[input_depth],
                        outputs=[output])
    # shape (1, H, 518)
    depth_map = resp.as_numpy("OUTPUT__0")[0]

    print("=== Depth Estimation ===")
    print(f"Depth map shape: {depth_map.shape}")

if __name__ == "__main__":
    # paths to your test images
    img1_path = r"C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\TrainAndTest\test\3_F_026.JPG"
    img2_path = r"C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\TrainAndTest\test\6_F_026.JPG"

    # 1) Feature distance
    test_feature_dist(img1_path, img2_path)

    # 2) Depth inference on each
    test_depth(img1_path)
    test_depth(img2_path)
