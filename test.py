import os
import cv2
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)

def load_and_preprocess(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img

# img1_path = r"C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\TrainAndTest\train\07198_s2_359.bmp"
# img2_path = r"C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\TrainAndTest\train\07191_s1_359.bmp"
# # img2_path = r"C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\TrainAndTest\train\11331_s1_566.bmp" #differ class

img1_path = r"C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\TrainAndTest\test\3_F_026.JPG"
img2_path = r"C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\TrainAndTest\test\6_F_026.JPG"
# img2_path = r"C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\TrainAndTest\test\3_F_015.JPG" #differ class

img1 = load_and_preprocess(img1_path)
img2 = load_and_preprocess(img2_path)

def make_input(name, array):
    inp = httpclient.InferInput(name, array.shape, "FP32")
    inp.set_data_from_numpy(array)
    return inp

input1 = make_input("INPUT__0", img1)
input2 = make_input("INPUT__0", img2)

output = httpclient.InferRequestedOutput("OUTPUT__0")

resp1 = client.infer(model_name="feature_extraction",
                     inputs=[input1],
                     outputs=[output])
resp2 = client.infer(model_name="feature_extraction",
                     inputs=[input2],
                     outputs=[output])

feat1 = resp1.as_numpy("OUTPUT__0").reshape(-1)
feat2 = resp2.as_numpy("OUTPUT__0").reshape(-1)

dist = np.linalg.norm(feat1 - feat2)

print(f"Feature 1 norm: {np.linalg.norm(feat1):.4f}")
print(f"Feature 2 norm: {np.linalg.norm(feat2):.4f}")
print(f"Euclidean distance between features: {dist:.4f}")
