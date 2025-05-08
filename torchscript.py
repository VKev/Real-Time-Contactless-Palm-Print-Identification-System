import os
import torch
from feature_extraction.model import MyModel
from depth_estimation.depth_anything_v2.dpt import DepthAnythingV2

def export_model_to_triton(model: torch.nn.Module,
                           dummy_input: torch.Tensor,
                           repo_root: str,
                           model_name: str,
                           version: str = "1"):
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input, check_trace=False)

    version_dir = os.path.join(repo_root, model_name, version)
    os.makedirs(version_dir, exist_ok=True)

    output_path = os.path.join(version_dir, "model.pt")
    traced.save(output_path)
    print(f"Saved TorchScript model for '{model_name}' to {output_path}")


if __name__ == "__main__":
    repo_root = "model_repository"
    version   = "1"
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_PATH_DEPTH = "depth_estimation/checkpoints/depth_anything_v2_vits.pth"
    depth_cfg = {
        'encoder':     'vits',
        'features':    64,
        'out_channels':[48, 96, 192, 384],
        'use_bn':      False,
        'use_clstoken':False,
    }
    model_depth = DepthAnythingV2(**depth_cfg).to(device)
    state_dict   = torch.load(MODEL_PATH_DEPTH, map_location="cpu")
    model_depth.load_state_dict(state_dict)
    depth_input  = torch.randn(1, 3, 252, 252, device=device)

    export_model_to_triton(
        model      = model_depth,
        dummy_input= depth_input,
        repo_root  = repo_root,
        model_name = "depth_anything_v2",
        version    = version
    )

    MODEL_PATH_FEAT = "feature_extraction/checkpoints/attempt_8.pth"
    model_feat = MyModel().to(device)
    ckpt       = torch.load(MODEL_PATH_FEAT, map_location="cpu")
    if "model_state_dict" in ckpt:
        model_feat.load_state_dict(ckpt["model_state_dict"])
    else:
        model_feat.load_state_dict(ckpt)
    feat_input = torch.randn(1, 3, 224, 224, device=device)

    export_model_to_triton(
        model       = model_feat,
        dummy_input = feat_input,
        repo_root   = repo_root,
        model_name  = "feature_extraction",
        version     = version
    )
