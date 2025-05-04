import argparse
from pathlib import Path
import sys

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns  # new
from sklearn.manifold import TSNE

try:
    import umap.umap_ as umap  # type: ignore
except ModuleNotFoundError as e:
    sys.stderr.write(
        "[ERROR] Could not import 'umap.umap_'. Make sure you have installed\n"
        "        the correct package with:  pip install --upgrade umap-learn\n"
    )
    raise e

from model import MyModel
from util import ImageDataset, transform, get_image_paths
from model.utils import print_total_params
import pandas as pd  # new

# ============================= CLI ============================= #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference on a trained model, compute Top-1 accuracy with "
            "NearestNeighbors, and visualise embeddings with UMAP & t-SNE via Seaborn."
        )
    )
    parser.add_argument("-c", "--checkpoint", type=Path,
                        default=Path("checkpoints/attempt_8.pth"), help="Model checkpoint")
    parser.add_argument("-d", "--data-dir", type=Path,
                        default=Path("../../Dataset/Palm-Print/TrainAndTest/test"), help="Image directory")
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-j", "--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vis-dir", type=Path, default=Path("visualize"))
    parser.add_argument("--vis-classes", type=int, default=20, metavar="N",
                        help="Plot at most N classes (default: 20)")
    return parser.parse_args()

# =========================== Helpers =========================== #

def load_model(ckpt: Path, device: str) -> torch.nn.Module:
    state = torch.load(ckpt, map_location=device)
    model = MyModel().to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model

def extract_embeddings(model: torch.nn.Module,
                       paths: list[Path],
                       batch: int,
                       workers: int,
                       device: str) -> np.ndarray:
    loader = DataLoader(
        ImageDataset(paths, transform),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device == "cuda")
    )
    vecs: list[np.ndarray] = []
    for imgs in tqdm(loader, desc="Extracting embeddings"):
        imgs = imgs.to(device)
        with torch.no_grad():
            vecs.append(model(imgs).cpu().numpy())
    return np.vstack(vecs)

def compute_top1(emb: np.ndarray, paths: list[Path] | list[str]):
    labels = np.array([
        Path(p).stem.split("_")[-1]
        if isinstance(p, (str, Path)) else str(p).split("_")[-1]
        for p in paths
    ])
    knn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(emb)
    _, idx = knn.kneighbors(emb)
    return float((labels == labels[idx[:, 1]]).mean()), labels

def save_umap_plot(emb: np.ndarray,
                   labels: np.ndarray,
                   vis_dir: Path,
                   limit_classes: int):
    """Plot up to `limit_classes` distinct label groups with Seaborn & UMAP."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")  # seaborn style

    # select classes
    unique = np.unique(labels)
    sel = unique[:limit_classes]
    mask = np.isin(labels, sel)
    emb_sel = emb[mask]
    lbl_sel = labels[mask]

    # UMAP projection
    proj = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        n_epochs=200
    ).fit_transform(emb_sel)

    # build DataFrame
    df = pd.DataFrame({
        "UMAP-1": proj[:, 0],
        "UMAP-2": proj[:, 1],
        "label": lbl_sel
    })

    # plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df,
        x="UMAP-1",
        y="UMAP-2",
        hue="label",
        palette=sns.color_palette("hsv", len(sel)),
        s=20,
        alpha=0.6,
        legend=False
    )
    plt.title(f"UMAP Projection (first {limit_classes} classes)")
    plt.tight_layout()

    out = vis_dir / f"embeddings_umap_top{limit_classes}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[INFO] UMAP visualisation saved → {out}")

def save_tsne_plot(emb: np.ndarray,
                   labels: np.ndarray,
                   vis_dir: Path,
                   limit_classes: int):
    """Plot up to `limit_classes` distinct label groups with Seaborn & t-SNE."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")

    # select classes
    unique = np.unique(labels)
    sel = unique[:limit_classes]
    mask = np.isin(labels, sel)
    emb_sel = emb[mask]
    lbl_sel = labels[mask]

    # t-SNE projection
    proj = TSNE(
        n_components=2,
        perplexity=7,
        n_iter=2000,
        metric="euclidean",
        random_state=42
    ).fit_transform(emb_sel)

    # build DataFrame
    df = pd.DataFrame({
        "t-SNE-1": proj[:, 0],
        "t-SNE-2": proj[:, 1],
        "label": lbl_sel
    })

    # plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df,
        x="t-SNE-1",
        y="t-SNE-2",
        hue="label",
        palette=sns.color_palette("hsv", len(sel)),
        s=20,
        alpha=0.6,
        legend=False
    )
    plt.title(f"t-SNE Projection (first {limit_classes} classes)")
    plt.tight_layout()

    out = vis_dir / f"embeddings_tsne_top{limit_classes}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[INFO] t-SNE visualisation saved → {out}")

def main():
    args = parse_args()
    print(f"[INFO] Loading model from {args.checkpoint} on {args.device}")
    model = load_model(args.checkpoint, args.device)
    print_total_params(model)

    paths = [Path(p) for p in get_image_paths(str(args.data_dir))]
    print(f"[INFO] Found {len(paths)} images in {args.data_dir}")

    emb = extract_embeddings(
        model, paths,
        args.batch_size,
        args.num_workers,
        args.device
    )
    acc, lbls = compute_top1(emb, paths)
    print(f"[RESULT] Top-1 Accuracy: {acc:.4f}")

    print("[INFO] Building UMAP plot …")
    save_umap_plot(emb, lbls, args.vis_dir, args.vis_classes)

    print("[INFO] Building t-SNE plot …")
    save_tsne_plot(emb, lbls, args.vis_dir, args.vis_classes)

if __name__ == "__main__":
    main()
