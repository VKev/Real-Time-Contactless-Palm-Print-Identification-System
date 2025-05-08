import argparse
from pathlib import Path
import sys
import random

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns  # new
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc  # new
import pandas as pd  # needed for visualizations

try:
    import umap.umap_ as umap  # type: ignore
except ModuleNotFoundError as e:
    sys.stderr.write(
        "[ERROR] Could not import 'umap.umap_'. Make sure you have installed\n"
        "        the correct package with:  pip install --upgrade umap-learn\n"
    )
    raise e
try:
    from model import MyModel
    from model.utils import print_total_params
    from util import ImageDataset, transform, get_image_paths
except ImportError:
    from feature_extraction.model.utils import print_total_params
    from feature_extraction.model import MyModel
    from feature_extraction.util import ImageDataset, transform, get_image_paths

# ============================= CLI ============================= #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference on a trained model, compute Top-1 accuracy with "
            "NearestNeighbors, and visualise embeddings with UMAP & t-SNE via Seaborn."
        )
    )
    parser.add_argument("-c", "--checkpoint", type=Path,
                        default=Path("checkpoints/attempt_6.pth"), help="Model checkpoint")
    parser.add_argument("-d", "--data-dir", type=Path,
                        default=Path("../../Dataset/Palm-Print/TrainAndTest/test"), help="Image directory")
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-j", "--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vis-dir", type=Path, default=Path("visualize"))
    parser.add_argument("--vis-classes", type=int, default=20, metavar="N",
                        help="Plot at most N classes (default: 20)")
    parser.add_argument("--register-ratio", type=float, default=0.5,
                        help="Ratio of embeddings to use as registered samples (default: 0.5)")
    parser.add_argument("--unregistered-ratio", type=float, default=0.5,
                        help="Ratio of classes to treat as unregistered (default: 0.5)")
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

def split_embeddings(embeddings: np.ndarray, paths: list[Path], register_ratio: float, unregistered_ratio: float):
    """
    Split embeddings into register and verifier sets.
    
    Args:
        embeddings: The extracted feature embeddings
        paths: List of paths to the original images
        register_ratio: Ratio of data to use for registration
        unregistered_ratio: Ratio of classes to treat as unregistered
    
    Returns:
        register_emb: Embeddings for registered samples
        register_labels: Labels for registered samples
        verifier_emb: Embeddings for verification samples
        verifier_labels: Labels for verification samples
        registered_classes: Set of registered classes
    """
    # Extract labels from paths
    labels = np.array([Path(p).stem.split("_")[-1] for p in paths])
    
    # Get unique classes
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    # Determine registered and unregistered classes
    n_unregistered = int(n_classes * unregistered_ratio)
    n_registered = n_classes - n_unregistered
    
    # Randomly select classes to be registered
    random.seed(42)  # For reproducibility
    registered_classes = set(random.sample(list(unique_classes), n_registered))
    unregistered_classes = set(unique_classes) - registered_classes
    
    # Create masks for register and verifier sets
    registered_mask = np.array([label in registered_classes for label in labels])
    
    # For register set, take register_ratio of registered samples
    register_indices = []
    for cls in registered_classes:
        cls_indices = np.where(labels == cls)[0]
        n_register = int(len(cls_indices) * register_ratio)
        register_indices.extend(cls_indices[:n_register])
    
    # All remaining registered samples and all unregistered samples go to verifier
    verifier_indices = list(set(range(len(labels))) - set(register_indices))
    
    # Extract the actual embeddings and labels
    register_emb = embeddings[register_indices]
    register_labels = labels[register_indices]
    verifier_emb = embeddings[verifier_indices]
    verifier_labels = labels[verifier_indices]
    
    print(f"[INFO] Split data into {len(register_labels)} registered samples and {len(verifier_labels)} verification samples")
    print(f"[INFO] {len(registered_classes)} registered classes, {len(unregistered_classes)} unregistered classes")
    
    return register_emb, register_labels, verifier_emb, verifier_labels, registered_classes

def compute_verification_metrics(register_emb: np.ndarray, 
                                register_labels: np.ndarray,
                                verifier_emb: np.ndarray, 
                                verifier_labels: np.ndarray,
                                registered_classes: set):
    """
    Compute verification metrics at different distance thresholds.
    
    Args:
        register_emb: Embeddings from registered samples
        register_labels: Labels for registered samples
        verifier_emb: Embeddings from verifier samples
        verifier_labels: Labels for verifier samples
        registered_classes: Set of registered class names
    
    Returns:
        thresholds: Array of distance thresholds
        fpr: False Positive Rate at each threshold
        tpr: True Positive Rate at each threshold
        auc: Area Under the Curve
    """
    # Compute distances between verifier and register embeddings
    distances = []
    ground_truth = []
    
    print("[INFO] Computing distances and verification metrics...")
    
    for i, v_emb in enumerate(tqdm(verifier_emb)):
        # Compute distance to all registered embeddings
        dists = np.sqrt(np.sum((register_emb - v_emb) ** 2, axis=1))
        min_dist = np.min(dists)
        distances.append(min_dist)
        
        # Ground truth: is this verifier sample from a registered class?
        v_label = verifier_labels[i]
        is_registered = v_label in registered_classes
        ground_truth.append(is_registered)
    
    # Convert to numpy arrays
    distances = np.array(distances)
    ground_truth = np.array(ground_truth)
    
    # Compute ROC curve (invert distances since lower distance = higher similarity)
    fpr, tpr, thresholds = roc_curve(ground_truth, -distances)
    roc_auc = auc(fpr, tpr)
    
    # Convert thresholds back to positive distances
    thresholds = -thresholds
    
    return thresholds, fpr, tpr, roc_auc

def save_roc_curve(fpr: np.ndarray, 
                  tpr: np.ndarray, 
                  auc_score: float,
                  vis_dir: Path):
    """Create and save ROC curve visualization."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    out = vis_dir / "palm_verification_roc.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[INFO] ROC curve visualization saved → {out}")

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
    
    # Split data for verification experiments
    print("[INFO] Setting up verification experiment...")
    register_emb, register_labels, verifier_emb, verifier_labels, registered_classes = split_embeddings(
        emb, paths, args.register_ratio, args.unregistered_ratio
    )
    
    # Compute verification metrics
    thresholds, fpr, tpr, roc_auc = compute_verification_metrics(
        register_emb, register_labels, verifier_emb, verifier_labels, 
        registered_classes
    )
    
    # Visualize ROC curve
    print("[INFO] Building ROC curve...")
    save_roc_curve(fpr, tpr, roc_auc, args.vis_dir)
    print("thresholds: ", thresholds)

if __name__ == "__main__":
    main()
