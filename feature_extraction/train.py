import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import wandb
# 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
try:
    from model import MyModel
    from util import TripletDataset, triplet_collate_fn
    from util import BatchAllTripletLoss
    from util import transform, augmentation
    from util import load_images, get_image_paths
    from test import compute_top1, extract_embeddings
except ImportError:
    from feature_extraction.model import MyModel
    from feature_extraction.util import TripletDataset, triplet_collate_fn
    from feature_extraction.util import BatchAllTripletLoss
    from feature_extraction.util import transform, augmentation
    from feature_extraction.util import load_images, get_image_paths
    from feature_extraction.test import compute_top1, extract_embeddings


def get_model(model_name: str, device: torch.device) -> nn.Module:
    if model_name == "mymodel":
        return MyModel().to(device)
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 128)
        return model.to(device)
    if model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 128)
        return model.to(device)
    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 128)
        return model.to(device)
    if model_name == "vits":
        model = models.vit_s_16(weights=models.ViT_S_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, 128)
        return model.to(device)
    raise ValueError(f"Unsupported model architecture: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training parameters for the feature extraction model.")
    parser.add_argument("--model", type=str, default="mymodel",
                        choices=["mymodel", "resnet50", "resnet101", "vgg16", "vits"],
                        help="Model architecture to use.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint for resuming.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--train_path", type=str, default=r"dataset/TrainAndTest/train",
                        help="Path to the training images folder.")
    parser.add_argument("--test_path", type=str, default=r"dataset/TrainAndTest/test",
                        help="Path to the testing images folder.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=2e-5, help="Weight decay for optimization.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"], help="Device to use for training.")
    parser.add_argument("--wandb", type=str, default="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", help="W&B API key.")
    parser.add_argument("--wandb_project", type=str, default="My-Model", help="W&B project name.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio from train data.")
    parser.add_argument("--margin", type=float, default=0.75, help="Triplet margin.")

    parser.add_argument("--train_negatives", type=int, default=2, help="Number of negatives for training.")
    parser.add_argument("--train_negatives_class", type=int, default=2, help="Number of negative classes in training.")
    parser.add_argument("--test_negatives", type=int, default=2, help="Number of negatives for evaluation.")
    parser.add_argument("--test_negatives_class", type=int, default=2, help="Number of negative classes in evaluation.")

    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["none", "cosine", "onecycle"],
                        help="Learning-rate scheduler.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for cosine scheduler.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training on CUDA.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm; <=0 disables clip.")
    parser.add_argument("--log_every", type=int, default=20, help="Batch logging interval.")
    parser.add_argument("--wandb_watch", action="store_true", help="Enable wandb.watch gradient logging.")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--max_eval_pairs", type=int, default=250000,
                        help="Max verification pairs for ROC/EER computation.")
    parser.add_argument("--far_targets", type=str, default="1e-2,1e-3,1e-4",
                        help="Comma-separated FAR targets for TAR metrics.")
    return parser.parse_args()


def parse_far_targets(raw: str) -> List[float]:
    values: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if 0.0 < value < 1.0:
            values.append(value)
    if not values:
        return [1e-2, 1e-3, 1e-4]
    return sorted(set(values), reverse=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_wandb(args: argparse.Namespace) -> None:
    key = args.wandb.strip()
    if key and key != "your wandb key":
        wandb.login(key=key)
    elif not os.getenv("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "disabled")
        print("[INFO] WANDB_API_KEY is not set. W&B logging is disabled for this run.")

    wandb.init(project=args.wandb_project, config=vars(args))

def overwrite_optimizer_lr(optimizer: optim.Optimizer, new_lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = new_lr
        group["initial_lr"] = new_lr


def sync_scheduler_lr(
    scheduler: Optional[optim.lr_scheduler.LRScheduler],
    optimizer: optim.Optimizer,
    new_lr: float,
) -> None:
    if scheduler is None:
        return

    # keep optimizer param groups aligned
    overwrite_optimizer_lr(optimizer, new_lr)

    if hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [new_lr for _ in optimizer.param_groups]

    # OneCycleLR
    if hasattr(scheduler, "max_lrs"):
        scheduler.max_lrs = [new_lr for _ in optimizer.param_groups]

    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]


def initialize_model(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, optim.Optimizer, int, Optional[Dict]]:
    model = get_model(args.model, device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    start_epoch = 0
    checkpoint = None
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # IMPORTANT: override checkpoint LR with current CLI LR
        overwrite_optimizer_lr(optimizer, args.learning_rate)

        start_epoch = int(checkpoint.get("epoch", 0))
        print(
            f"[INFO] Resumed from checkpoint: {args.checkpoint_path} "
            f"(epoch={start_epoch}, overridden_lr={args.learning_rate})"
        )

    return model, optimizer, start_epoch, checkpoint

def stratified_split_indices(labels: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for label_indices in label_to_indices.values():
        rng.shuffle(label_indices)
        if len(label_indices) <= 1:
            train_indices.extend(label_indices)
            continue

        val_count = max(1, int(round(len(label_indices) * val_ratio)))
        val_count = min(val_count, len(label_indices) - 1)
        val_indices.extend(label_indices[:val_count])
        train_indices.extend(label_indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def select_by_indices(items: List, indices: List[int]) -> List:
    return [items[i] for i in indices]


def setup_dataloaders(args: argparse.Namespace, device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader]:
    image_paths, labels = load_images(args.train_path)
    test_image_paths, test_labels = load_images(args.test_path)

    train_indices, val_indices = stratified_split_indices(labels, args.val_ratio, args.seed)
    train_paths = select_by_indices(image_paths, train_indices)
    train_labels = select_by_indices(labels, train_indices)
    val_paths = select_by_indices(image_paths, val_indices)
    val_labels = select_by_indices(labels, val_indices)

    train_set = TripletDataset(
        train_paths,
        train_labels,
        transform=transform,
        augmentation=augmentation,
        n_negatives=args.train_negatives,
        num_classes_for_negative=args.train_negatives_class,
    )
    val_set = TripletDataset(
        val_paths,
        val_labels,
        transform=transform,
        n_negatives=args.test_negatives,
        num_classes_for_negative=args.test_negatives_class,
    )
    test_set = TripletDataset(
        test_image_paths,
        test_labels,
        transform=transform,
        n_negatives=args.test_negatives,
        num_classes_for_negative=args.test_negatives_class,
    )

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": triplet_collate_fn,
        "num_workers": args.num_workers,
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    print(f"Train samples: {len(train_set)}")
    print(f"Validate samples: {len(val_set)}")
    print(f"Test samples: {len(test_set)}")

    return train_loader, val_loader, test_loader


def build_scheduler(
    args: argparse.Namespace,
    optimizer: optim.Optimizer,
    steps_per_epoch: int,
    resume_state: Optional[Dict],
) -> Tuple[Optional[optim.lr_scheduler.LRScheduler], str]:
    if args.scheduler == "none":
        return None, "none"

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.min_lr,
        )
        step_mode = "epoch"
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=args.epochs,
            steps_per_epoch=max(1, steps_per_epoch),
            pct_start=0.1,
            anneal_strategy="cos",
        )
        step_mode = "batch"

    if resume_state and "scheduler_state_dict" in resume_state:
        try:
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            # IMPORTANT: after loading scheduler state, force it to use current CLI LR
            sync_scheduler_lr(scheduler, optimizer, args.learning_rate)
            print(f"[INFO] Scheduler state restored and LR synced to {args.learning_rate}")
        except Exception as exc:
            print(f"[WARN] Could not load scheduler state dict: {exc}")
            sync_scheduler_lr(scheduler, optimizer, args.learning_rate)
    else:
        sync_scheduler_lr(scheduler, optimizer, args.learning_rate)

    return scheduler, step_mode

def unpack_triplet_features(
    all_features: torch.Tensor,
    num_anchors: int,
    num_negatives_per_anchor: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchors_features = F.normalize(all_features[:num_anchors], p=2, dim=1)
    positives_features = F.normalize(all_features[num_anchors:2 * num_anchors], p=2, dim=1)
    negatives_features = F.normalize(
        all_features[2 * num_anchors:].view(num_anchors, num_negatives_per_anchor, -1),
        p=2,
        dim=2,
    )
    return anchors_features, positives_features, negatives_features


def compute_triplet_stats(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    margin: float,
) -> Dict[str, float]:
    with torch.no_grad():
        pos_dist = torch.norm(anchors - positives, p=2, dim=1)
        neg_dist = torch.norm(anchors.unsqueeze(1) - negatives, p=2, dim=2)
        hardest_neg = neg_dist.min(dim=1).values
        violation = (pos_dist.unsqueeze(1) - neg_dist + margin > 0).float().mean()

        return {
            "pos_dist": float(pos_dist.mean().item()),
            "neg_dist": float(neg_dist.mean().item()),
            "hard_neg_dist": float(hardest_neg.mean().item()),
            "triplet_violation_rate": float(violation.item()),
        }


def aggregate_epoch_stats(weighted_stats: Dict[str, float], weight_sum: float) -> Dict[str, float]:
    if weight_sum <= 0:
        return {k: 0.0 for k in weighted_stats}
    return {k: v / weight_sum for k, v in weighted_stats.items()}


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler.LRScheduler],
    scheduler_step_mode: str,
    scaler: torch.amp.GradScaler,
    triplet_loss: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> Tuple[Dict[str, float], int]:
    model.train()
    autocast_enabled = args.amp and device.type == "cuda"

    weighted_stats = {
        "train_loss": 0.0,
        "train_pos_dist": 0.0,
        "train_neg_dist": 0.0,
        "train_hard_neg_dist": 0.0,
        "train_triplet_violation_rate": 0.0,
    }
    sample_weight_sum = 0.0

    epoch_iterator = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args.epochs}]", unit="batch")
    for step, (all_images, num_anchors, num_negatives_per_anchor) in enumerate(epoch_iterator, start=1):
        all_images = all_images.to(device, non_blocking=(device.type == "cuda"))
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            all_features = model(all_images)
            anchors, positives, negatives = unpack_triplet_features(all_features, num_anchors, num_negatives_per_anchor)
            loss = triplet_loss(anchors, positives, negatives)

        if autocast_enabled:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if scheduler is not None and scheduler_step_mode == "batch":
            scheduler.step()

        batch_stats = compute_triplet_stats(anchors, positives, negatives, args.margin)
        sample_weight = float(num_anchors)
        weighted_stats["train_loss"] += float(loss.item()) * sample_weight
        weighted_stats["train_pos_dist"] += batch_stats["pos_dist"] * sample_weight
        weighted_stats["train_neg_dist"] += batch_stats["neg_dist"] * sample_weight
        weighted_stats["train_hard_neg_dist"] += batch_stats["hard_neg_dist"] * sample_weight
        weighted_stats["train_triplet_violation_rate"] += batch_stats["triplet_violation_rate"] * sample_weight
        sample_weight_sum += sample_weight

        global_step += 1
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_iterator.set_postfix(loss=float(loss.item()), lr=current_lr)

        if step % args.log_every == 0:
            wandb.log(
                {
                    "batch_loss": float(loss.item()),
                    "lr": current_lr,
                    "batch_pos_dist": batch_stats["pos_dist"],
                    "batch_neg_dist": batch_stats["neg_dist"],
                    "batch_hard_neg_dist": batch_stats["hard_neg_dist"],
                    "batch_triplet_violation_rate": batch_stats["triplet_violation_rate"],
                },
                step=global_step,
            )

    return aggregate_epoch_stats(weighted_stats, sample_weight_sum), global_step


def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    triplet_loss: nn.Module,
    margin: float,
    prefix: str,
) -> Dict[str, float]:
    model.eval()
    weighted_stats = {
        f"{prefix}_loss": 0.0,
        f"{prefix}_pos_dist": 0.0,
        f"{prefix}_neg_dist": 0.0,
        f"{prefix}_hard_neg_dist": 0.0,
        f"{prefix}_triplet_violation_rate": 0.0,
    }
    sample_weight_sum = 0.0

    with torch.no_grad():
        for all_images, num_anchors, num_negatives_per_anchor in tqdm(data_loader, desc=f"Evaluating {prefix}"):
            all_images = all_images.to(device, non_blocking=(device.type == "cuda"))
            all_features = model(all_images)
            anchors, positives, negatives = unpack_triplet_features(all_features, num_anchors, num_negatives_per_anchor)
            loss = triplet_loss(anchors, positives, negatives)

            batch_stats = compute_triplet_stats(anchors, positives, negatives, margin)
            sample_weight = float(num_anchors)
            weighted_stats[f"{prefix}_loss"] += float(loss.item()) * sample_weight
            weighted_stats[f"{prefix}_pos_dist"] += batch_stats["pos_dist"] * sample_weight
            weighted_stats[f"{prefix}_neg_dist"] += batch_stats["neg_dist"] * sample_weight
            weighted_stats[f"{prefix}_hard_neg_dist"] += batch_stats["hard_neg_dist"] * sample_weight
            weighted_stats[f"{prefix}_triplet_violation_rate"] += batch_stats["triplet_violation_rate"] * sample_weight
            sample_weight_sum += sample_weight

    return aggregate_epoch_stats(weighted_stats, sample_weight_sum)


def compute_verification_metrics(
    embeddings: np.ndarray,
    paths: List[str],
    far_targets: List[float],
    max_pairs: int,
    seed: int,
) -> Dict[str, float]:
    labels = np.array([Path(p).stem.split("_")[-1] for p in paths])
    n = len(labels)
    if n < 2:
        return {
            "test_roc_auc": 0.0,
            "test_eer": 1.0,
            **{f"test_tar_at_far_{far:.0e}": 0.0 for far in far_targets},
        }

    distances_chunks: List[np.ndarray] = []
    positive_chunks: List[np.ndarray] = []
    for i in range(n - 1):
        diffs = embeddings[i + 1:] - embeddings[i]
        distances_chunks.append(np.linalg.norm(diffs, axis=1))
        positive_chunks.append((labels[i + 1:] == labels[i]).astype(np.uint8))

    distances = np.concatenate(distances_chunks)
    positive_pairs = np.concatenate(positive_chunks)

    if len(distances) > max_pairs > 0:
        rng = np.random.default_rng(seed)
        sampled_idx = rng.choice(len(distances), size=max_pairs, replace=False)
        distances = distances[sampled_idx]
        positive_pairs = positive_pairs[sampled_idx]

    if positive_pairs.min() == positive_pairs.max():
        return {
            "test_roc_auc": 0.0,
            "test_eer": 1.0,
            **{f"test_tar_at_far_{far:.0e}": 0.0 for far in far_targets},
        }

    fpr, tpr, _ = roc_curve(positive_pairs, -distances)
    roc_auc = float(auc(fpr, tpr))
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float(0.5 * (fpr[eer_idx] + fnr[eer_idx]))

    result = {
        "test_roc_auc": roc_auc,
        "test_eer": eer,
    }
    for far in far_targets:
        valid = tpr[fpr <= far]
        result[f"test_tar_at_far_{far:.0e}"] = float(valid.max()) if valid.size else 0.0
    return result


def evaluate_test_identification_and_verification(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    test_path: str,
    far_targets: List[float],
    max_eval_pairs: int,
    seed: int,
) -> Dict[str, float]:
    image_paths = sorted(get_image_paths(test_path))
    embeddings = extract_embeddings(
        model,
        image_paths,
        batch=data_loader.batch_size,
        workers=data_loader.num_workers,
        device=device.type,
        normalize=True,
    )
    top1_accuracy, _ = compute_top1(embeddings, image_paths)
    verification_metrics = compute_verification_metrics(
        embeddings=embeddings,
        paths=image_paths,
        far_targets=far_targets,
        max_pairs=max_eval_pairs,
        seed=seed,
    )
    return {
        "test_top1_accuracy": float(top1_accuracy),
        **verification_metrics,
    }


def build_checkpoint_state(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler.LRScheduler],
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    metrics: Dict[str, float],
    best_top1: float,
    best_val_loss: float,
) -> Dict:
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "best_top1": best_top1,
        "best_val_loss": best_val_loss,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler.is_enabled():
        state["scaler_state_dict"] = scaler.state_dict()
    return state


def save_checkpoint(state: Dict, checkpoint_path: str) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(state, checkpoint_path)
    print(f"[INFO] Checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    args = parse_args()
    far_targets = parse_far_targets(args.far_targets)
    set_seed(args.seed)

    device = torch.device(args.device)
    setup_wandb(args)

    model, optimizer, start_epoch, resume_state = initialize_model(args, device)
    train_loader, val_loader, test_loader = setup_dataloaders(args, device)
    scheduler, scheduler_step_mode = build_scheduler(args, optimizer, len(train_loader), resume_state)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if resume_state and "scaler_state_dict" in resume_state and scaler.is_enabled():
        try:
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        except Exception as exc:
            print(f"[WARN] Could not load GradScaler state dict: {exc}")

    if args.wandb_watch:
        wandb.watch(model, log="gradients", log_freq=max(1, args.log_every))

    print("========Model========")
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    triplet_loss = BatchAllTripletLoss(margin=args.margin)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    loss_log_path = os.path.join(args.checkpoint_dir, "loss.txt")
    global_step = int(resume_state.get("global_step", 0)) if resume_state else 0
    best_top1 = float(resume_state.get("best_top1", -1.0)) if resume_state else -1.0
    best_val_loss = float(resume_state.get("best_val_loss", float("inf"))) if resume_state else float("inf")
    last_completed_epoch = start_epoch

    try:
        with open(loss_log_path, "a", encoding="utf-8") as f:
            f.write("\n")

        for epoch in range(start_epoch, args.epochs):
            train_metrics, global_step = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scheduler_step_mode=scheduler_step_mode,
                scaler=scaler,
                triplet_loss=triplet_loss,
                device=device,
                args=args,
                epoch=epoch,
                global_step=global_step,
            )

            if scheduler is not None and scheduler_step_mode == "epoch":
                scheduler.step()

            val_metrics = evaluate_epoch(
                model=model,
                data_loader=val_loader,
                device=device,
                triplet_loss=triplet_loss,
                margin=args.margin,
                prefix="val",
            )
            test_triplet_metrics = evaluate_epoch(
                model=model,
                data_loader=test_loader,
                device=device,
                triplet_loss=triplet_loss,
                margin=args.margin,
                prefix="test",
            )
            test_id_ver_metrics = evaluate_test_identification_and_verification(
                model=model,
                data_loader=test_loader,
                device=device,
                test_path=args.test_path,
                far_targets=far_targets,
                max_eval_pairs=args.max_eval_pairs,
                seed=args.seed,
            )

            lr = optimizer.param_groups[0]["lr"]
            epoch_metrics = {
                "epoch": epoch + 1,
                "lr": lr,
                **train_metrics,
                **val_metrics,
                **test_triplet_metrics,
                **test_id_ver_metrics,
            }

            wandb.log(epoch_metrics, step=global_step)
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"train_loss={train_metrics['train_loss']:.6f} "
                f"val_loss={val_metrics['val_loss']:.6f} "
                f"test_loss={test_triplet_metrics['test_loss']:.6f} "
                f"test_top1={test_id_ver_metrics['test_top1_accuracy']:.6f} "
                f"test_eer={test_id_ver_metrics['test_eer']:.6f}"
            )

            with open(loss_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"train_loss={train_metrics['train_loss']:.6f}, "
                    f"val_loss={val_metrics['val_loss']:.6f}, "
                    f"test_loss={test_triplet_metrics['test_loss']:.6f}, "
                    f"test_top1={test_id_ver_metrics['test_top1_accuracy']:.6f}, "
                    f"test_eer={test_id_ver_metrics['test_eer']:.6f}, "
                    f"lr={lr:.8f}\n"
                )

            checkpoint_state = build_checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                global_step=global_step,
                metrics=epoch_metrics,
                best_top1=best_top1,
                best_val_loss=best_val_loss,
            )

            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    checkpoint_state,
                    os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
                )

            current_top1 = test_id_ver_metrics["test_top1_accuracy"]
            current_val_loss = val_metrics["val_loss"]
            is_best = (current_top1 > best_top1) or (
                np.isclose(current_top1, best_top1) and current_val_loss < best_val_loss
            )
            if is_best:
                best_top1 = current_top1
                best_val_loss = current_val_loss
                checkpoint_state["best_top1"] = best_top1
                checkpoint_state["best_val_loss"] = best_val_loss
                save_checkpoint(
                    checkpoint_state,
                    os.path.join(args.checkpoint_dir, "best_model.pth"),
                )

            last_completed_epoch = epoch + 1

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Error occurred during training: {e}")
        raise
    finally:
        try:
            final_state = build_checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=last_completed_epoch,
                global_step=global_step,
                metrics={},
                best_top1=best_top1,
                best_val_loss=best_val_loss,
            )
            save_checkpoint(final_state, os.path.join(args.checkpoint_dir, "final_model.pth"))
        except Exception as e:
            print(f"[WARN] Error saving final checkpoint: {e}")
        wandb.finish()
