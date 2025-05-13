import torch
import argparse
from typing import Tuple
import wandb
import os
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
try:
    from model import MyModel
    from util import TripletDataset, CombinedDataset, triplet_collate_fn
    from util import BatchAllTripletLoss
    from util import transform, augmentation
    from util import load_images, get_image_paths
    from test import compute_top1, extract_embeddings
except ImportError:
    from feature_extraction.model import MyModel
    from feature_extraction.util import TripletDataset, CombinedDataset, triplet_collate_fn
    from feature_extraction.util import BatchAllTripletLoss
    from feature_extraction.util import transform, augmentation
    from feature_extraction.util import load_images, get_image_paths
    from feature_extraction.test import compute_top1, extract_embeddings

def get_model(model_name: str, device: str) -> nn.Module:
    """Initialize the selected model architecture."""
    if model_name == "mymodel":
        return MyModel().to(device)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 128)  # Match MyModel output dim
        return model.to(device)
    elif model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 128)
        return model.to(device)
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 128)
        return model.to(device)
    elif model_name == "vits":
        model = models.vit_s_16(weights=models.ViT_S_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, 128)
        return model.to(device)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training parameters for the model.')
    parser.add_argument('--model', type=str, default='mymodel',
                      choices=['mymodel', 'resnet50', 'resnet101', 'vgg16', 'vits'],
                      help='Model architecture to use')
    parser.add_argument('--checkpoint_path', type=str, default="", help='Path to checkpoint file for continuing training')
    parser.add_argument('--train_path', type=str, default=r"../../Dataset/Palm-Print/TrainAndTest/train", help='Path to the training images folder')
    parser.add_argument('--test_path', type=str, default=r"../../Dataset/Palm-Print/TrainAndTest/test", help='Path to the testing images folder')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=2e-5, help='Weight decay for optimization')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train the model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda'], help='Device to use for training (cpu or cuda)')
    parser.add_argument('--wandb', type=str, default='your wandb key', help='your wandb key')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    parser.add_argument('--train_negatives', type=int, default=1, help='Number of negatives for training')
    parser.add_argument('--train_negatives_class', type=int, default=1, help='Number of negative labels')
    parser.add_argument('--test_negatives', type=int, default=1, help='Number of test negatives')
    parser.add_argument('--test_negatives_class', type=int, default=1, help='Number of test negatives per class')
    return parser.parse_args()

def initialize_model(args: argparse.Namespace) -> Tuple[torch.nn.Module, optim.Optimizer, object, int]:
    def learning_rate(epoch):
        return args.learning_rate
    if not args.checkpoint_path:
        start_epoch = 0
        model = get_model(args.model, args.device)
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = 1
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)
    else:
        checkpoint = torch.load(args.checkpoint_path)

        model = get_model(args.model, args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        start_epoch = checkpoint['epoch']
        
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = args.learning_rate
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler, start_epoch


def setup_wandb():
    wandb.login(key=args.wandb)
    wandb.init(
        project="My-Model",
    )

def setup_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    image_paths, labels = load_images(args.train_path)
    test_image_paths, test_labels = load_images(args.test_path)
    
    test_set = TripletDataset(
        test_image_paths, test_labels, transform=transform,
        n_negatives=args.test_negatives, num_classes_for_negative=args.test_negatives_class
    )

    train_set = TripletDataset(
        image_paths, labels, transform=transform,
        n_negatives=args.train_negatives, num_classes_for_negative=args.train_negatives_class
    )
    augmentation_set = TripletDataset(
        image_paths, labels, transform=transform, augmentation=augmentation,
        n_negatives=args.train_negatives, num_classes_for_negative=args.train_negatives_class
    )

    train_set = CombinedDataset(train_set,augmentation_set)

    print("Train samples: ", int(0.9 * len(train_set)))
    print("Validate samples: ", len(train_set) - int(0.9 * len(train_set)))
    print("Test samples: ", len(test_set))

    train_size = int(0.9 * len(train_set))
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=triplet_collate_fn,
        num_workers=args.num_workers,
        persistent_workers=False,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=triplet_collate_fn,
        num_workers=args.num_workers,
        persistent_workers=False,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=triplet_collate_fn,
        num_workers=args.num_workers,
        persistent_workers=False,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader

def train_epoch(model: torch.nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                scheduler: object, triplet_loss: object, device: torch.device, 
                epoch: int, total_epochs: int) -> float:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    total_batches = 0
    
    epoch_iterator = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{total_epochs}]", unit="batch")

    for all_images, num_anchors, num_negatives_per_anchor in epoch_iterator:
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        all_images = all_images.to(device)
        all_features = model.forward(all_images)
        
        # Split and normalize features
        anchors_features = F.normalize(all_features[:num_anchors], p=2, dim=1)
        positives_features = F.normalize(all_features[num_anchors:2*num_anchors], p=2, dim=1)
        negatives_features = F.normalize(
            all_features[2*num_anchors:].view(num_anchors, num_negatives_per_anchor, -1),
            p=2, dim=2
        )
        
        # Calculate loss and update
        loss = triplet_loss(anchors_features, positives_features, negatives_features)
        wandb.log({"batch_loss": loss})

        if loss.item() > 0:
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            gradient_histograms = {
            f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())
            for name, param in model.named_parameters() if param.grad is not None
            }
            wandb.log(gradient_histograms)
            
        total_batches += 1
        epoch_iterator.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        
        scheduler.step()
    
    return running_loss / total_batches if total_batches > 0 else running_loss

def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device, 
            triplet_loss: object, is_test: bool = False, test_path: str = None) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for all_images, num_anchors, num_negatives_per_anchor in tqdm(data_loader, desc="Evaluating"):
            all_images = all_images.to(device)
            all_features = model.forward(all_images)

            anchors_features = all_features[:num_anchors]
            positives_features = all_features[num_anchors:2*num_anchors]
            negatives_features = all_features[2*num_anchors:].view(
                num_anchors, num_negatives_per_anchor, -1
            )

            anchors_features = F.normalize(anchors_features, p=2, dim=1)
            positives_features = F.normalize(positives_features, p=2, dim=1)
            negatives_features = F.normalize(negatives_features, p=2, dim=2)

            loss = triplet_loss(anchors_features, positives_features, negatives_features)
            if loss.item() > 0:
                total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches if total_batches > 0 else total_loss
    
    # Calculate top-1 accuracy if this is test data
    top1_accuracy = 0.0
    if is_test and test_path:
        # Use extract_embeddings from test.py
        image_paths = get_image_paths(test_path)
        paths = [path for path in image_paths]
        
        # Extract embeddings using the function from test.py
        embeddings = extract_embeddings(
            model, 
            paths, 
            batch=data_loader.batch_size, 
            workers=data_loader.num_workers, 
            device=device
        )
        
        top1_accuracy, _ = compute_top1(embeddings, paths)
    
    if not is_test:
        return avg_loss, None          # always a tuple
    else:
        return avg_loss, top1_accuracy

def save_checkpoint(model: torch.nn.Module, optimizer: optim.Optimizer, scheduler: object, 
                   epoch: int, loss: float, checkpoint_dir: str = "checkpoints"):
    """Save model checkpoint and training state."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    args = parse_args()
    setup_wandb()
    model, optimizer, scheduler, start_epoch = initialize_model(args)
    train_loader, val_loader, test_loader = setup_dataloaders(args)

    print("========Model========")
    print(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    triplet_loss = BatchAllTripletLoss(margin=0.75)

    torch.cuda.empty_cache()
    try:
        with open("checkpoints/loss.txt", "a") as f:
            f.write(f"\n")
            
        test_loss, test_acc = evaluate(model, test_loader, args.device, triplet_loss, 
                                      is_test=True, test_path=args.test_path)
        torch.cuda.empty_cache()
        print("test loss: ", test_loss)
        print("test top-1 accuracy: ", test_acc)
        
        for epoch in range(start_epoch, args.epochs):
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler,
                triplet_loss, args.device, epoch, args.epochs
            )
            
            # Save checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch+1, train_loss)
            
            # Evaluate
            torch.cuda.empty_cache()
            val_loss, _ = evaluate(model, val_loader, args.device, triplet_loss,is_test=False)
            torch.cuda.empty_cache()
            test_loss, test_acc = evaluate(model, test_loader, args.device, triplet_loss, 
                                          is_test=True, test_path=args.test_path)
            
            # Log results
            with open("checkpoints/loss.txt", "a") as f:
                f.write(f"Epoch [{epoch+1}/{args.epochs}]: ")
                f.write(f"Training Loss: {train_loss:.6f}, ")
                f.write(f"Validation Loss: {val_loss:.6f}, ")
                f.write(f"Test Loss: {test_loss:.6f}, ")
                f.write(f"Test Top-1 Accuracy: {test_acc:.6f}\n")
                
            wandb.log({
                "training_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "test_top1_accuracy": test_acc
            })
            
            print(f"Epoch [{epoch+1}/{args.epochs}]: "
                  f"Training Loss: {train_loss:.6f}, "
                  f"Validation Loss: {val_loss:.6f}, "
                  f"Test Loss: {test_loss:.6f}, "
                  f"Test Top-1 Accuracy: {test_acc:.6f}")
            torch.cuda.empty_cache()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
    except Exception as e:
        print(f"\nError occurred during training: {str(e)}")
        raise
    finally:
        # Save final checkpoint
        try:
            final_checkpoint_path = os.path.join("checkpoints", "final_model.pth")
            save_checkpoint(model, optimizer, scheduler, epoch+1, train_loss)
            print(f"Final checkpoint saved at {final_checkpoint_path}")
        except Exception as e:
            print(f"Error saving final checkpoint: {str(e)}")
    
