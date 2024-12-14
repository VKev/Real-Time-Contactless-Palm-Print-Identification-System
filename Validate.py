import torch
import os
import argparse
from model import MyModel
from torch.utils.data import DataLoader
from util import ImageDataset
from util import transform
from util import get_image_paths
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training parameters for the model.')
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/checkpoint_epoch_27.pth", help='Path to checkpoint file for continuing training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda'], help='Device to use for training (cpu or cuda)')
    return parser.parse_args()


def run_inference(image_paths, batch_size=16, device ="cuda"):
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_outputs = []
    progress_bar = tqdm(total=len(dataloader), desc="Processing Batches", unit="batch")

    with torch.no_grad():
        for input_tensor in dataloader:
            input_tensor = input_tensor.to(device)
            batch_outputs = model.forward(input_tensor)
            all_outputs.append(batch_outputs)
            progress_bar.update(1)
    progress_bar.close()

    return torch.cat(all_outputs, dim=0)

def extract_class(image_path):
    filename = os.path.basename(image_path)
    filename_without_ext = os.path.splitext(filename)[0]
    return filename_without_ext.split("_")[-1]

def l2_normalize(embeddings):
    l2_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    normalized_embeddings = embeddings / l2_norm
    return normalized_embeddings

if __name__ == "__main__":
    args = parse_args()
    checkpoint = torch.load(args.checkpoint_path)
    model = MyModel().to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    train_images_path = get_image_paths(r"raw/test-train")
    image_classes = [extract_class(p) for p in train_images_path]

    vectors = run_inference(train_images_path, batch_size=16 , device=args.device)
    


    vectors_np = vectors.cpu().numpy()
    distances = euclidean_distances(vectors_np)
    closest_indices = distances.argsort(axis=1)[:, 1]
    score = 0
    for i in range(len(image_classes)):
        if image_classes[i] == image_classes[closest_indices[i]]:
            score += 1

    accuracy = score / len(image_classes)

    print(f"Top 1 Accuracy: {accuracy:.4f}")


