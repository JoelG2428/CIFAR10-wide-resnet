"""Classify your own image with the fixed trained checkpoint."""
import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "models/final_model.pt")
    args = parser.parse_args()
    import torch
    from PIL import Image
    from torchvision import transforms
    sys.path.insert(0, str(Path(__file__).resolve().parent / "deadline_wrn"))
    from load_model import load_model
    torch.set_num_threads(2)
    model, norm, classes = load_model(args.checkpoint)
    with Image.open(args.image) as image:
        image = image.convert("RGB")
        if image.size != (32, 32):
            print("Resizing your image to 32 x 32; this is a custom-image demo, not the benchmark protocol.")
            image = image.resize((32, 32), Image.Resampling.BILINEAR)
        batch = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm["mean"], norm["std"])])(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(batch).softmax(dim=1)[0]
    values, indices = probabilities.topk(3)
    for value, index in zip(values.tolist(), indices.tolist()):
        print(f"{classes[index]:12s} {value:.2%}")
    print("Scores are softmax outputs, not calibrated confidence estimates.")

if __name__ == "__main__":
    main()
