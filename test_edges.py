from gammanet.models import VGG16GammaNetV2
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Paths
# -----------------------------
input_dir = "/home/yentl/pytorch_gammanet/Images"
output_dir = "/home/yentl/pytorch_gammanet/Output_Images"
checkpoint_path = "/home/yentl/pytorch_gammanet/checkpoint_epoch_40.pt"

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load trained model
# -----------------------------
#checkpoint = torch.load(checkpoint_path, map_location="cpu")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = checkpoint['config']['model']

model = VGG16GammaNetV2(config)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
# model.eval()
# model.cuda()  # Gebruik GPU als beschikbaar
model.to(device)
model.eval()

print(f"Loaded GammaNet model from checkpoint (epoch {checkpoint['epoch']})")

# -----------------------------
# Model config
# -----------------------------
# config = {
#     'layers': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2',
#                'conv3_1', 'conv3_2', 'conv3_3', 'pool3'],
#     'kernel_size': 3,
#     'hidden_channels': 48,
#     'num_timesteps': 8,
#     'fgru': {
#         'use_separate_ei_states': True,
#         'use_symmetric_conv': True
#     }
# }

# model = VGG16GammaNetV2(config)
# model.eval()

# -----------------------------
# Image transform
# -----------------------------
transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
])

# -----------------------------
# Loop over images
# -----------------------------
for image_name in os.listdir(input_dir):

    if not image_name.lower().endswith((".png",".jpg",".jpeg",".bmp",".tiff")):
        continue

    input_path = os.path.join(input_dir, image_name)
    base_name = os.path.splitext(image_name)[0]

    print(f"Processing {image_name}")

    # Create output folder per image
    image_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Load image
    image = Image.open(input_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run model
    with torch.no_grad():
        edge_logits = model(input_tensor)
        edge_prob = torch.sigmoid(edge_logits)

    edge_map = edge_prob[0,0].cpu()
    edge_np = edge_map.numpy()

    # Print statistics
    print("Edge statistics:")
    print("Min:", edge_np.min())
    print("Max:", edge_np.max())
    print("Mean:", edge_np.mean())
    print("Std:", edge_np.std())

    #Edge map
    threshold = 0.5
    binary_edges = edge_np > threshold

    edge_pixels = binary_edges.sum()
    total_pixels = binary_edges.size
    edge_ratio = edge_pixels / total_pixels
    print(f"Threshold {threshold:.1f} → edge ratio {edge_ratio:.4f}")

    if edge_ratio > 0.05:
        print("Contour detected")
    else:
        print("No clear contour")

    # Save single threshold map
    threshold_output_path = os.path.join(image_output_dir, f"edges_threshold_{threshold:.1f}.png")
    plt.figure(figsize=(6,6))
    plt.imshow(binary_edges, cmap="gray")
    plt.title(f"Edges @ {threshold:.1f}")
    plt.axis("off")
    plt.savefig(threshold_output_path)
    plt.close()

    # Threshold sweep 0.1 → 1.0
    thresholds = np.arange(0.1, 1.1, 0.1)
    edge_ratios = []

    n_cols = 5
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))

    for idx, t in enumerate(thresholds):
        binary_edges_t = edge_np > t
        ratio = binary_edges_t.sum() / binary_edges_t.size
        edge_ratios.append(ratio)

        row = idx // n_cols
        col = idx % n_cols

        axes[row, col].imshow(binary_edges_t.astype(float), cmap="gray", vmin=0, vmax=1)
        axes[row, col].set_title(f"t={t:.1f}\nratio={ratio:.2f}", fontsize=10)
        axes[row, col].axis("off")

    plt.tight_layout()
    threshold_overview_path = os.path.join(image_output_dir, "threshold_overview.png")
    plt.savefig(threshold_overview_path)
    plt.close()

    print(f"Saved threshold overview: {threshold_overview_path}")

    # Save threshold curve
    curve_path = os.path.join(image_output_dir, "threshold_curve.png")
    plt.figure(figsize=(6,4))
    plt.plot(thresholds, edge_ratios, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Edge Ratio")
    plt.title(f"Edge Detection Curve ({base_name})")
    plt.grid(True)
    plt.savefig(curve_path)
    plt.close()

    print(f"Saved results to {image_output_dir}")

    # Plot input + edge map
    overview_path = os.path.join(image_output_dir, f"edges_overview_{base_name}.png")
    fig, axes = plt.subplots(1,2,figsize=(10,5))

    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(edge_map, cmap="gray")
    axes[1].set_title("Edges")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(overview_path)
    plt.close()

    print(f"Saved: {overview_path}")

# print("Saved edge detection result to edges_output.png")
print("Finished processing all images.")