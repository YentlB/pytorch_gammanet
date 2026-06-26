# Imports
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from gammanet.models import VGG16GammaNetV2

# Device setup
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
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

config = checkpoint['config']['model']

model = VGG16GammaNetV2(config)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

model.to(device)
model.eval()

print(f"Loaded GammaNet model from checkpoint (epoch {checkpoint['epoch']})")
print(f"Validation F1: {checkpoint.get('best_metric', 'N/A')}")

# -----------------------------
# Register hooks for feature maps
# (Conv layers + fGRU layers)
# -----------------------------
activations = {}

def get_activation(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]  # excitatory state
        activations[name] = output.detach()
    return hook

# Hook VGG convolution blocks
vgg_blocks = {
    "block1": model.block1_conv,
    "block2": model.block2_conv,
    "block3": model.block3_conv,
    "block4": model.block4_conv,
    "block5": model.block5_conv,
}

for block_name, block in vgg_blocks.items():
    for i, layer in enumerate(block.layers):
        layer_name = f"{block_name}_layer{i}"
        layer.register_forward_hook(get_activation(layer_name))

# Hook all fGRU layers
for i in range(5):
    fgru_layer = getattr(model, f"fgru_{i}")
    fgru_layer.register_forward_hook(get_activation(f"fgru_{i}"))

# -----------------------------
# Image preprocessing transform
# -----------------------------
transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -----------------------------
# Function: Save feature maps
# -----------------------------
def save_feature_maps(tensor, layer_name, output_folder, num_channels=6):

    fmap = tensor.squeeze(0)
    n = min(num_channels, fmap.shape[0])

    plt.figure(figsize=(12,4))

    for i in range(n):

        plt.subplot(1,n,i+1)
        plt.imshow(fmap[i].cpu(), cmap="viridis")
        plt.axis("off")

    plt.suptitle(f"{layer_name} (showing {n} channels)")
    plt.tight_layout()

    save_path = os.path.join(output_folder, f"featuremap_{layer_name}.png")

    plt.savefig(save_path)
    plt.close()

# -----------------------------
# Loop over all images
# -----------------------------
for image_name in os.listdir(input_dir):

    if not image_name.lower().endswith((".png",".jpg",".jpeg",".bmp",".tiff")):
        continue

    print(f"\nProcessing {image_name}")

    input_path = os.path.join(input_dir, image_name)
    base_name = os.path.splitext(image_name)[0]

    # -----------------------------
    # Create output folder per image
    # -----------------------------
    image_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(image_output_dir, exist_ok=True)

    featuremap_dir = os.path.join(image_output_dir, "featuremaps")
    os.makedirs(featuremap_dir, exist_ok=True)

    # -----------------------------
    # Load image
    # -----------------------------
    image = Image.open(input_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Clear previous activations
    activations.clear()

    # -----------------------------
    # Run model inference
    # -----------------------------
    with torch.no_grad():

        edge_logits = model(input_tensor)

        edge_prob = torch.sigmoid(edge_logits)

    edge_map = edge_prob[0,0].cpu()

    edge_np = edge_map.numpy()

    # -----------------------------
    # Print edge statistics
    # -----------------------------
    print("Edge statistics:")
    print("Min:", edge_np.min())
    print("Max:", edge_np.max())
    print("Mean:", edge_np.mean())
    print("Std:", edge_np.std())

    # -----------------------------
    # Binary edge detection
    # -----------------------------
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

    # -----------------------------
    # Save binary edge map
    # -----------------------------
    threshold_output_path = os.path.join(
        image_output_dir,
        f"edges_threshold_{threshold:.1f}.png"
    )

    plt.figure(figsize=(6,6))
    plt.imshow(binary_edges, cmap="gray")
    plt.title(f"Edges @ {threshold:.1f}")
    plt.axis("off")
    plt.savefig(threshold_output_path)
    plt.close()

    # -----------------------------
    # Threshold sweep (0.1 → 1.0)
    # -----------------------------
    thresholds = np.arange(0.1, 1.1, 0.1)

    edge_ratios = []

    n_cols = 5
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15,6))

    for idx, t in enumerate(thresholds):

        binary_edges_t = edge_np > t

        ratio = binary_edges_t.sum() / binary_edges_t.size

        edge_ratios.append(ratio)

        row = idx // n_cols
        col = idx % n_cols

        axes[row,col].imshow(binary_edges_t.astype(float), cmap="gray", vmin=0, vmax=1)
        axes[row,col].set_title(f"t={t:.1f}\nratio={ratio:.2f}", fontsize=10)
        axes[row,col].axis("off")

    plt.tight_layout()

    threshold_overview_path = os.path.join(
        image_output_dir,
        "threshold_overview.png"
    )

    plt.savefig(threshold_overview_path)
    plt.close()
    print(f"Saved threshold overview: {threshold_overview_path}")

    # -----------------------------
    # Threshold curve plot
    # -----------------------------
    curve_path = os.path.join(
        image_output_dir,
        "threshold_curve.png"
    )

    plt.figure(figsize=(6,4))
    plt.plot(thresholds, edge_ratios, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Edge Ratio")
    plt.title(f"Edge Detection Curve ({base_name})")
    plt.grid(True)
    plt.savefig(curve_path)
    plt.close()

    # -----------------------------
    # Save input + edge map overview
    # -----------------------------
    overview_path = os.path.join(
        image_output_dir,
        f"edges_overview_{base_name}.png"
    )

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

    # -----------------------------
    # Save feature maps
    # -----------------------------

    for name, fmap in activations.items():

        print(f"{name}: {list(fmap.shape)}")

        save_feature_maps(
            fmap,
            name,
            featuremap_dir
        )

    print(f"Saved feature maps to: {featuremap_dir}")

# -----------------------------
# Finished
# -----------------------------
print("\nFinished processing all images.")    