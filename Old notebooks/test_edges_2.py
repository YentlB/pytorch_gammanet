from gammanet.models import VGG16GammaNetV2
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt


from gammanet.models import VGG16GammaNetV2
import torch

# 1. Create model with E/I populations (v2 architecture)
# config = {
#     'layers': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2',
#                'conv3_1', 'conv3_2', 'conv3_3', 'pool3'],
#     'kernel_size': 3,
#     'hidden_channels': 48,
#     'num_timesteps': 8,  # 8 recurrent processing steps
#     'fgru': {
#         'use_separate_ei_states': True,  # Separate E/I populations
#         'use_symmetric_conv': True       # Symmetric horizontal connections
#     }
# }

# model = VGG16GammaNetV2(config, )
# model.eval()

# Load trained model
checkpoint = torch.load('/home/yentl/pytorch_gammanet/checkpoint_epoch_40.pt', map_location='cpu', weights_only=False)
config = checkpoint['config']

model = VGG16GammaNetV2(config=config['model'])
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
#model.cuda()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Validation F1: {checkpoint.get('best_metric', 'N/A')}")

# 2. Load and preprocess image
image = Image.open('bC_3.png').convert('RGB')
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)

# 3. Run inference with recurrent processing
with torch.no_grad():
    edge_logits = model(input_tensor)
    edge_prob = torch.sigmoid(edge_logits)

# 4. Visualize results
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title('Input Image')
axes[0].axis('off')
axes[1].imshow(edge_prob[0, 0].cpu(), cmap='gray')
axes[1].set_title('Detected Edges')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('edges_output.png')
print("Saved edge detection result to edges_output.png")
 