import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as io
from models.get_model import get_model
from utils.dataset import load_mat_hsi
from utils.utils_HSI import count_sliding_window, sliding_window, grouper, metrics
import os

# Create outputs directory
os.makedirs('inference_outputs', exist_ok=True)

class Opts:
    pass

opts = Opts()
opts.model = 'BiDA_Agent'
opts.source_name = 'Houston13'
opts.target_name = 'Houston18'
opts.dataset_dir = './Houston/'
opts.patch_size = 13
opts.dim = 64
opts.depth = 3
opts.num_tokens = 4

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print('Loading dataset...')
img_tar, gt_tar, labels = load_mat_hsi(opts.target_name, opts.dataset_dir, norm='normband')
n_classes = len(labels)

checkpoint_dir = f'./checkpoints/{opts.model}/{opts.source_name}to{opts.target_name}'
model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
model_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
best_model_file = model_files[-1]
checkpoint_path = os.path.join(checkpoint_dir, best_model_file)

network = get_model(opts.model, opts.source_name, opts.patch_size, opts)
network.load_state_dict(torch.load(checkpoint_path, map_location=device))
network.to(device)
network.eval()

patch_size = opts.patch_size
batch_size = 128
window_size = (patch_size, patch_size)
image_w, image_h = img_tar.shape[:2]
pad_size = patch_size // 2

img_padded = np.pad(img_tar, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
probs = np.zeros(img_padded.shape[:2] + (n_classes, ))

iterations = count_sliding_window(img_padded, step=1, window_size=window_size) // batch_size
print('Running inference...')
for batch in grouper(batch_size, sliding_window(img_padded, step=1, window_size=window_size)):
    with torch.no_grad():
        data = [b[0] for b in batch]
        data = np.copy(data).transpose((0, 3, 1, 2))
        data = torch.from_numpy(data).float().unsqueeze(1).to(device)
        indices = [b[1:] for b in batch]
        
        output = network(data, data) 
        if isinstance(output, tuple):
            output = output[1]
            
        output = output.cpu().numpy()

        for (x, y, w, h), out in zip(indices, output):
            probs[x + w // 2, y + h // 2] += out

probs = probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]
pred_map = np.argmax(probs, axis=-1)

pred_map_masked = np.copy(pred_map)
pred_map_masked[gt_tar == -1] = -1

valid_mask = gt_tar >= 0
y_true = gt_tar[valid_mask]
y_pred = pred_map[valid_mask]

results = metrics(y_pred, y_true, n_classes=n_classes)

# Save Confusion Matrix
plt.figure(figsize=(10, 8))
cm = results['Confusion_matrix']
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f'Confusion Matrix (OA: {results["Accuracy"]:.2f}%)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('inference_outputs/learnable_agent_confusion_matrix.png', bbox_inches='tight', dpi=300)
plt.close()

# Save Classification Maps
cmap = plt.get_cmap('jet', n_classes + 1)
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.title('Predicted Classification Map')
plt.imshow(pred_map_masked, cmap=cmap, vmin=-1, vmax=n_classes-1)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Ground Truth Map')
plt.imshow(gt_tar, cmap=cmap, vmin=-1, vmax=n_classes-1)
plt.axis('off')

plt.tight_layout()
plt.savefig('inference_outputs/learnable_agent_classification_maps.png', bbox_inches='tight', dpi=300)
plt.close()

print('Outputs saved to inference_outputs/')
