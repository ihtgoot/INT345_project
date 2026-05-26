import os
import glob
import cv2
import torch
import models

from torchvision import transforms
from torchvision.utils import save_image

# =========================================================
# 1. DEVICE SETUP
# =========================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

# =========================================================
# 2. IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# =========================================================
# 3. LOAD MODEL
# =========================================================

net = models.FastResCNN().to(device)

net.load_state_dict(
    torch.load(
        'best_model.pth',
        map_location=device
    )
)

net.eval()

print("Model loaded successfully.")

# =========================================================
# 4. IMAGE LOADER
# =========================================================

def load_image(path):

    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Could not load image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# =========================================================
# 5. PREPARE INPUT
# =========================================================

def load_and_stack(folder_path):

    print("\nSearching in folder:")
    print(os.path.abspath(folder_path))

    # ---------------------------------------------
    # Find blur images
    # ---------------------------------------------

    blur_pattern = os.path.join(folder_path, 'blur*.png')
    blur_files = sorted(glob.glob(blur_pattern))

    # ---------------------------------------------
    # Find RPCA image
    # ---------------------------------------------

    rpca_pattern = os.path.join(folder_path, 'rpca*.png')
    rpca_files = sorted(glob.glob(rpca_pattern))

    # ---------------------------------------------
    # Debug prints
    # ---------------------------------------------

    print("\nBlur files:")
    print(blur_files)

    print("\nRPCA files:")
    print(rpca_files)

    # ---------------------------------------------
    # Validation
    # ---------------------------------------------

    if len(blur_files) < 3:
        raise FileNotFoundError(
            f"Expected 3 blur images, found {len(blur_files)}"
        )

    if len(rpca_files) == 0:
        raise FileNotFoundError(
            "No RPCA image found"
        )

    # ---------------------------------------------
    # Use first 3 blur images
    # ---------------------------------------------

    blur_files = blur_files[:3]

    # Use first RPCA image
    rpca_file = rpca_files[0]

    # ---------------------------------------------
    # Load images
    # ---------------------------------------------

    imgs = []

    for f in blur_files:
        imgs.append(load_image(f))

    imgs.append(load_image(rpca_file))

    # ---------------------------------------------
    # Transform images to tensors
    # ---------------------------------------------

    tensors = []

    for img in imgs:
        tensors.append(transform(img))

    # ---------------------------------------------
    # Concatenate channels
    # Shape:
    # 4 images × 3 channels = 12 channels
    # Final:
    # [12, 512, 512]
    # ---------------------------------------------

    stacked_tensor = torch.cat(tensors, dim=0)

    print("\nStacked tensor shape:")
    print(stacked_tensor.shape)

    # Add batch dimension
    # Final shape:
    # [1, 12, 512, 512]

    stacked_tensor = stacked_tensor.unsqueeze(0)

    return stacked_tensor.to(device)

# =========================================================
# 6. TEST FOLDER
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

test_folder = os.path.join(
    BASE_DIR,
    'test_data',
    '0001'
)

print("\nTest folder:")
print(test_folder)

# =========================================================
# 7. LOAD INPUT
# =========================================================

input_tensor = load_and_stack(test_folder)

print("\nInput tensor shape:")
print(input_tensor.shape)

# =========================================================
# 8. INFERENCE
# =========================================================

with torch.no_grad():

    output = net(input_tensor)

# =========================================================
# 9. SAVE OUTPUT
# =========================================================

output_path = os.path.join(
    BASE_DIR,
    'output_restored.png'
)

save_image(output, output_path)

print(f"\nRestored image saved at:\n{output_path}")