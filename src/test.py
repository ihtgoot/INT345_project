import torch, models, cv2, glob, os
from torchvision import transforms
from torchvision.utils import save_image

# 1. Setup - Match these exactly to your training transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((512, 512)), 
    transforms.ToTensor()
])

# 2. Load Model
net = models.FastResCNN().to(device)
# Ensure best_model.pth is in the same directory as this script
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()

# 3. Prepare Input Function
def load_and_stack(folder_path):
    # Sort files to ensure consistency: 3 blurs + 1 RPCA
    blurs = sorted(glob.glob(os.path.join(folder_path, 'blur*.png')))[:3]
    rpca_list = glob.glob(os.path.join(folder_path, 'rpca*.png'))
    
    if not blurs or not rpca_list:
        raise FileNotFoundError(f"Missing images in {folder_path}. Need 3 blur*.png and 1 rpca*.png")
    
    # Read images (OpenCV reads as BGR, convert to RGB)
    imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in blurs]
    imgs.append(cv2.cvtColor(cv2.imread(rpca_list[0]), cv2.COLOR_BGR2RGB))
    
    # Transform each and concatenate into a [1, 12, 512, 512] tensor
    tensors = [transform(img) for img in imgs]
    return torch.cat(tensors, dim=0).unsqueeze(0).to(device)

# 4. Inference
test_folder = '../test_data/0001' # Update as needed
input_tensor = load_and_stack(test_folder)

with torch.no_grad():
    output = net(input_tensor)
    # Clamp to [0, 1] to ensure valid image data
    output = torch.clamp(output, 0, 1)
    save_image(output, 'output_restored.png')
    print(f"Restored image saved successfully from {test_folder}")