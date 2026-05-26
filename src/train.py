import torch, models, dataset, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = '/home/ihtgoot/CV/DATASET/data_root/train'
BATCH_SIZE = 4             # Kept low to fit 512x512 in VRAM
ACCUMULATION_STEPS = 8     # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS = 16
EPOCHS = 100

# Data Loading
transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((512, 512)), 
    transforms.ToTensor()
])
ds = dataset.DeblurDataset(DATA_PATH, transform=transform)
train_ds, val_ds = random_split(ds, [0.75, 0.25])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# Model & Optimization
net = models.FastResCNN().to(device)
opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
loss_fn = torch.nn.L1Loss()

train_losses, val_losses = [], []

print(f"Starting training on {len(ds)} images at 512x512 resolution...")

for epoch in range(EPOCHS):
    net.train()
    running_train = 0.0
    
    # Training Loop with Gradient Accumulation
    for i, (inp, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        inp, target = inp.to(device), target.to(device)
        
        # Forward pass
        loss = loss_fn(net(inp), target) / ACCUMULATION_STEPS
        loss.backward()
        
        # Accumulate gradients and update
        if (i + 1) % ACCUMULATION_STEPS == 0:
            opt.step()
            opt.zero_grad()
            
        running_train += loss.item() * ACCUMULATION_STEPS
    
    avg_train = running_train / len(train_loader)
    
    # Validation Loop
    net.eval()
    running_val = 0.0
    with torch.no_grad():
        for inp, target in val_loader:
            inp, target = inp.to(device), target.to(device)
            running_val += loss_fn(net(inp), target).item()
    
    avg_val = running_val / len(val_loader)
    train_losses.append(avg_train)
    val_losses.append(avg_val)
    print(f"Epoch {epoch+1} Complete | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

# Save results
plt.figure()
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.legend()
plt.savefig('loss_plot.png')
torch.save(net.state_dict(), 'best_model.pth')
print("Training finished. Model saved as best_model.pth")