import os, glob, cv2, torch
from torch.utils.data import Dataset

class DeblurDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.folders = sorted(glob.glob(os.path.join(root_dir, '*')))

    def __len__(self): return len(self.folders)

    def __getitem__(self, i):
        folder = self.folders[i]
        blurs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in sorted(glob.glob(os.path.join(folder, 'blur*.png')))[:3]]
        rpca = cv2.cvtColor(cv2.imread(glob.glob(os.path.join(folder, 'rpca*.png'))[0]), cv2.COLOR_BGR2RGB)
        clean = cv2.cvtColor(cv2.imread(glob.glob(os.path.join(folder, 'clean*.png'))[0]), cv2.COLOR_BGR2RGB)
        
        imgs = [self.transform(img) for img in (blurs + [rpca])]
        return torch.cat(imgs, dim=0), self.transform(clean)