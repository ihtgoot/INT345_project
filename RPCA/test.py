import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# add build directory
sys.path.append("./build")

import rpca_module


img1 = cv2.imread("0011.png", cv2.IMREAD_COLOR)
img2 = cv2.imread("0012.png", cv2.IMREAD_COLOR)
img3 = cv2.imread("0013.png", cv2.IMREAD_COLOR)


if img1 is None:
    print("img1 failed")
    exit()

if img2 is None:
    print("img2 failed")
    exit()

if img3 is None:
    print("img3 failed")
    exit()


H, W, C = img1.shape

print("Image shape:", H, W, C)


v1 = img1.flatten()
v2 = img2.flatten()
v3 = img3.flatten()


D = np.stack([v1, v2, v3], axis=1)

# convert to float64
D = D.astype(np.float64)

print("D shape:", D.shape)


results = rpca_module.process_batch([D])


L = results[0]

print("L shape:", L.shape)


clean = L[:, 0]

clean_img = clean.reshape(H, W, C)


clean_img = np.clip(clean_img, 0, 255)

clean_img = clean_img.astype(np.uint8)

cv2.imwrite("clean_color.png", clean_img)

print("saved clean_color.png")

# Load original clean image
img_orig = cv2.imread("0001.png", cv2.IMREAD_COLOR)

if img_orig is not None:
    # Convert BGR to RGB for matplotlib
    img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    clean_img_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)

    # Plot images
    plt.figure(figsize=(20, 5))
    titles = ["Original Clean", "Distorted 1", "Distorted 2", "Distorted 3", "Recovered (New Clean)"]
    images = [img_orig_rgb, img1_rgb, img2_rgb, img3_rgb, clean_img_rgb]

    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.title(titles[i])
        plt.imshow(images[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("comparison.png", bbox_inches='tight')
    print("saved comparison.png")
else:
    print("Could not load original image 0001.png for comparison")