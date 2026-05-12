"""
=========================================================
  Atmospheric Turbulence (Heat Haze) Dataset Generator
=========================================================
PURPOSE:
  This script takes a folder of clean images and applies
  realistic heat haze distortion to every image.
  
WHY:
  Machine Learning models need PAIRED data to learn.
  Paired = (distorted input image, clean ground truth image)
  This script creates the distorted version for you.

HOW TO RUN:
  python create_dataset.py --input clean_images/ --output distorted_images/
=========================================================
"""

import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm


# ---- HELPER FUNCTION ----
def make_smooth_noise(height, width, kernel_size):
    """
    Creates a smooth, cloudy noise field.
    Think of it as a 'heat map' showing where the hot air pockets are.
    
    - We start with random static (like TV static)
    - Then blur it heavily to make it smooth and cloud-like
    """
    # Step 1: Create random static noise
    noise = np.random.randn(height, width).astype(np.float32)
    
    # Step 2: Blur it heavily — this turns sharp static into smooth, rolling clouds
    #         The bigger the kernel, the wider the smooth blobs will be
    kernel = max(kernel_size, 5) | 1  # kernel must always be an odd number (OpenCV rule)
    smooth = cv2.GaussianBlur(noise, (kernel, kernel), 0)
    
    # Step 3: Divide by std so no matter what kernel we use, the output range is the same
    #         Without this, bigger kernels would give weaker distortions — not what we want!
    return smooth / np.std(smooth)


# ---- MAIN DISTORTION FUNCTION ----
def apply_heat_haze(image, frame_index, intensity, macro_scale, micro_scale, wave_mix):
    """
    Applies realistic heat haze to a single image.

    CONCEPT:
      Real heat haze works by bending light rays as they pass through hot air.
      We simulate this by:
        1. Multi-octave turbulence (3 layers of noise for organic look)
        2. Rising heat wave ripple (sine wave that moves upward like real hot air)
        3. Vertical gradient — stronger distortion near ground, fades toward sky
        4. Warping the image using the combined displacement map
        5. Chromatic aberration (color channel split — real optical effect)
    """
    height, width = image.shape[:2]

    # Each frame_index gives a different random seed
    np.random.seed(frame_index * 1337)

    # --- STEP 1: MULTI-OCTAVE TURBULENCE ---
    # We use 3 layers of noise (like layers of a painting):
    #   - macro  : big slow pockets of hot air   (like distant heat shimmer over a road)
    #   - medium : mid-size rolling waves         (like heat rising off a car roof)
    #   - micro  : tiny fast shimmering           (like looking over a lit stove)
    # Blending all 3 creates a far more organic, non-repetitive distortion
    medium_scale = (macro_scale + micro_scale) // 2  # automatically compute middle scale

    macro_noise_x  = make_smooth_noise(height, width, width // macro_scale)
    macro_noise_y  = make_smooth_noise(height, width, width // macro_scale)
    medium_noise_x = make_smooth_noise(height, width, width // medium_scale)
    medium_noise_y = make_smooth_noise(height, width, width // medium_scale)
    micro_noise_x  = make_smooth_noise(height, width, width // micro_scale)
    micro_noise_y  = make_smooth_noise(height, width, width // micro_scale)

    # Blend the 3 octaves: 60% macro + 25% medium + 15% micro
    shift_x = macro_noise_x * 0.60 + medium_noise_x * 0.25 + micro_noise_x * 0.15
    shift_y = macro_noise_y * 0.60 + medium_noise_y * 0.25 + micro_noise_y * 0.15

    # Scale by intensity (heat rises → vertical distortion is 1.5x stronger)
    shift_x = shift_x * intensity
    shift_y = shift_y * intensity * 1.5

    # --- STEP 2: RISING HEAT WAVE RIPPLE ---
    # Classic "wavy bands" visible above hot roads
    row_positions = np.arange(height)
    phase = frame_index * 0.3
    heat_wave = np.sin(row_positions * 0.05 + phase) * intensity * 3.0  # much stronger amplitude
    heat_wave = heat_wave[:, np.newaxis]
    shift_y = shift_y + heat_wave

    # --- STEP 3: VERTICAL GRADIENT FALLOFF ---
    # Softer falloff (exponent 0.6 instead of 1.5) so the WHOLE image is visibly distorted,
    # not just the bottom sliver
    vertical_fade = (row_positions / height) ** 0.6
    vertical_fade = np.clip(vertical_fade, 0.3, 1.0)  # minimum 30% distortion even at the top
    vertical_fade = vertical_fade[:, np.newaxis]
    shift_x = shift_x * vertical_fade
    shift_y = shift_y * vertical_fade

    # --- STEP 4: WARP THE IMAGE ---
    # Every pixel gets new coordinates based on the displacement map
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x + shift_x).astype(np.float32)
    map_y = (grid_y + shift_y).astype(np.float32)

    # Remap: physically move every pixel to its new distorted position
    distorted = cv2.remap(image, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

    # --- STEP 5: CHROMATIC ABERRATION ---
    # Real heat haze bends different wavelengths of light by different amounts.
    # Result: red and blue edges shift slightly — creating a color fringe.
    ca_strength = intensity * 0.4
    b, g, r = cv2.split(distorted)
    M_r = np.float32([[1, 0,  ca_strength], [0, 1,  ca_strength]])
    M_b = np.float32([[1, 0, -ca_strength], [0, 1, -ca_strength]])
    r = cv2.warpAffine(r, M_r, (width, height), borderMode=cv2.BORDER_REFLECT)
    b = cv2.warpAffine(b, M_b, (width, height), borderMode=cv2.BORDER_REFLECT)
    distorted = cv2.merge([b, g, r])

    return distorted



# ---- BATCH PROCESSING FUNCTION ----
def process_images(input_folder, output_folder, config):
    """
    Loops through all images in input_folder,
    applies heat haze to each one,
    and saves them to output_folder.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files from the folder (sorted so order is consistent)
    images = sorted([f for f in os.listdir(input_folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {len(images)} images. Starting distortion...")
    
    for frame_index, filename in enumerate(tqdm(images, desc="Distorting Images")):
        # Load the clean image
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            print(f"  Skipping (could not read): {filename}")
            continue
        
        # Apply heat haze distortion
        distorted_img = apply_heat_haze(img, frame_index, **config)
        
        # Save with 'distorted_' prefix so you know which is which
        save_path = os.path.join(output_folder, f"distorted_{filename}")
        cv2.imwrite(save_path, distorted_img)
    
    print(f"Done! All distorted images saved to: '{output_folder}'")


# =========================================================
# CONFIGURATION — Change these values to tune the output
# =========================================================
CONFIG = {
    "intensity"    : 3,    # Overall wave strength.          Low=3 (subtle) | High=15 (heavy distortion)
    "macro_scale"  : 8,    # Size of large air pockets.      Low=4 (wider waves) | High=16 (tighter)
    "micro_scale"  : 35,   # Size of tiny shimmers.          Low=10 (bigger shimmer) | High=60 (finer)
    "wave_mix"     : 0.7,  # Macro vs Micro ratio (0.0-1.0). 1.0=only big waves | 0.0=only shimmer
}
# =========================================================


# ---- ENTRY POINT ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Heat Haze distortion to a folder of clean images.")
    parser.add_argument("--input",  required=True, help="Path to folder containing clean images")
    parser.add_argument("--output", required=True, help="Path to folder where distorted images will be saved")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_images(args.input, args.output, CONFIG)
    else:
        print(f"Error: '{args.input}' is not a valid folder. Please provide a folder of images.")
