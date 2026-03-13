import cv2
import numpy as np
import time
import os
import glob

def compute_otsu_threshold(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    total_pixels = image.size
    
    sum_all = np.dot(np.arange(256), hist)
    sum_background = 0
    weight_background = 0
    maximum_variance = 0
    optimal_threshold = 0
    
    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
            
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
            
        sum_background += i * hist[i]
        
        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground
        
        var_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if var_between > maximum_variance:
            maximum_variance = var_between
            optimal_threshold = i
            
    return optimal_threshold

def apply_morphology_closing(binary_img):
    # Setup padding for 3x3 structuring element
    padded = np.pad(binary_img, 1, mode='constant', constant_values=0)
    
    # Placeholder: just return the unpadded image for now
    return binary_img

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
        
    threshold_val = compute_otsu_threshold(img)
    binary_img = np.where(img < threshold_val, 1, 0).astype(np.uint8)
    
    # Call morphology function
    cleaned_img = apply_morphology_closing(binary_img)
        
    cv2.imshow(f"Cleaned Image - {os.path.basename(image_path)}", cleaned_img * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_files = glob.glob("*.jpeg") + glob.glob("*.jpg") + glob.glob("*.png")
    
    if not image_files:
        print("No images found in the current directory.")
    else:
        for img_path in image_files:
            if "input_file" in img_path and int(img_path.split("_")[-1].split(".")[0]) > 16:
                continue
            process_image(img_path)