import cv2
import numpy as np
import time
import os
import glob

def compute_otsu_threshold(image):
    # Calculate histogram manually using numpy
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
        
        # Calculate Between Class Variance
        var_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if var_between > maximum_variance:
            maximum_variance = var_between
            optimal_threshold = i
            
    return optimal_threshold

def process_image(image_path):
    # Read image in greyscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
        
    threshold_val = compute_otsu_threshold(img)
        
    # Display the result temporarily
    cv2.imshow(f"O-Ring Inspection - {os.path.basename(image_path)}", img)
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