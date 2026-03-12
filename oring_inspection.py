import cv2
import numpy as np
import time
import os
import glob

def compute_otsu_threshold(image):
    # Calculate histogram manually using numpy
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    total_pixels = image.size
    
    # Placeholder for threshold
    optimal_threshold = 128 
    return optimal_threshold

def process_image(image_path):
    # Read image in greyscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
        
    threshold_val = compute_otsu_threshold(img)
    print(f"Calculated threshold for {os.path.basename(image_path)}: {threshold_val}")
        
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