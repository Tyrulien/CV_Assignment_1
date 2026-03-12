import cv2
import numpy as np
import time
import os
import glob

def process_image(image_path):
    # Read image in greyscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
        
    # Display the result temporarily to check it loads
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