import cv2
import numpy as np
import time
import os
import glob

def compute_otsu_threshold(image):
    """
    Calculates the optimal threshold using Otsu's method.
    (Histogram and Thresholding)
    """
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

def apply_morphology_closing(binary_img):
    """
    Performs a morphological Closing (Dilation followed by Erosion) to close interior holes.
    Added using fast numpy array slicing.
    (Binary Morphology)
    """
    # 1. Dilation (3x3 square structuring element)
    padded = np.pad(binary_img, 1, mode='constant', constant_values=0)
    dilated = np.maximum.reduce([
        padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
        padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
        padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]
    ])
    
    # 2. Erosion (3x3 square structuring element)
    padded_d = np.pad(dilated, 1, mode='constant', constant_values=1)
    closed = np.minimum.reduce([
        padded_d[:-2, :-2], padded_d[:-2, 1:-1], padded_d[:-2, 2:],
        padded_d[1:-1, :-2], padded_d[1:-1, 1:-1], padded_d[1:-1, 2:],
        padded_d[2:, :-2], padded_d[2:, 1:-1], padded_d[2:, 2:]
    ])
    
    return closed

def connected_component_labelling(binary_img):
    """
    Extracts regions using a Depth First Search (DFS) approach.
    (Connected Component Labelling)
    """
    labels = np.zeros_like(binary_img, dtype=np.int32)
    current_label = 1
    rows, cols = binary_img.shape
    
    for r in range(rows):
        for c in range(cols):
            if binary_img[r, c] == 1 and labels[r, c] == 0:
                # Start a new component
                stack = [(r, c)]
                labels[r, c] = current_label
                
                while stack:
                    curr_r, curr_c = stack.pop()
                    
                    # Check 4-connected neighbours
                    for dr, dc in[(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if binary_img[nr, nc] == 1 and labels[nr, nc] == 0:
                                labels[nr, nc] = current_label
                                stack.append((nr, nc))
                current_label += 1
                
    return labels, current_label - 1

def analyse_regions(binary_img):
    """
    Analyses the regions to classify the O-ring as a pass or fail.
    (Analyse regions)
    """
    # 1. Find the O-ring (largest foreground component)
    labels, num_labels = connected_component_labelling(binary_img)
    if num_labels == 0:
        return False
        
    counts = np.bincount(labels.flatten())
    counts[0] = 0 # Ignore the background (label 0)
    oring_label = np.argmax(counts)
    oring_mask = (labels == oring_label).astype(np.uint8)
    
    # 2. Fill the inner hole to create a solid circle for analysis
    # We do this by finding the background components. The outer background touches the edges.
    inv_mask = 1 - oring_mask
    bg_labels, _ = connected_component_labelling(inv_mask)
    outer_bg_label = bg_labels[0, 0] # Top-left pixel is guaranteed to be outer background
    
    # The filled O-ring is everything that is NOT the outer background
    filled_oring = (bg_labels != outer_bg_label).astype(np.uint8)
    
    # 3. Calculate properties
    actual_area = np.sum(filled_oring)
    
    # Get bounding box
    rows, cols = np.where(filled_oring == 1)
    if len(rows) == 0:
        return False
        
    height = np.max(rows) - np.min(rows)
    width = np.max(cols) - np.min(cols)
    
    # Calculate the ideal area of a perfect circle/ellipse fitting this bounding box
    ideal_area = np.pi * (width / 2.0) * (height / 2.0)
    
    # 4. Classification Logic
    # A perfect O-ring will have an actual area very close to the ideal bounding area.
    # Missing chunks (cuts) will lower the ratio. Extra material will distort the bounding box, also lowering the ratio.
    ratio = actual_area / ideal_area
    
    # Thresholds for passing 
    if 0.93 < ratio < 1.07:
        return True # Pass
    else:
        return False # Fail

def process_image(image_path):
    """
    Main pipeline handling the overall program structure.
    (Overall program structure, timing, and text annotation)
    """
    start_time = time.time()
    
    # Read image in greyscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
        
    # 1. Thresholding
    threshold_val = compute_otsu_threshold(img)
    # O-rings are dark on a light background, so we invert the binary condition
    binary_img = np.where(img < threshold_val, 1, 0).astype(np.uint8)
    
    # 2. Binary Morphology
    cleaned_img = apply_morphology_closing(binary_img)
    
    # 3 & 4. Connected Components and Analysis
    is_pass = analyse_regions(cleaned_img)
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000 # in milliseconds
    
    # Annotate output image
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    status_text = "PASS" if is_pass else "FAIL"
    colour = (0, 255, 0) if is_pass else (0, 0, 255) # Green for Pass, Red for Fail
    
    cv2.putText(out_img, f"Status: {status_text}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
    cv2.putText(out_img, f"Time: {processing_time:.1f} ms", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Display the result
    cv2.imshow(f"O-Ring Inspection - {os.path.basename(image_path)}", out_img)
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