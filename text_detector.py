import cv2
import numpy as np
from paddleocr import PaddleOCR
import os

def process_images(input_dir, output_dir):
    # Initialize PaddleOCR without angle classification and with Chinese language
    ocr = PaddleOCR(use_angle_cls=False, lang='ch', use_gpu=True, det_db_score_mode="slow", use_dilation=True)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Construct full file path
            image_path = os.path.join(input_dir, filename)
            
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read {image_path}")
                continue

            # Perform OCR to detect text regions
            result = ocr.ocr(image, cls=False)

            # Create a black image with the same dimensions as the original
            black_image = np.zeros_like(image)
            print(result)

            if result and result != [None]:
                # Fill detected text regions with white color
                for line in result:
                    for word_info in line:
                        bbox = word_info[0]
                        bbox = np.array(bbox).astype(np.int32)
                        cv2.fillPoly(black_image, [bbox], color=(255, 255, 255))

            # Save the result
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, black_image)
            print(f"Processed image saved to {output_path}")

# Example usage:
# process_images('path/to/input_folder', 'path/to/output_folder')
