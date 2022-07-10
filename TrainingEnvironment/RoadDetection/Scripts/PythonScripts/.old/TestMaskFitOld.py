import json
import os
import numpy as np
import cv2 as cv

cur_path = os.path.dirname(__file__)

def merge_images_and_masks(src_images_path, src_masks_path, dest_path):
    images_path = os.path.relpath(src_images_path, cur_path)
    masks_path = os.path.relpath(src_masks_path, cur_path)

    image_files = list(sorted(os.listdir(images_path)))
    mask_files = list(sorted(os.listdir(masks_path)))

    for image_name in image_files:
        print("Opened image file:", image_name)

        image_path = images_path + "\\" + image_name
        mask_path = masks_path + "\\" + image_name.split(".")[0] + ".png"

        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        image = cv.imread(image_path)

        masked = cv.bitwise_and(image, image, mask=mask)
        
        final_path = dest_path + "\\masked_" + image_name
        cv.imwrite(final_path, masked)
        print("Saved preprocessed image: ", final_path)

if __name__ == "__main__":
    src_images_path = "..\\PreprocessedData\\Images"
    src_masks_path = "..\\PreprocessedData\\Masks"
    dest_path = "..\\PreprocessedData\\MaskTest"
    merge_images_and_masks(src_images_path, src_masks_path, dest_path)
