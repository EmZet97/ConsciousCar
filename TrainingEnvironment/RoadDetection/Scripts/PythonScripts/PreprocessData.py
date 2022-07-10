import json
import os
import numpy as np
import cv2 as cv

cur_path = os.path.dirname(__file__)

final_data_width = 600
final_data_height = 600


def preprocess_source_image(image):
    resized = cv.resize(image, (final_data_width, final_data_height))

    return resized

def preprocess_data(source_path, destination_path, color_mode):
    images_fin_path = os.path.relpath(source_path, cur_path)
    image_files = list(sorted(os.listdir(images_fin_path)))

    for image_name in image_files:
        print("Opened image file:", image_name)

        image_path = source_path + "\\" + image_name
        image = cv.imread(image_path, color_mode)

        image = preprocess_source_image(image)
        
        cv.imwrite(destination_path + "\\" + image_name, image)
        print("Saved preprocessed image:", image_name)

if __name__ == "__main__":
    src_images_path = "..\\RawData\\Images"
    dest_images_path = "..\\PreprocessedData\\Images"
    preprocess_data(src_images_path, dest_images_path, cv.IMREAD_UNCHANGED)

    src_masks_path = "..\\RawData\\Masks"
    dest_masks_path = "..\\PreprocessedData\\Masks"
    preprocess_data(src_masks_path, dest_masks_path, cv.IMREAD_GRAYSCALE)
