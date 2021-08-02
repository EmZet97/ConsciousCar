import json
import os
import numpy as np
import cv2 as cv

cur_path = os.path.dirname(__file__)
mask_path = "..\\RawData\\Masks"
labels_path = "..\\RawData\\Labels"

obj_name = "Road"
others_name = "Other"

def load_labels():
    labels_fin_path = os.path.relpath(labels_path, cur_path)
    label_files = list(sorted(os.listdir(labels_path)))
    labels = []

    for label in label_files:
        print("Opened label file:", label)

        label_path = labels_path + "\\" + label
        f = open(label_path,)

        data = json.load(f)
        data["name"] = label
        
        yield data

def draw_polygon(image, points, color):
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1,1,2))

    cv.fillPoly(image,[pts],(color)) 

def save_image_mask(image, imageName):
    cv.imwrite(mask_path + "\\" + imageName, image)
    print("Saved generated image mask:", imageName)
    
def generate_png_masks():
    for label in load_labels():
        img_x = label["imageHeight"]
        img_y = label["imageWidth"]
        img_name = label["name"].split(".")[0]

        img_mask = np.zeros((img_x, img_y, 1), np.uint8)

        #i = 0
        for shape in label["shapes"]:
            label_name = shape["label"]
            points = shape["points"]

            draw_polygon(img_mask, points, 1 if label_name==obj_name else 0)            
            #i+=1
        
        save_image_mask(img_mask, img_name + ".png")


if __name__ == "__main__":
    generate_png_masks()