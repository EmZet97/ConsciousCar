import json
import os
import numpy as np
import cv2 as cv

cur_path = os.path.dirname(__file__)
mask_path = "..\\RawData\\Masks"
labels_path = "..\\RawData\\Labels"

obj_name = "Road"
others_name = "Other"

labels = ["other",
     "road", "sidewalk", "parking", "rail track",
     "person", "rider", 
     "car", "truck", "bus", "on rails", "motorcycle", "bicycle", "caravan", "trailer",
     "building", "wall", "fence", "guard rail", "bridge", "tunnel",
     "pole", "pole group", "traffic sign", "traffic light",
     "vegetation", "terrain",
     "sky",
     "ground", "dynamic", "static"]

def load_labels():
    label_files = list(sorted(os.listdir(labels_path)))

    for label_name in label_files:
        print("Opened label file:", label_name)

        label_path = labels_path + "\\" + label_name
        f = open(label_path,)

        data = json.load(f)
        data["label_name"] = label_name
        data["image_name"] = label_name.replace("_gtFine_polygons.json", "_leftImg8bit.png")
        
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
        img_x = label["imgHeight"]
        img_y = label["imgWidth"]
        img_name = label["image_name"]

        img_mask = np.zeros((img_x, img_y, 1), np.uint8)

        for shape in label["objects"]:
            label_name = shape["label"]
            points = shape["polygon"]

            if label_name in labels:
                draw_polygon(img_mask, points, labels.index(label_name))
        
        save_image_mask(img_mask, img_name)


if __name__ == "__main__":
    generate_png_masks()