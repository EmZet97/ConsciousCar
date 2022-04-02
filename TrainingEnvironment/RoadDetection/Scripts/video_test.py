import cv2
import numpy as np
import os
import numpy as np
import torch
from PIL import Image
import train as m

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import torchvision.transforms as T

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_model(src):
    model = torch.load(src)

    return model


def predict(model, image, img_size):
    with torch.no_grad():
        #print(image)
        prediction = model([image.to(device)])
        mask = np.zeros(img_size, dtype=int)
        print(">", prediction[0]['masks'].shape[0])
        for i in range(prediction[0]['masks'].shape[0]):
            print("> >", prediction[0]['masks'][i])
            cmask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
            mask = np.add(mask, cmask)

            if i%1==0 and i == 0:
                break

        return np.clip(mask, 0, 255).astype('uint8')

def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def capture_frame(src):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(src)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            yield frame

        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def main():
    model = load_model("../Models/model.t")
    model.eval()

    for frame in capture_frame("../Video/video.mp4"):
        frame = cv2.resize(frame, (600, 600))
        cv2.imshow('Frame', frame)
        frame = Image.fromarray(frame).convert('RGB')
        transforms = get_transform()
        frame = transforms(frame)
        print("frame[0]:", frame[0].shape)
        
        print(frame.shape)
        mask = predict(model, frame, (600, 600))
        img_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        cv2.imshow('Mask',mask)
        cv2.waitKey(1)
        


main()