import json
import os
import numpy as np
import torch
from PIL import Image
import cv2 as cv

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import sys
sys.path.append('vision/references/detection/')

from engine import train_one_epoch, evaluate
import utils
import transforms as T


images_path = "..\\RawData\\Images"
labels_path = "..\\RawData\\Labels"

train_eval_prop = 0.8
epochs = 500

label_names = ["Road", "RoadLine"]
label_to_remove = "Other"


class Dataset(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
        self.imgs = list(sorted(os.listdir(images_path)))
        self.labels = list(sorted(os.listdir(labels_path)))

    def generate_mask(self, mask, points, value):
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1,1,2))

        cv.fillPoly(mask,[pts],(value))

        return mask

    def read_label(self, label_name):        
        label_path = os.path.join(labels_path, label_name)
        label = open(label_path,)
        label = json.load(label)
        label["name"] = label_name

        return label

    def __getitem__(self, idx):
        img_path = os.path.join(images_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        label = self.read_label(self.labels[idx])

        image_height = label["imageHeight"]
        image_width = label["imageWidth"]

        shapes = [s for s in label["shapes"] if s["label"] in label_names]
        shapes_to_remove = [s for s in label["shapes"] if s["label"] not in label_names]

        masks = []
        boxes = []
        labels = []
        for shape in shapes:
            label_name = shape["label"]
            
            # generate mask
            points = shape["points"]
            mask = np.zeros((image_width, image_height), np.uint8)
            mask = self.generate_mask(mask, points, 1)

            for rm_shape in shapes_to_remove:
                rm_points = rm_shape["points"]
                mask = self.generate_mask(mask, rm_points, 0)

            mask = mask == 1
            # generate bounding boxes from mask
            pos = np.where(mask)
            print(">-1>>>>>>>>>>>>>>>>>>>>>>>", points, "<<<<<<<<<<<<<<<<<<<<<<<")
            print(">0>>>>>>>>>>>>>>>>>>>>>>>", label_name, "<<<<<<<<<<<<<<<<<<<<<<<")
            print(">1>>>>>>>>>>>>>>>>>>>>>>>", mask.shape, "<<<<<<<<<<<<<<<<<<<<<<<")
            print(">2>>>>>>>>>>>>>>>>>>>>>>>", mask, "<<<<<<<<<<<<<<<<<<<<<<<")
            print(">3>>>>>>>>>>>>>>>>>>>>>>>", pos, "<<<<<<<<<<<<<<<<<<<<<<<")
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            box = [xmin, ymin, xmax, ymax]

            boxes.append(box)
            masks.append(mask)
            labels.append(label_names.index(label_name) + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(shapes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    #fasterrcnn_resnet50_fpn
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def test_model(model, dataset_test):
    img, _ = dataset_test[0]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Cuda available: " + str(torch.cuda.is_available()))

    # our dataset has two classes only - background and road
    num_classes = 2
    # use our dataset and defined transformations
    dataset = Dataset(get_transform(train=True))
    dataset_test = Dataset(get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:int(len(indices) * train_eval_prop)])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[int(len(indices) * train_eval_prop):])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = epochs

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    model_name = f"../Models/model_e{num_epochs}_t{len(data_loader)}_e{len(data_loader_test)}.t"
    
    torch.save(model, model_name)
    print("Saved trained model in: ", model_name)
    
    
if __name__ == "__main__":
    main()
