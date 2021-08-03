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
import transforms as T

def main():    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load("../Models/model.t")
    
    dataset_test = m.Dataset(m.get_transform(train=False))
    indices = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    img, _ = dataset_test[1]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])



    img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    mask = prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()
    print(prediction)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.")
    print(prediction[0])
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.")
    print(prediction[0]['masks'].shape)
    for i in range(prediction[0]['masks'].shape[0]):
        mask = np.add(mask, prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
        #mask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
        #print(mask)
        #print("------------------------------------------------")
        #break
        

    #print(mask)
    img_mask = Image.fromarray(np.clip(mask * 255, 0, 255))
    #img_mask = Image.fromarray(mask)
    img.show()
    img_mask.show()

if __name__ == "__main__":
    main()