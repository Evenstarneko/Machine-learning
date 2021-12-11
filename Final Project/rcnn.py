"""
@file rcnn.py
@author Gary Yang (yyang117@jhu.edu)
"""

import torchvision
import torch
import os

class Rcnn:
    
    def __init__(self, path, name, pre = True):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pre)
        self.path = os.path.join(path, name) 
        
    def train(self, images, boxes):
        labels = torch.ones(images.shape[0])
        images = torch.from_numpy(images)
        boxes = torch.from_numpy(boxes)
        images = list(image for image in images)
        targets = []
        for i in range(len(images)):
            d = {}
            d['boxes'] = boxes[i]
            d['labels'] = labels[i]
            targets.append(d)
        output = self.model(images, targets)
        return output
        
    def predict(self, images):
        images = torch.from_numpy(images)
        images = list(image for image in images)
        self.model.eval()
        predictions = self.model(images)
        return predictions
    
    def save(self):
        torch.save(self.model.state_dict(), self.path)
        
    def load(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()
        