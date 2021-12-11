"""
@file rcnn.py
@author Chang Yan (cyan13@jhu.edu)
"""

import torchvision
import torch
import os

class Rcnn:
    
    def __init__(self, path, name, pre = True):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pre)
        self.path = os.path.join(path, name)
        self.device = torch.device("cuda:0" if torch.cuda.is_abailable() else "cpu")
        self.model.to(self.device)
        
    def train(self, images, boxes):
        labels = torch.ones((images.shape[0], 1)).to(self.device)
        images = torch.from_numpy(images).to(self.device)
        boxes = torch.from_numpy(boxes.reshape((boxes.shape[0], 1, 4))).to(self.device)
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
        images = torch.from_numpy(images).to(self.device)
        images = list(image for image in images)
        self.model.eval()
        predictions = self.model(images)
        boxes = []
        scores = []
        for case in predictions:
            best_i = 0
            best_scores = torch.zeros(1)
            for i in range(case['scores'].shape[0]):
                if case['scores'][i] > best_scores:
                    best_scores = case['scores'][i]
                    best_i = i
            boxes.append(case['boxes'][best_i].detach().cpu().numpy())
            scores.append(best_scores.detach().cpu().numpy())
        return boxes, scores
    
    def save(self):
        torch.save(self.model.state_dict(), self.path)
        
    def load(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()
        