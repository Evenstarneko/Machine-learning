"""
@file rcnn.py
@author Chang Yan (cyan13@jhu.edu)
"""

import torchvision
import torch
import os, time
import torch.optim as optim
import numpy as np
from logger import Logger
from torch import nn

class Rcnn:
    
    def __init__(self, path, name, num_epochs, pre=True):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pre, num_classes = 2)
        self.path = os.path.join(path, name)
        self.device = torch.device("cuda:0" if torch.cuda.is_abailable() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optmz = optim.Adam(self.model.parameters(), lr=1e-3)
        self.cur_epoch = 0
        self.epochs = num_epochs
        self.loss = np.infty
        self.best_loss = np.infty
        self.start_time = int(time.time())
        self.freq_for_save = 5
        
    def train_val(self, Xtrain, Ytrain, Xval, Yval):
        """
        This function handles both train and validation.
        """
        val_history = []
        train_history = []

        Logger.log("will start training")
        
        for epoch in range(self.cur_epoch, self.epochs):
            self.cur_epoch = epoch
            train_history.append(self.train(Xtrain, Ytrain))
            Logger.log(f"train loss {train_history[-1]}")
            
            if not epoch % (self.freq_for_save):
                val_history.append(self.val(Xval, Yval))
                Logger.log(f"{epoch} epoch - val loss {val_history[-1]}")
                self.loss = val_history[-1]
                if epoch >= 100 and np.sum(val_history[-7:]) > np.sum(val_history[-14:-7]):
                    Logger.log("early stop")
                    Logger.log(f"train history {train_history}")
                    Logger.log(f"validation history {val_history}")
                    
                    break

            if not (epoch % (self.freq_for_save)) or self.best_loss > self.loss:
                Logger.log("saving")
                self.save()
                Logger.log(f"{self.cur_epoch} epoch loss {self.loss} best loss {self.best_loss}")
    
    def train(self, images, boxes):
        self.model.train()
        self.optmz.zero_grad()
        labels = torch.ones((images.shape[0], 1)).to(self.device)
        boxes = torch.from_numpy(boxes.reshape((boxes.shape[0], 1, 4))).to(self.device)
        images = list(torch.from_numpy(image).to(self.device) for image in images)
        targets = []
        for i in range(len(images)):
            d = {}
            d['boxes'] = boxes[i]
            d['labels'] = labels[i]
            targets.append(d)
        output = self.model(images, targets)
        loss = torch.sum(loss for loss in output.values())
        loss.backward()
        self.optmz.step()
            
        return float(loss)
        
    def predict(self, images):
        images = list(torch.from_numpy(image).to(self.device) for image in images)
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
    
    def val(self, images, boxes):
        """
        This function validates one epoch of the model.
        """
        self.model.eval()

        with torch.no_grad():
            self.optmz.zero_grad()
            labels = torch.ones((images.shape[0], 1)).to(self.device)
            boxes = torch.from_numpy(boxes.reshape((boxes.shape[0], 1, 4))).to(self.device)
            images = list(torch.from_numpy(image).to(self.device) for image in images)
            targets = []
            for i in range(len(images)):
                d = {}
                d['boxes'] = boxes[i]
                d['labels'] = labels[i]
                targets.append(d)
            output = self.model(images, targets)
            loss = torch.sum(loss for loss in output.values())

            return float(loss)
    
    def save(self):
        torch.save(self.model.state_dict(), self.path)
        
    def load(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()
        