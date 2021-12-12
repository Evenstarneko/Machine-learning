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
    
    def __init__(self, path, name, num_epochs, batch, pre=True):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pre)
        self.path = os.path.join(path, name)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.float()
        self.criterion = nn.CrossEntropyLoss()
        self.optmz = optim.Adam(self.model.parameters(), lr=1e-3)
        self.cur_epoch = 0
        self.epochs = num_epochs
        self.loss = np.infty
        self.best_loss = np.infty
        self.start_time = int(time.time())
        self.freq_for_save = 5
        self.batch = batch
        
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
            Logger.log(f"train loss {train_history[-1]:.8f}")
            
            if not epoch % (self.freq_for_save):
                val_history.append(self.val(Xval, Yval))
                Logger.log(f"{epoch} epoch - val loss {val_history[-1]:.8f}")
                self.loss = val_history[-1]
                if epoch >= 25 and np.sum(val_history[-2:]) > np.sum(val_history[-4:-2]):
                    Logger.log("early stop")
                    Logger.log(f"train history {train_history}")
                    Logger.log(f"validation history {val_history}")
                    
                    break

            if not (epoch % (self.freq_for_save)) or self.best_loss > self.loss:
                Logger.log("saving")
                self.save()
                Logger.log(f"{self.cur_epoch} epoch loss {self.loss:.8f} best loss {self.best_loss:.8f}")
    
    def train(self, images, boxes):
        self.model.train()
        labels = torch.ones((len(images), 1)).float().to(self.device)
        boxes = torch.from_numpy(boxes.reshape((boxes.shape[0], 1, 4))).float().to(self.device)
        images = list(torch.from_numpy(image).float().to(self.device) for image in images)
        targets = []
        for i in range(len(images)):
            d = {}
            d['boxes'] = boxes[i]
            d['labels'] = labels[i]
            targets.append(d)
            
        history = []
        size = int(len(images) / self.batch)
        for i in range(self.batch):
            self.optmz.zero_grad()
            X = images[i*size: (i+1)*size]
            Y = targets[i*size: (i+1)*size]
            output = self.model(X, Y)
            for loss in output.values():
                loss.backward() 
            self.optmz.step()
            loss = torch.sum(loss for loss in output.values()).detach().cpu().numpy()
            history.append(loss)
            
        return np.sum(np.array(history)) / self.batch
        
    def predict(self, images):
        images = list(torch.from_numpy(image).float().to(self.device) for image in images)
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
        self.model.train()

        with torch.no_grad():
            labels = torch.ones((images.shape[0], 1)).to(self.device)
            boxes = torch.from_numpy(boxes.reshape((boxes.shape[0], 1, 4))).float().to(self.device)
            images = list(torch.from_numpy(image).float().to(self.device) for image in images)
            targets = []
            for i in range(len(images)):
                d = {}
                d['boxes'] = boxes[i]
                d['labels'] = labels[i]
                targets.append(d)
                
            history = []
            size = int(len(images) / self.batch)
            for i in range(self.batch):
                self.optmz.zero_grad()
                X = images[i*size: (i+1)*size]
                Y = targets[i*size: (i+1)*size]
                output = self.model(X, Y)
                loss = torch.sum(loss for loss in output.values()).detach().cpu().numpy()
                history.append(loss)
        
        return np.sum(np.array(history)) / self.batch
    
    def save(self):
        if self.best_loss > self.loss:
            self.best_loss = self.loss
            torch.save(self.model.state_dict(), self.path)
        
    def load(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()
        