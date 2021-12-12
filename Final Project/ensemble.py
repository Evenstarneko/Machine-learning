"""
@file ensemble.py
@author Chang Yan (cyan13@jhu.edu)
"""
import torchvision
import torch
import os, time
from torchvision import datasets, models, transforms
from torch import nn
import torch.optim as optim
import numpy as np
from logger import Logger

class Ensemble(nn.Module):
    def __init__(self, num_classes, pre=True):
        self.model1 = models.resnet101(pre=True, num_classes = num_classes)
        weight = self.model1.cov1.weight.clone()
        self.model1.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.model1.conv1.weight[:, :3] = weight
        self.model1.conv1.weight[:, 3] = (self.model1.conv1.weight[:,0] 
                                          + self.model1.conv1.weight[:,1] + self.model1.conv1.weight[:,2]) / 3
        self.model1.conv1.weight[:, 4] = (self.model1.conv1.weight[:,0] 
                                          + self.model1.conv1.weight[:,1] + self.model1.conv1.weight[:,2]) / 3
        
        self.model2 = models.squeezenet1_1(pre=True, num_classes = num_classes) 
        weight = self.model2.features[0].weight.clone()
        self.model2.features[0] = nn.Conv2d(5, 64, kernel_size=3, stride=2)
        self.model2.features[0].weight[:, :3] = weight
        self.model2.features[0].weight[:, 3] = (self.model2.features[0].weight[:,0] 
                                          + self.model2.features[0].weight[:,1] + self.model2.features[0].weight[:,2]) / 3
        self.model2.features[0].weight[:, 4] = (self.model2.features[0].weight[:,0] 
                                          + self.model2.features[0].weight[:,1] + self.model2.features[0].weight[:,2]) / 3  

        self.model3 = models.densenet169(pre=True, num_classes = num_classes)   
        weight = self.model3.features['conv0'].weight.clone()  
        self.model3.features['conv0'] = nn.Conv2d(5, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)
        self.model3.features['conv0'].weight[:, :3] = weight
        self.model3.features['conv0'].weight[:, 3] = (self.model3.features['conv0'].weight[:,0] 
                                          + self.model3.features['conv0'].weight[:,1] + self.model3.features['conv0'].weight[:,2]) / 3
        self.model3.features['conv0'].weight[:, 4] = (self.model3.features['conv0'].weight[:,0] 
                                          + self.model3.features['conv0'].weight[:,1] + self.model3.features['conv0'].weight[:,2]) / 3
        
        self.output = nn.Linear(num_classes * 3, num_classes)
        
    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x = torch.cat((torch.cat((x1,x2),1),x3),1)
        x = self.output(x)
        return x
        
class EnsembleWrapper:
    def __init__(self, path, name, num_classes, num_epochs, pre=True):
        self.model = Ensemble(num_classes, pre)
        self.path = os.path.join(path, name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optmz = optim.Adam(self.model.parameters(), lr=1e-3)
        self.cur_epoch = 0
        self.epochs = num_epochs
        self.loss = np.infty
        self.best_loss = np.infty
        self.start_time = int(time.time())
        self.freq_for_save = 5
        self.num_classes = num_classes
    
    
    def train_val(self, Xtrain, Ytrain, Xval, Yval):
        """
        This function handles both train and validation.
        """
        val_history = []
        train_history = []

        Logger.log("will start training")
        
        for epoch in range(self.cur_epoch, self.epochs):
            self.cur_epoch = epoch
            train_history.append(self.train_epoch(Xtrain, Ytrain))
            Logger.log(f"train loss {train_history[-1]:.8f}")
            
            if not epoch % (self.freq_for_save):
                val_history.append(self.validate_epoch(Xval, Yval))
                Logger.log(f"{epoch} epoch - val loss {val_history[-1]:.8f}")
                self.loss = val_history[-1]
                if epoch >= 100 and np.sum(val_history[-7:]) > np.sum(val_history[-14:-7]):
                    Logger.log("early stop")
                    Logger.log(f"train history {train_history}")
                    Logger.log(f"validation history {val_history}")
                    
                    break

            if not (epoch % (self.freq_for_save)) or self.best_loss > self.loss:
                Logger.log("saving")
                self.save()
                Logger.log(f"{self.cur_epoch} epoch loss {self.loss:.8f} best loss {self.best_loss:.8f}")
        
    def predict(self, X):
        """
        This function predicts/inference a single instance
        """
        self.model.eval()
        with torch.no_grad():
            X.requires_grad = False
            pred = self.model.forward(X).detach().cpu().to_numpy()
            result = np.argmax(pred, axis=1)
            return result
        
    def train_epoch(self, Xtrain, Ytrain):
        """
        This function trains one epoch of the model. 
        """
        self.model.train()
        self.optmz.zero_grad()
        Xtrain = torch.from_numpy(Xtrain)
        Ytrain = torch.from_numpy(Ytrain)
        Xtrain.requires_grad = True
        pred = self.model(Xtrain)
        loss = self.criterion(pred, Ytrain)
        loss.backward()
        self.optmz.step()
            
        return float(loss)
    
    def validate_epoch(self, Xval, Yval):
        """
        This function validates one epoch of the model.
        """
        self.model.eval()

        with torch.no_grad():
            self.optmz.zero_grad()
            Xval.requires_grad = False
            pred = self.model(Xval)
            loss = self.criterion(pred.detach().cpu().to_numpy(), Yval)
            result = np.argmax(pred, axis=1)
            diff = result - Yval
            acc = (diff.shape[0] - np.count_nonzero(diff)) / diff.shape[0]
            Logger.log(f"val acc {acc:.8f}")

            return float(loss)

        
    def save(self):
        if self.best_loss > self.loss:
            self.best_loss = self.loss
            torch.save(self.model.state_dict(), self.path)
        
    def load(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()