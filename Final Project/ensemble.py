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
        super(Ensemble, self).__init__()
        self.model1 = models.resnet18(pretrained=True)
        weight = self.model1.conv1.weight.clone()
        self.model1.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        with torch.no_grad(): 
            self.model1.conv1.weight[:, :3] = weight
            self.model1.conv1.weight[:, 3] = (weight[:,0] + weight[:,1] + weight[:,2]) / 3                                            
            self.model1.conv1.weight[:, 4] = (weight[:,0] + weight[:,1] + weight[:,2]) / 3
        num_ftrs = self.model1.fc.in_features
        self.model1.fc = nn.Linear(num_ftrs, num_classes)
        
        self.model2 = models.squeezenet1_1(pretrained=True) 
        weight = self.model2.features[0].weight.clone()
        self.model2.features[0] = nn.Conv2d(5, 64, kernel_size=3, stride=2)
        with torch.no_grad(): 
            self.model2.features[0].weight[:, :3] = weight
            self.model2.features[0].weight[:, 3] = (weight[:,0] + weight[:,1] + weight[:,2]) / 3
            self.model2.features[0].weight[:, 4] = (weight[:,0] + weight[:,1] + weight[:,2]) / 3
        self.model2.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        self.model2.num_classes = num_classes

        self.model3 = models.densenet121(pretrained=True)   
        weight = self.model3.features.conv0.weight.clone()  
        self.model3.features.conv0 = nn.Conv2d(5, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)
        with torch.no_grad(): 
            self.model3.features.conv0.weight[:, :3] = weight
            self.model3.features.conv0.weight[:, 3] = (weight[:,0] + weight[:,1] + weight[:,2]) / 3
            self.model3.features.conv0.weight[:, 4] = (weight[:,0] + weight[:,1] + weight[:,2]) / 3
        num_ftrs =  self.model3.classifier.in_features
        self.model3.classifier = nn.Linear(num_ftrs, num_classes)
        
        self.output = nn.Sequential(
            nn.Linear(num_classes * 3, num_classes),
            nn.Softmax(dim = 1)
        )
        
    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x = torch.cat((torch.cat((x1,x2),1),x3),1)
        x = self.output(x)
        return x
        
class EnsembleWrapper:
    def __init__(self, path, name, num_classes, num_epochs, batch, pre=True):
        self.model = Ensemble(num_classes, pre)
        self.path = os.path.join(path, name)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
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
            train_history.append(self.train_epoch(Xtrain, Ytrain))
            Logger.log(f"{epoch} epoch - train loss {train_history[-1]:.8f}")
            
            if not epoch % (self.freq_for_save):
                val_history.append(self.validate_epoch(Xval, Yval))
                Logger.log(f"{epoch} epoch - val loss {val_history[-1]:.8f}")
                self.loss = val_history[-1]
                if epoch >= 25 and np.sum(val_history[-2:]) >= np.sum(val_history[-4:-2]):
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
        X = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            X.requires_grad = False
            pred = self.model.forward(X).detach().cpu().numpy()
            result = np.argmax(pred, axis=1)
            return result
        
    def train_epoch(self, Xtrain, Ytrain):
        """
        This function trains one epoch of the model. 
        """
        self.model.train()
        history = []
        size = int(Xtrain.shape[0] / self.batch)
        for i in range(self.batch):
            X = Xtrain[i*size: (i+1)*size,:,:,:]
            Y = Ytrain[i*size: (i+1)*size]
            self.optmz.zero_grad()
            X = torch.from_numpy(X).float().to(self.device)
            Y = torch.from_numpy(Y).long().to(self.device)
            X.requires_grad = True
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            loss.backward()
            self.optmz.step()
            history.append(loss.detach().cpu().numpy())
            
        return np.sum(np.array(history)) / self.batch
    
    def validate_epoch(self, Xval, Yval):
        """
        This function validates one epoch of the model.
        """
        self.model.eval()
        history = []
        acc = 0
        size = int(Xval.shape[0] / self.batch)
        for i in range(self.batch):
            X = Xval[i*size: (i+1)*size,:,:,:]
            Y = Yval[i*size: (i+1)*size]        
            X = torch.from_numpy(X).float().to(self.device)
            Y = torch.from_numpy(Y).long().to(self.device)
            with torch.no_grad():
                self.optmz.zero_grad()
                X.requires_grad = False
                pred = self.model(X)
                loss = self.criterion(pred, Y)
                history.append(loss.detach().cpu().numpy())
                result = np.argmax(pred.detach().cpu().numpy(), axis=1)
                diff = result - Y.detach().cpu().numpy()
                acc += (diff.shape[0] - np.count_nonzero(diff)) / diff.shape[0]
        acc /= self.batch
        Logger.log(f"val acc {acc:.8f}")

        return np.sum(np.array(history)) / self.batch

        
    def save(self):
        if self.best_loss > self.loss:
            self.best_loss = self.loss
            torch.save(self.model.state_dict(), self.path)
        
    def load(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()