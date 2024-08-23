#In this file, we define a NN that has a .fit method() and a .predict_proba() method

import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset, random_split
from torch.utils.data import DataLoader


class NNClassifier(object):
    def __init__(self, model, batch_size=128, n_epochs=10, lr=1e-3):

        self.model = model #eg resnet152 = models.resnet152(pretrained=True)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.loss_fn = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.n_iter_no_change=10
        self.tol = 1e-4

    def fit(self, train_dataset, val_dataset=None):

        train_loader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=False)
        if val_dataset is not None: 
            val_loader = DataLoader(val_dataset,batch_size=self.batch_size,shuffle=False)

        #for early stopping
        best_val_acc = 0
        it_no_change = 0

        for _ in range(self.n_epochs):
            for x_batch, y_batch in train_loader:
                self.model.train()
                y_pred = self.model(x_batch)
                y_batch = y_batch.squeeze().long()
                loss = self.loss_fn(y_pred,y_batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            #must use early stopping to avoid overfitting: if validation accuracy has not improved by tol in n_iter_no_change steps, then stop
            if val_dataset is not None: 
                with torch.no_grad(): #after each epoch
                    val_acc =0
                    for x_val, y_val in val_loader:
                        self.model.eval()
                        yhat = self.model(x_val)
                        yhat = yhat.softmax(dim=-1) #probabilities 
                        y_val = y_val.squeeze().long()
                        acc = torch.sum(torch.argmax(yhat,dim=1) == y_val).item() #to check
                        val_acc += acc
                    val_acc /= len(val_loader.dataset)
                
                    if val_acc > best_val_acc + self.tol:
                        best_val_acc = val_acc
                        it_no_change =0 #reset 
                    else:
                        it_no_change+=1 #increment counter of number of iterations without val acc improvement 
                        if it_no_change > self.n_iter_no_change: #break if too much 
                            break 
        #print("val accuracy", val_acc)


    def predict_proba(self, test_dataset):
        test_loader = DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=False)

        with torch.no_grad():
            for x,y in test_loader: 
                self.model.eval()
                yhat_unnormed = self.model(x)
                yhat = yhat_unnormed.softmax(dim=-1) #probabilities 
                return yhat.numpy()
            




class ConvNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ConvNet, self).__init__()
        self.dim_target=num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x






