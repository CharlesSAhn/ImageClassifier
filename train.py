import sys
import os 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict


class BuildModel:
    
    def __init__(self, directory=None):
        self.directory = directory
        self.train_dir = directory + '/train'
        self.valid_dir = directory + '/valid'
        self.test_dir = directory + '/test'
        self.dataloaders = None
        self.valid_dataloaders = None
        self.test_dataloaders = None
        self.model = None
        
    def load_data(self):
        print("------------------ loading data ------------------------")
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]) 

        test_trainsforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])]) 


        image_datasets = datasets.ImageFolder(self.train_dir, transform=train_transforms)
        valid_datasets = datasets.ImageFolder(self.valid_dir, transform=test_trainsforms)
        test_datasets = datasets.ImageFolder(self.test_dir, transform=test_trainsforms)


        self.dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
        self.valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
        self.test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64)
        
        print("------------------ loading data  completed ------------------------")
    
    def loadModel(self, model, hidden_units):
        
        if model == "vgg16":
            self.model = models.vgg16(pretrained=True)
        
        if model == "alexnet":
            self.model = models.alexnet(pretrained=True)
            
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.classifier[6] = nn.Sequential(
                          nn.Linear(4096, 256),
                          nn.ReLU(),
                          nn.Dropout(p=0.4),
                          nn.Linear(256, hidden_units),
                          nn.LogSoftmax(dim=1))
        
        print("------------------ model loaded & updated ------------------------")
    
    def trainModel(self, learning_rate, epochs, mode):
        criterion = nn.NLLLoss()

        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(self.model.classifier[6].parameters(), lr=learning_rate)
        
        device = "cpu"
        if mode == "gpu":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(device)
        

        steps = 0
        running_loss = 0
        print_every = 5
        for epoch in range(epochs):
            for inputs, labels in self.dataloaders:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = self.model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.valid_dataloaders:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = self.model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {test_loss/len(self.valid_dataloaders):.3f}.. "
                          f"Validation accuracy: {accuracy/len(self.valid_dataloaders):.3f}")
                    running_loss = 0
                    self.model.train()
    
    def saveModel(self, save_dir):
        checkpoint = {
            'state_dict': self.model.state_dict()
        }

        torch.save(checkpoint, os.path.join(save_dir, 'model_checkpoint.pth'))
        print("------------------ model saved ------------------------")
    
    def proces(self, model, save_dir, learning_rate, hidden_units, epochs, mode):
        self.load_data()
        self.loadModel(model, hidden_units)
        self.trainModel(learning_rate, epochs, mode)
        self.saveModel(save_dir)

if __name__ == "__main__":
    
   parser = argparse.ArgumentParser()

   parser.add_argument("data_directory", type=str, help="directory path for data")
   parser.add_argument("--save_dir", type=str, help="directory where the model will be saved")
   parser.add_argument("--arch", type=str, choices=["vgg16", "alexnet"], help="models")
   parser.add_argument("--learning_rate", type=float, help="learning rate")
   parser.add_argument("--hidden_units", type=int, help="hidden units")
   parser.add_argument("--epochs", type=int, help="epochs")
   parser.add_argument("--gpu", action="store_true", help="str")

   args = parser.parse_args()
   
   data_directory = args.data_directory
   save_dir = args.save_dir
   model = args.arch
   learning_rate = args.learning_rate
   hidden_units = args.hidden_units
   epochs = args.epochs
    
   if data_directory is None:
       data_directory = "flowers"     
   if save_dir is None:
       save_dir = os.getcwd()
   if model is None:
       model = "vgg16"
   if learning_rate is None:
       learning_rate = 0.001
   if hidden_units is None:
       hidden_units = 102
   if epochs is None:
       epochs = 1
     
   mode = "cpu"
   if args.gpu: 
       mode = "gpu"
  

   build = BuildModel(data_directory)
   build.proces(model, save_dir, learning_rate, hidden_units, epochs, mode)