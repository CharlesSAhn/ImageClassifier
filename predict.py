import sys
import json
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image


class PredictImage:
    
    def __init__(self, image_path, checkpoint, category_names):
        self.image_path = image_path
        self.checkpoint = checkpoint
        self.model = None
        self.cat_to_name = category_names
      
    def loadCatToName(self):
        with open(category_names, 'r') as f:
            self.cat_to_name = json.load(f)
        
        print("------------------ Cat to Name file loaded ------------------------")
        
    def loadModel(self):
        self.model = models.vgg16(pretrained=True)

        self.model.classifier[6] = nn.Sequential(
                              nn.Linear(4096, 256),
                              nn.ReLU(),
                              nn.Dropout(p=0.4),
                              nn.Linear(256, 102),
                              nn.LogSoftmax(dim=1))


        checkpoint = torch.load(self.checkpoint)

        self.model.load_state_dict(checkpoint['state_dict'])
        
        print("------------------ checkpoint loaded ------------------------")
        
    def process_image(self):
        
        image = Image.open(self.image_path) 

        size_tuple = (256, int(667/500* 256))

        image.thumbnail(size_tuple)

        width, height = image.size
        l,t,r,b = (width - 224)/2 , (256 - 224)/2, (256 + 224)/2 , (256 + 224)/2
        image = image.crop((l, t, r, b))

        image = np.array(image)
        image = image /255

        image.transpose((2, 0, 1))

        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        image = np.transpose(image, (2, 0, 1))

        image = torch.from_numpy(image).float().to("cpu")
        
        print("------------------ image preprocessed ------------------------")

        return image
        
        
    def prediction(self, category, topk, mode):
        self.loadCatToName()
        self.loadModel()
        
        image = self.process_image()
        image = image.unsqueeze(0)
        
        device = "cpu"
        if mode == "gpu":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(device)
        image = image.to(device)
        
        with torch.no_grad():
            output = self.model.forward(image)
            
        ps = torch.exp(output)
        result = torch.topk(ps, int(topk))
        
        print("------------------ top predictions for {}------------------------".format(self.cat_to_name.get(str(category)))) 
        topLabels = []
        probabilities = result[0][0].cpu().numpy()
        counter = 0
        for x in result[1][0].cpu().numpy():
            topLabels.append(self.cat_to_name.get(str(x)))
            print("{}:  {}".format(self.cat_to_name.get(str(x)),  int(probabilities[counter]*10000)/100 ))
            counter = counter + 1
        return topLabels

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("image_path", type=str, help="iamge's path")
    parser.add_argument("checkpoint", type=str, help="checkpoint")
    parser.add_argument("--top_k", type=int, help="checkpoint")
    parser.add_argument("--category_names", type=str, help="json cat to name mapper")
    parser.add_argument("--gpu", action="store_true", help="str")
    
    args = parser.parse_args()
    
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    
    if checkpoint is None:
        checkpoint = "model_checkpoint.pth"
    if top_k is None:
        top_k = 5
    if category_names is None:
        category_names = "cat_to_name.json"
    
    mode = "cpu"
    if args.gpu: 
        mode = "gpu"
                  
    category = image_path.split("/")[-2]
    
    prediction = PredictImage(image_path, checkpoint, category_names)
    result = prediction.prediction(category, top_k, mode)
