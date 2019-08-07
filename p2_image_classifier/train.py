import torch
import torch.nn.functional as F
import time
import json
import os, random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

from utils import save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Training process")
    parser.add_argument('data_dir')
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16', 'vgg13'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='2048')
    parser.add_argument('--epochs', dest='epochs', default='5')
    parser.add_argument('--gpu', action="store_true", default=True)
    return parser.parse_args()


def train(model, criterion, optimizer, train_loaders, valid_loaders, epochs, device):
    steps = 0
    running_loss = 0
    print_every = 50

    start=time.time()
    print("Training started ...")
    for epoch in range(epochs):
        for inputs, labels in train_loaders:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_loaders):.3f}.. "
                      f"Accuracy: {accuracy/len(valid_loaders):.3f}")
                running_loss = 0
                model.train()
    end = time.time()
    print("Training done in {} seconds!".format(round(end-start, 2)))

def main():
    args = parse_args()

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    train_loaders = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    valid_loaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    test_loaders = torch.utils.data.DataLoader(test_datasets, batch_size=32)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    output_size = len(cat_to_name)

    model = getattr(models, args.arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    hidden_units = int(args.hidden_units)
    feature_num = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(feature_num, 4096)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.2)),
                              ('fc2', nn.Linear(4096, hidden_units)),
                              ('relu2', nn.ReLU()),
                              ('dropout2', nn.Dropout(p=0.2)),
                              ('output', nn.Linear(hidden_units, output_size)),
                              ('softmax', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), float(args.learning_rate))

    gpu = args.gpu
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    print(model)
    epochs = int(args.epochs)

    train(model, criterion, optimizer, train_loaders, valid_loaders, epochs, device)

    model.class_to_idx = train_datasets.class_to_idx
    save_checkpoint(model, classifier)

if __name__ == "__main__":
    main()
