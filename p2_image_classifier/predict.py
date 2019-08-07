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

from utils import load_checkpoint, process_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='5')
    parser.add_argument('--filepath', dest='filepath', default=None)
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    # parser.add_argument('--gpu', action='store_true', default=True)
    return parser.parse_args()

def predict(image_path, model, topk):
    model.eval() # Turn off dropout
    model.cpu()  # No need to use gpu for predict

    # load image as torch.Tensor
    image = process_image(image_path)

    # Unsqueeze returns a new tensor with a dimension of size one
    # Sometimes, the dimensionality of the input is unknown when the operation is being used.
    # The nn.Unsqueeze is useful and lets us insert the dimension without explicitly being aware of the other dimensions when writing the code.
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, topk)
        top_prob = top_prob.exp()

    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()

    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])

    return top_prob.numpy()[0], mapped_classes

def main():
    args = parse_args()

    # gpu = args.gpu
    # device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    # print(device)

    chkp_model = load_checkpoint(args.checkpoint)
    print(chkp_model)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    if args.filepath == None:
        img_num = random.randint(1, 102)
        image = random.choice(os.listdir('./flowers/test/' + str(img_num) + '/'))
        img_path = './flowers/test/' + str(img_num) + '/' + image
    else:
        img_path = args.filepath
    print('Image selected: ' + img_path)

    top_prob, top_classes = predict(img_path, chkp_model, int(args.top_k))
    print(top_prob)
    print(top_classes)
    print([cat_to_name[x] for x in top_classes])

if __name__ == "__main__":
    main()
