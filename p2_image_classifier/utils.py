import torch
from PIL import Image
from torchvision import transforms

def save_checkpoint(model, classifier):
    model_state = {'model': model,
                   'classifier': classifier,
                   'state_dict': model.state_dict(),
                   'class_to_idx': model.class_to_idx}
    torch.save(model_state, 'checkpoint.pth')

def load_checkpoint(file='checkpoint.pth'):
    checkpoint = torch.load(file)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image):
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    pil_image = Image.open(image)
    in_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means, std)])
    pil_image = in_transforms(pil_image)

    return pil_image
