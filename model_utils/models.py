import torch
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms


imagenet_mean_pxs = np.array([0.485, 0.456, 0.406])
imagenet_std_pxs = np.array([0.229, 0.224, 0.225])

imagenet_resnet_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean_pxs, imagenet_std_pxs)
])

imagenet_resnet_train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean_pxs, imagenet_std_pxs)])

class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])
        self.in_features = original_model.fc.in_features

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def set_parameter_requires_grad(model, eval_mode=True):
    if eval_mode:
        for param in model.parameters():
            param.requires_grad = False
    return

def get_model(args, get_full_model=False, eval_mode=True):
    if args.model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model = model.to(args.device)
        train_preprocess = imagenet_resnet_train_transforms
        val_preprocess = imagenet_resnet_transforms
        set_parameter_requires_grad(model, eval_mode)
        model_bottom = ResNetBottom(model)
        model_top = ResNetTop(model)

    elif args.model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model = model.to(args.device)
        train_preprocess = imagenet_resnet_train_transforms
        val_preprocess = imagenet_resnet_transforms
        set_parameter_requires_grad(model, eval_mode)
        model_bottom = ResNetBottom(model)
        model_top = ResNetTop(model)

    else:
        raise ValueError(model_name)

    if get_full_model:
        return model, model_bottom, model_top, train_preprocess, val_preprocess
    else:
        return model_bottom, model_top, val_preprocess

