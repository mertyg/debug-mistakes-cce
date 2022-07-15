# Author: merty
# A simple training script for metashift scenarios.

import argparse
import sys
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from dataset import load_data
from utils import AverageMeter
sys.path.append('../')
from model_utils import get_model

def config():
    parser = argparse.ArgumentParser()
    
    # Model and data details
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='bear-bird-cat-dog-elephant:dog(water)')
    
    # Training details
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1)
    
    # File details
    parser.add_argument('--out_dir', type=str, default='/oak/stanford/groups/jamesz/merty/cce')
   
    return parser.parse_args()


@torch.no_grad()
def eval_loop(loader, model, device):
    model.eval()
    accuracymeter = AverageMeter()
    lossmeter = AverageMeter()
    criterion = nn.CrossEntropyLoss() 
    tqdm_loader = tqdm(loader)
    for inputs, labels in tqdm_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        preds = torch.argmax(outputs, 1)
        
        # Log the accuracy and the loss
        accuracymeter.update((preds == labels).float().mean().cpu().numpy(), inputs.size(0))
        lossmeter.update((preds == labels).float().mean().cpu().numpy(), inputs.size(0))
        tqdm_loader.set_postfix(Acc=accuracymeter.avg, Loss=lossmeter.avg)
    


def train_model(train_loader, val_loader, model, optimizer, num_epochs, device):    
    criterion = nn.CrossEntropyLoss() 
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()
        
  
        tqdm_loader = tqdm(train_loader)
        for inputs, labels in tqdm_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            tqdm_loader.set_postfix(loss=loss.item())

        model.eval()
        print('-' * 10)
        print('Evaluating on the training set...')
        eval_loop(train_loader, model, device)
        print('-' * 10)
        print('Evaluating on validation set...')
        eval_loop(val_loader, model, device)



def main(args):
    model, _, _, train_preprocess, val_preprocess = get_model(args, get_full_model=True, eval_mode=True)
    loaders, cls_to_lbl, data_meta_info = load_data(args, train_preprocess, val_preprocess, args.dataset)

    # Change the classifier layer
    model.fc = nn.Linear(model.fc.in_features, len(cls_to_lbl))
    for p in model.fc.parameters():
        p.requires_grad = True
    

    model = model.to(args.device)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    

    # Train the model
    train_model(loaders["train"], loaders["val"], model, optimizer, args.num_epochs, args.device)

    scenario_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(scenario_dir, exist_ok=True)
    
    
    # Save the model and configurations
    model = model.to("cpu")
    torch.save(model, os.path.join(scenario_dir, "confounded-model.pt"))
    with open(os.path.join(scenario_dir, "data_meta_info.pkl"), "wb") as f:
        pickle.dump(data_meta_info, f)
    with open(os.path.join(scenario_dir, "args"), "wb") as f:
        pickle.dump(args, f)
    
if __name__ == "__main__":
    args = config()
    main(args)