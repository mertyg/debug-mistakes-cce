# Simple script to demonstrate CCE
import os
import pickle
import argparse
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from model_utils import get_model, ResNetBottom, ResNetTop
from model_utils import imagenet_resnet_transforms as preprocess
from concept_utils import conceptual_counterfactual, ConceptBank



def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./examples/models/dog(snow).pth", type=str)
    
    parser.add_argument("--concept-bank", default="./examples/resnet18_bank.pkl", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--image-folder", default="./examples/images/", type=str)
    parser.add_argument("--explanation-folder", default="./examples/explanations/", type=str)
    return parser.parse_args()


def viz_explanation(image, explanation, class_to_idx):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    exp_text = [f"Label: {class_to_idx[explanation.label]}"]
    exp_text.append(f"Prediction: {class_to_idx[explanation.prediction]}")
    exp_text.extend([f"{c}: {explanation.concept_scores[c]:.2f}" for c in explanation.concept_scores_list[:2]])
    exp_text.extend([f"{c}: {explanation.concept_scores[c]:.2f}" for c in explanation.concept_scores_list[-2:]])
    exp_text = "\n".join(exp_text)
    ax.imshow(image)
    props = dict(boxstyle='round', facecolor='salmon', alpha=0.9)
    ax.axis("off")
    ax.text(0, 1.0, exp_text,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=10,
            bbox=props)
    fig.tight_layout()
    return fig
    
    
def main(args):
    sns.set_context("poster")
    np.random.seed(args.seed)
    
    # Load the model
    model = torch.load(args.model_path)
    model = model.to(args.device)
    model = model.eval()
    
    # TODO: Class indices are here
    idx_to_class = {0: "bear", 1: "bird", 2: "cat", 3: "dog", 4: "elephant"}
    cls_to_idx = {v: k for k, v in idx_to_class.items()}
    
    # Split the model into the backbone and the predictor layer
    backbone, model_top = ResNetBottom(model), ResNetTop(model)
    
    # Load the concept bank
    concept_bank = ConceptBank(pickle.load(open(args.concept_bank, "rb")), device=args.device)

    os.makedirs(args.explanation_folder, exist_ok=True)
    
    for image_path in os.listdir(args.image_folder):
        # Read the image and label
        image = Image.open(os.path.join(args.image_folder, image_path))
        image_tensor = preprocess(image).to(args.device)
        
        label = cls_to_idx["dog"]*torch.ones(1, dtype=torch.long).to(args.device)
        
        # Get the embedding for the image
        embedding = backbone(image_tensor.unsqueeze(0))
        # Get the model prediction
        pred = model_top(embedding).argmax(dim=1)
        
        # Only evaluate over mistakes
        if pred.item() == label.item():
            print(f"Warning: {image_path} is correctly classified, but we'll still try to increase the confidence if you really want that.")
        
        # Get the embedding for the image
        embedding = backbone(image_tensor.unsqueeze(0)).detach()
        # Run CCE
        explanation = conceptual_counterfactual(embedding, label, concept_bank, model_top) 
        
        # Visualize the explanation, and save it to a figure
        fig = viz_explanation(image, explanation, idx_to_class) 
        fig.savefig(os.path.join(args.explanation_folder, f"{image_path.split('.')[0]}_explanation.png"))
        
if __name__ == "__main__":
    args = config()
    main(args)