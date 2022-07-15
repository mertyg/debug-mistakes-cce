import os
import pickle
import argparse
import torch
import numpy as np

from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

from concept_utils import learn_concept_bank, ListDataset
from model_utils import get_model

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-dir", required=True, type=str, 
                        help="Directory containing concept images. See below for a detailed description.")
    parser.add_argument("--out-dir", default="/oak/stanford/groups/jamesz/merty/cce", type=str,
                        help="Where to save the concept bank.")
    parser.add_argument("--model-name", default="resnet18", type=str, help="Name of the model to use.")
    parser.add_argument("--device", default="cuda", type=str)
    
    
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--C", nargs="+", default=[1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0], type=float, 
                        help="Regularization parameter for SVMs. Can specify multiple values.")
    
    parser.add_argument("--n-samples", default=50, type=int, 
                        help="Number of pairs of positive/negative samples used to train SVMs.")
    return parser.parse_args()


def main(args):
    np.random.seed(args.seed)
    
    # Concept images are expected in the following format:
    # args.concept_dir/concept_name/positives/1.jpg, args.concept_dir/concept_name/positives/2.jpg, ...
    # args.concept_dir/concept_name/negatives/1.jpg, args.concept_dir/concept_name/negatives/2.jpg, ...
    
    concept_names = os.listdir(args.concept_dir)
    
    # Get the backbone
    backbone, _, preprocess = get_model(args)
    backbone = backbone.to(args.device)
    backbone = backbone.eval()
    
    print(f"Attempting to learn {len(concept_names)} concepts.")
    concept_lib = {C: {} for C in args.C}
    for concept in concept_names:
        pos_ims = glob(os.path.join(args.concept_dir, concept, "positives", "*"))
        neg_ims = glob(os.path.join(args.concept_dir, concept, "negatives", "*"))
        
        pos_dataset = ListDataset(pos_ims, preprocess=preprocess)
        neg_dataset = ListDataset(neg_ims, preprocess=preprocess)
        print(len(pos_dataset), len(neg_dataset))
        pos_loader = torch.utils.data.DataLoader(pos_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, args.n_samples, args.C, device=args.device)
        # Store CAV train acc, val acc, margin info for each regularization parameter and each concept
        for C in args.C:
            concept_lib[C][concept] = cav_info[C]
            print(f"{concept} with C={C}: Training Accuracy: {cav_info[C][1]:.2f}, Validation Accuracy: {cav_info[C][2]:.2f}")
    
    # Save CAV results 
    os.makedirs(args.out_dir, exist_ok=True)
    for C in concept_lib.keys():
        lib_path = os.path.join(args.out_dir, f"{args.model_name}_{C}_{args.n_samples}.pkl")
        with open(lib_path, "wb") as f:
            pickle.dump(concept_lib[C], f)
        print(f"Saved to: {lib_path}")        
    


if __name__ == "__main__":
    args = config()
    main(args)
