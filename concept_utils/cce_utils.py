from collections import defaultdict
import numpy as np
import torch
from torch._C import Value
import torch.nn as nn
import torch.optim as optim

from .cav_utils import ConceptBank, EasyDict


def conceptual_counterfactual(embedding: torch.Tensor, label: torch.Tensor, concept_bank, model_top: nn.Module,
                              alpha: float =1e-1, beta: float =1e-2, n_steps: int =100,
                              step_size: float=1e-2, momentum: float =0.9, enforce_validity: bool=True,
                              kappa="mean"):
    """Generating conceptual counterfactual explanations.

    Args:
        embedding (torch.Tensor): Embedding of the image to explain.
        label (torch.Tensor): Label of the image to explain.
        concept_bank : ConceptBank object that includes the concept vectors and their margin info.
        model_top (nn.Module): Predictor layer(s) of the model. 
        alpha (float, optional): L1 regularization strength. Defaults to 1e-4.
        beta (float, optional): L2 regularization strength. Defaults to 1e-4.
        n_steps (int, optional): Maximum number of steps to generate the counterfactual. Defaults to 100.
        step_size (float, optional): Learning rate for the optimizer. Defaults to 1e-1.
        momentum (float, optional): Momentum for the optimizer. Defaults to 0.9.
        enforce_validity (bool, optional): Whether to enforce the validity constraints. Defaults to True.
        kappa (str, optional): How to use the validity constraints (see the paper). Defaults to "mean".

    Returns:
        dict: Dictionary of the counterfactual. Particularly,
            "success": bool: Whether the counterfactual was successfully generated.
            "concept_scores": list: List of the concept scores.
            "concept_names": list: List of the names corresponding to the concept scores.
            "W": torch.Tensor: Counterfactual concept weights.
            
    """
    start_prediction = model_top(embedding).argmax(dim=1).detach().cpu().item()
    
    max_margins = concept_bank.margin_info.max
    min_margins = concept_bank.margin_info.min
    concept_norms = concept_bank.norms
    concept_intercepts = concept_bank.intercepts
    concepts = concept_bank.vectors
    concept_names = concept_bank.concept_names.copy()
    device = embedding.device
    emb_shape = embedding.shape
    embedding = embedding.flatten(1)

    criterion = nn.CrossEntropyLoss()
    

    
    # Concept weights to explain the mistake
    W = nn.Parameter(torch.zeros(1, concepts.shape[0], device=device), requires_grad=True)

    # Normalize the concept vectors
    normalized_C = max_margins * concepts / concept_norms
    # Compute the current distance of the sample to decision boundaries of SVMs
    margins = (torch.matmul(concepts, embedding.T) + concept_intercepts) / concept_norms
    # Computing constraints for the concepts scores
    W_clamp_max = (max_margins * concept_norms -
                   concept_intercepts - torch.matmul(concepts, embedding.T))
    W_clamp_min = (min_margins * concept_norms -
                   concept_intercepts - torch.matmul(concepts, embedding.T))

    W_clamp_max = (W_clamp_max / (max_margins * concept_norms)).T
    W_clamp_min = (W_clamp_min / (max_margins * concept_norms)).T

    if enforce_validity:
        if kappa == "mean":
            W_clamp_max[(margins > concept_bank.margin_info.pos_mean).T] = 0.
            W_clamp_min[(margins < concept_bank.margin_info.neg_mean).T] = 0.
        elif kappa == "zero":
            W_clamp_max[(margins > torch.zeros_like(margins)).T] = 0.
            W_clamp_min[(margins < torch.zeros_like(margins)).T] = 0.
        else:
            raise ValueError(f"{kappa} validation strategy is not defined.")

    zeros = torch.zeros_like(W_clamp_max)
    W_clamp_max = torch.where(
        W_clamp_max < zeros, zeros, W_clamp_max).detach().flatten(1)
    W_clamp_min = torch.where(
        W_clamp_min > zeros, zeros, W_clamp_min).detach().flatten(1)

    optimizer = optim.SGD([W], lr=step_size, momentum=momentum)
    history = []
    

    for i in range(n_steps):
        optimizer.zero_grad()
        new_embedding = embedding + torch.matmul(W, normalized_C)
        new_out = model_top(new_embedding.view(*emb_shape))
        
        l1_loss = torch.norm(W, dim=1, p=1)/W.shape[1]
        l2_loss = torch.norm(W, dim=1, p=2)/W.shape[1]
        ce_loss = criterion(new_out, label)
        loss = ce_loss + l1_loss * alpha + l2_loss * beta
        history.append(
            f"{ce_loss.detach().item()}, L1:{l1_loss.detach().item()}, L2: {l2_loss.detach().item()}")
        loss.backward()
        optimizer.step()
        if enforce_validity:
            W_projected = torch.where(W < W_clamp_min, W_clamp_min, W).detach()
            W_projected = torch.where(
                W > W_clamp_max, W_clamp_max, W_projected)
            W.data = W_projected.detach()
            W.grad.zero_()


    final_emb = embedding + torch.matmul(W, normalized_C)
    W = W[0].detach().cpu().numpy().tolist()

    concept_scores = dict()
    for i, n in enumerate(concept_names):
        concept_scores[n] = W[i]
    concept_names = sorted(concept_names, key=concept_scores.get, reverse=True)

    new_out = model_top(final_emb.view(*emb_shape))
    
    label = label.detach().cpu().item()
    # Check if the counterfactual could flip the label
    if (start_prediction == label):
        success = True
    else:
        success = False
    
    explanation = {"success": success,
                  "concept_scores": concept_scores,
                  "concept_scores_list": concept_names,
                  "W": np.array(W),
                  "prediction": start_prediction,
                  "label": label}
    return EasyDict(explanation)

