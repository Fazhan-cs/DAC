import sys
import numpy as np
import torch
import torch.nn.functional as F
from .loralib import layers as lora_layers,LinearLoRA
import torch.nn as nn
import sys
import torch
def load_lora_weights(lora_paths):
    all_loaded_lora_Bs = {}
    for lora_path in lora_paths:
        save_data = torch.load(lora_path)
        loaded_weights = save_data['weights']
        print(f"Loaded LoRA weights from: {lora_path}")
        print(f"Total layers found: {len(loaded_weights)}")
        for layer_key in loaded_weights:
            layer_idx = int(layer_key.split('_')[1])  # layer_0->0
            layer_data = loaded_weights[layer_key]
            for proj_type in layer_data:
                if 'w_lora_B' in layer_data[proj_type]:
                    loaded_B = layer_data[proj_type]['w_lora_B'].detach()
                    if 'num' in layer_data:
                        num = layer_data['num']
                        _, topk_indices = torch.topk(num, 4)
                        topk_mask = torch.zeros(loaded_B.size(1), dtype=torch.bool)
                        topk_mask[topk_indices] = True
                        loaded_B = loaded_B[:,topk_mask]
                    key = (layer_idx, proj_type)
                    if key in all_loaded_lora_Bs:
                        
                        all_loaded_lora_Bs[key] = torch.cat([all_loaded_lora_Bs[key], loaded_B], dim=1)
                    else:
                        all_loaded_lora_Bs[key] = loaded_B
    return all_loaded_lora_Bs


def compute_orthogonal_loss(model, loaded_lora_Bs):
    orthogonal_loss = 0.0
    valid_pairs_count = 0
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    for (layer_idx, proj_type), loaded_B in loaded_lora_Bs.items():
            layer_idx = layer_idx // 3
            if layer_idx < 12: 
                encoder = model.transformer  
                layer = encoder.resblocks[layer_idx] 
            else:  
                encoder = model.visual.transformer  
                layer = encoder.resblocks[layer_idx - 12]  
            if proj_type == 'linear':
                mlp = layer.mlp
                for name, submodule in mlp.named_children():
                    if isinstance(submodule, LinearLoRA) and submodule.w_lora_B.shape[0] == loaded_B.shape[0]:
                        current_module = submodule
                        break
            else:
                attn_layer = layer.attn  
                current_module = getattr(attn_layer, proj_type)

            current_B = current_module.w_lora_B  
            rank_activation_counts = current_module.rank_activation_counts
            _, topk_indices = torch.topk(rank_activation_counts, 4)
            topk_mask = torch.zeros_like(rank_activation_counts, dtype=torch.bool)
            topk_mask[topk_indices] = True
            current_B_active = current_B[:,topk_mask]

            loaded_B = loaded_B.to(device=current_B_active.device, dtype=current_B_active.dtype)
            loss_value = torch.abs(torch.mm(current_B_active.T, loaded_B)).sum()
            
            orthogonal_loss += loss_value
            valid_pairs_count += 1 

    
    if valid_pairs_count > 0:
        average_orthogonal_loss = orthogonal_loss / valid_pairs_count
    else:
        average_orthogonal_loss = 0.0
    print(f"\nTotal orthogonal loss: {orthogonal_loss}")
        
    return average_orthogonal_loss


def print_model_with_gradients(model):
    for name, module in model.named_children():
        print(f"{name}: {module.__class__.__name__}")
        for param_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                print(f"  ({param_name}): {param.size()}")
        
        print_model_with_gradients(module)

def get_module_path(model, target_module):
    path = []
    def find_path(module, current_path):
        for name, sub_module in module.named_children():
            new_path = current_path + [name]
            if sub_module is target_module:
                nonlocal path
                path = new_path
                return
            find_path(sub_module, new_path)
    find_path(model, [])
    return ".".join(path)
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
def clip_loss(logits_per_text,labels):
    batch_size = logits_per_text.size(0)
    targets = torch.zeros(batch_size, batch_size, device=logits_per_text.device)
    for i in range(batch_size):
        for j in range(batch_size):
            if labels[i] == labels[j]:
                targets[i, j] = 1
    targets = targets / targets.sum(dim=1, keepdim=True)

    texts_loss = F.cross_entropy(logits_per_text, targets)
    images_loss = F.cross_entropy(logits_per_text.t(), targets.t())
    loss = (images_loss + texts_loss) / 2.0
    return loss