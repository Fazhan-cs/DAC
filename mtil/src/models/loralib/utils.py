import os

import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer, PlainMultiheadAttentionLoRA,LinearLoRA

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}




def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters_change(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params
def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if 'initial_w_lora' in name:
            continue
        if bias == 'none':
            if 'lora_A' in name or 'lora_B' in name or 'lora_router' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params

def apply_lora_all(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            #print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if args.at and isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
                       # print(f"Added text lora layer at index {len(list_lora_layers) - 1}")
                    if args.linear and isinstance(submodule, nn.Sequential):
                        mlp = getattr(block, 'mlp', None) 
                        if mlp is not None:
                            for seq_name, seq_module in mlp.named_children():  
                                if isinstance(seq_module, nn.Linear):
                                    new_linear_lora = LinearLoRA(
                                        seq_module, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate
                                    )
                                    setattr(mlp, seq_name, new_linear_lora) 
                                    list_lora_layers.append(new_linear_lora)
                                   # print(f"Added LinearLoRA layer in text encoder at index {len(list_lora_layers) - 1}")

                    

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            #print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if args.at and isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
                       # print(f"Added vision lora layer at index {len(list_lora_layers) - 1}")
                    if args.linear and isinstance(submodule, nn.Sequential):
                        mlp = getattr(block, 'mlp', None) 
                        if mlp is not None:
                            for seq_name, seq_module in mlp.named_children(): 
                                if isinstance(seq_module, nn.Linear):
                                    new_linear_lora = LinearLoRA(
                                        seq_module, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate
                                    )
                                    setattr(mlp, seq_name, new_linear_lora)  
                                    list_lora_layers.append(new_linear_lora)
                                   # print(f"Added LinearLoRA layer in text encoder at index {len(list_lora_layers) - 1}")
    return list_lora_layers

def save_lora_all(args, list_lora_layers):
    save_lora_qkv(args,list_lora_layers)
    
def save_lora_qkv(args, list_lora_layers):
    if args.adaw:
        weights = {}
        for i, layer in enumerate(list_lora_layers):
            layer_weights = {}
            if args.at and isinstance(layer, PlainMultiheadAttentionLoRA):
                if 'q' in args.params:
                    layer_weights['q_proj'] = {
                        'w_lora_A': layer.q_proj.w_lora_A.data,
                        'w_lora_B': layer.q_proj.w_lora_B.data,
                        #'w_lora_w': layer.q_proj.w_lora_w.data,
                        'num':layer.q_proj.rank_activation_counts.data
                    }
                if 'k' in args.params:
                    layer_weights['k_proj'] = {
                        'w_lora_A': layer.k_proj.w_lora_A.data,
                        'w_lora_B': layer.k_proj.w_lora_B.data,
                       # 'w_lora_w': layer.k_proj.w_lora_w.data,
                        'num':layer.k_proj.rank_activation_counts.data
                    }
                if 'v' in args.params:
                    layer_weights['v_proj'] = {
                        'w_lora_A': layer.v_proj.w_lora_A.data,
                        'w_lora_B': layer.v_proj.w_lora_B.data,
                        #'w_lora_w': layer.v_proj.w_lora_w.data,
                        'num':layer.v_proj.rank_activation_counts.data

                    }
                if 'o' in args.params:
                    layer_weights['proj'] = {
                        'w_lora_A': layer.proj.w_lora_A.data,
                        'w_lora_B': layer.proj.w_lora_B.data,
                       # 'w_lora_w': layer.proj.w_lora_w.data,
                        'num':layer.proj.rank_activation_counts.data

                    }
            if args.linear and isinstance(layer,LinearLoRA):
                layer_weights['linear'] = {
                'w_lora_A': layer.w_lora_A.data,
                'w_lora_B': layer.w_lora_B.data,
                #'w_lora_w': layer.w_lora_w.data,
                'num':layer.rank_activation_counts.data

            }

            weights[f'layer_{i}'] = layer_weights

        metadata = {
            'r': args.r,
            'alpha': args.alpha,
            'encoder': args.encoder,
            'params': args.params,
            'position': args.position
        }

        save_data = {
            'weights': weights,
            'metadata': metadata
        }

        # to manage names like ViT-B/16
        backbone = args.backbone.replace('/', '').replace('-', '').lower()
        save_dir = args.save
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(args.save, f"{args.train_dataset}.pt")
        #save_path = f'{save_dir}/{args.filename}.pt'
        torch.save(save_data, save_path)
        print(f'LoRA weights saved to {save_path}')
    else:
        weights = {}
        for i, layer in enumerate(list_lora_layers):
            layer_weights = {}
            if args.at and isinstance(layer, PlainMultiheadAttentionLoRA):
                if 'q' in args.params:
                    layer_weights['q_proj'] = {
                        'w_lora_A': layer.q_proj.w_lora_A.data,
                        'w_lora_B': layer.q_proj.w_lora_B.data,
                    }
                if 'k' in args.params:
                    layer_weights['k_proj'] = {
                        'w_lora_A': layer.k_proj.w_lora_A.data,
                        'w_lora_B': layer.k_proj.w_lora_B.data,
                    }
                if 'v' in args.params:
                    layer_weights['v_proj'] = {
                        'w_lora_A': layer.v_proj.w_lora_A.data,
                        'w_lora_B': layer.v_proj.w_lora_B.data,

                    }
                if 'o' in args.params:
                    layer_weights['proj'] = {
                        'w_lora_A': layer.proj.w_lora_A.data,
                        'w_lora_B': layer.proj.w_lora_B.data,

                    }
            if args.linear and isinstance(layer,LinearLoRA):
                layer_weights['linear'] = {
                'w_lora_A': layer.w_lora_A.data,
                'w_lora_B': layer.w_lora_B.data,
            }
            weights[f'layer_{i}'] = layer_weights

        metadata = {
            'r': args.r,
            'alpha': args.alpha,
            'encoder': args.encoder,
            'params': args.params,
            'position': args.position
        }

        save_data = {
            'weights': weights,
            'metadata': metadata
        }

        # to manage names like ViT-B/16
        save_dir = args.save
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(args.save, f"{args.train_dataset}.pt")
        #save_path = f'{save_dir}/{args.filename}.pt'
        torch.save(save_data, save_path)
        print(f'LoRA weights saved to {save_path}')


def load_lora_all(args,list_lora_layers,path):
    load_lora_qkv(args,list_lora_layers,path)

def load_lora_qkv(args, list_lora_layers,path):
    if args.adaw:
        backbone = args.backbone.replace('/', '').replace('-', '').lower()
        load_path = path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f'File {load_path} does not exist.')

        loaded_data = torch.load(load_path)

        metadata = loaded_data['metadata']
        if metadata['r'] != args.r:
            raise ValueError(
                f"r mismatch: expected {args.r}, found {metadata['r']}")
        if metadata['alpha'] != args.alpha:
            raise ValueError(
                f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
        if metadata['encoder'] != args.encoder:
            raise ValueError(
                f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
        if metadata['params'] != args.params:
            raise ValueError(
                f"Params mismatch: expected {args.params}, found {metadata['params']}")
        if metadata['position'] != args.position:
            raise ValueError(
                f"Position mismatch: expected {args.position}, found {metadata['position']}")

        weights = loaded_data['weights']
        for i, layer in enumerate(list_lora_layers):
            layer_weights = weights[f'layer_{i}']
            if args.at and isinstance(layer, PlainMultiheadAttentionLoRA):
                if 'q' in args.params and 'q_proj' in layer_weights:
                    layer.q_proj.w_lora_A.data.copy_(
                        layer_weights['q_proj']['w_lora_A'])
                    layer.q_proj.w_lora_B.data.copy_(
                        layer_weights['q_proj']['w_lora_B'])
                    #layer.q_proj.w_lora_w.data.copy_(
                    #    layer_weights['q_proj']['w_lora_w'])
                    layer.q_proj.rank_activation_counts.data.copy_(
                        layer_weights['q_proj']['num'])
                if 'k' in args.params and 'k_proj' in layer_weights:
                    layer.k_proj.w_lora_A.data.copy_(
                        layer_weights['k_proj']['w_lora_A'])
                    layer.k_proj.w_lora_B.data.copy_(
                        layer_weights['k_proj']['w_lora_B'])
                   # layer.k_proj.w_lora_w.data.copy_(
                    #    layer_weights['k_proj']['w_lora_w'])
                    layer.k_proj.rank_activation_counts.data.copy_(
                        layer_weights['k_proj']['num'])
                if 'v' in args.params and 'v_proj' in layer_weights:
                    layer.v_proj.w_lora_A.data.copy_(
                        layer_weights['v_proj']['w_lora_A'])
                    layer.v_proj.w_lora_B.data.copy_(
                        layer_weights['v_proj']['w_lora_B'])
                    #layer.v_proj.w_lora_w.data.copy_(
                    #    layer_weights['v_proj']['w_lora_w'])
                    layer.v_proj.rank_activation_counts.data.copy_(
                        layer_weights['v_proj']['num'])
                if 'o' in args.params and 'proj' in layer_weights:
                    layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
                    layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])
                    #layer.proj.w_lora_w.data.copy_(layer_weights['proj']['w_lora_w'])
                    layer.proj.rank_activation_counts.data.copy_(layer_weights['proj']['num'])

            if args.linear and isinstance(layer, LinearLoRA):
                layer_weights_linear = layer_weights.get('linear', {})
                if 'w_lora_A' in layer_weights_linear:
                    layer.w_lora_A.data.copy_(layer_weights_linear['w_lora_A'])
                if 'w_lora_B' in layer_weights_linear:
                    layer.w_lora_B.data.copy_(layer_weights_linear['w_lora_B'])
                #if 'w_lora_w' in layer_weights_linear:
                #    layer.w_lora_w.data.copy_(layer_weights_linear['w_lora_w'])
                    
                layer.rank_activation_counts.data.copy_(layer_weights_linear['num'])

        print(f'LoRA weights loaded from {load_path}')
    else:
    
        backbone = args.backbone.replace('/', '').replace('-', '').lower()
        #load_path = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}/{args.filename}.pt'
        load_path = path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f'File {load_path} does not exist.')

        loaded_data = torch.load(load_path)

        metadata = loaded_data['metadata']
        if metadata['r'] != args.r:
            raise ValueError(
                f"r mismatch: expected {args.r}, found {metadata['r']}")
        if metadata['alpha'] != args.alpha:
            raise ValueError(
                f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
        if metadata['encoder'] != args.encoder:
            raise ValueError(
                f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
        if metadata['params'] != args.params:
            raise ValueError(
                f"Params mismatch: expected {args.params}, found {metadata['params']}")
        if metadata['position'] != args.position:
            raise ValueError(
                f"Position mismatch: expected {args.position}, found {metadata['position']}")

        weights = loaded_data['weights']
        for i, layer in enumerate(list_lora_layers):
            layer_weights = weights[f'layer_{i}']
            if args.at and isinstance(layer, PlainMultiheadAttentionLoRA):
                if 'q' in args.params and 'q_proj' in layer_weights:
                    layer.q_proj.w_lora_A.data.copy_(
                        layer_weights['q_proj']['w_lora_A'])
                    layer.q_proj.w_lora_B.data.copy_(
                        layer_weights['q_proj']['w_lora_B'])
                if 'k' in args.params and 'k_proj' in layer_weights:
                    layer.k_proj.w_lora_A.data.copy_(
                        layer_weights['k_proj']['w_lora_A'])
                    layer.k_proj.w_lora_B.data.copy_(
                        layer_weights['k_proj']['w_lora_B'])
                if 'v' in args.params and 'v_proj' in layer_weights:
                    layer.v_proj.w_lora_A.data.copy_(
                        layer_weights['v_proj']['w_lora_A'])
                    layer.v_proj.w_lora_B.data.copy_(
                        layer_weights['v_proj']['w_lora_B'])
                if 'o' in args.params and 'proj' in layer_weights:
                    layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
                    layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])
            if args.linear and isinstance(layer, LinearLoRA):
                layer_weights_linear = layer_weights.get('linear', {})
                if 'w_lora_A' in layer_weights_linear:
                    layer.w_lora_A.data.copy_(layer_weights_linear['w_lora_A'])
                if 'w_lora_B' in layer_weights_linear:
                    layer.w_lora_B.data.copy_(layer_weights_linear['w_lora_B'])

        print(f'LoRA weights loaded from {load_path}')






def merge_lora_back_to_original_all(args, clip_model, list_lora_layers,indexing,max_len):
    # if indexing==max_len-1:
    #     topk=8
    # else:
    #     topk=4
    topk=4
    index=12
    if args.linear:
        index=36
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        lora_layer_index = 0
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                   
                    if args.at and isinstance(submodule, PlainMultiheadAttentionLoRA):
                        original_mha = nn.MultiheadAttention(
                            embed_dim=submodule.embed_dim,
                            num_heads=submodule.num_heads,
                            dropout=submodule.dropout,
                            bias=submodule.k_proj.bias is not None,
                            add_bias_kv=False,
                            add_zero_attn=False,
                            kdim=submodule.kdim,
                            vdim=submodule.vdim
                        )

                        q_weight = submodule.q_proj.ori.weight if 'q' in args.params else submodule.q_proj.weight
                        k_weight = submodule.k_proj.ori.weight if 'k' in args.params else submodule.k_proj.weight
                        v_weight = submodule.v_proj.ori.weight if 'v' in args.params else submodule.v_proj.weight
                        original_mha.in_proj_weight.data.copy_(torch.cat([
                            q_weight.data,
                            k_weight.data,
                            v_weight.data
                        ], dim=0))
                        # original_mha.in_proj_weight.data.copy_(torch.cat([
                        #     submodule.q_proj.weight.data,
                        #     submodule.k_proj.weight.data,
                        #     submodule.v_proj.weight.data
                        # ], dim=0))
                        if submodule.k_proj.bias is not None:
                            q_bias = submodule.q_proj.ori.bias if 'q' in args.params and hasattr(submodule.q_proj.ori, 'bias') else submodule.q_proj.bias
                            k_bias = submodule.k_proj.ori.bias if 'k' in args.params and hasattr(submodule.k_proj.ori, 'bias') else submodule.k_proj.bias
                            v_bias = submodule.v_proj.ori.bias if 'v' in args.params and hasattr(submodule.v_proj.ori, 'bias') else submodule.v_proj.bias

                            original_mha.in_proj_bias.data.copy_(torch.cat([
                                q_bias.data,
                                k_bias.data,
                                v_bias.data
                            ], dim=0))

                        o_weight = submodule.proj.ori.weight if 'o' in args.params else submodule.proj.weight
                        original_mha.out_proj.weight.data.copy_(o_weight.data)

                        o_bias = submodule.proj.ori.bias if 'o' in args.params else submodule.proj.bias
                        if submodule.proj.bias is not None:
                            original_mha.out_proj.bias.data.copy_(o_bias.data)

                        # merge
                        lora_layer = list_lora_layers[lora_layer_index]
                        if 'q' in args.params and isinstance(lora_layer.q_proj, LinearLoRA):
                            lora_adjustment_q = lora_layer.q_proj.merge_BA('weight',topk) * lora_layer.q_proj.scaling
                            original_mha.in_proj_weight.data[:submodule.embed_dim, :] += lora_adjustment_q
                        if 'k' in args.params and isinstance(lora_layer.k_proj, LinearLoRA):
                            lora_adjustment_k = lora_layer.k_proj.merge_BA('weight',topk) * lora_layer.k_proj.scaling
                            original_mha.in_proj_weight.data[submodule.embed_dim:2*submodule.embed_dim, :] += lora_adjustment_k
                        if 'v' in args.params and isinstance(lora_layer.v_proj, LinearLoRA):
                            lora_adjustment_v = lora_layer.v_proj.merge_BA('weight',topk) * lora_layer.v_proj.scaling
                            original_mha.in_proj_weight.data[2*submodule.embed_dim:, :] += lora_adjustment_v
                        if 'o' in args.params and isinstance(lora_layer.proj, LinearLoRA):
                            lora_adjustment_proj = lora_layer.proj.merge_BA('weight',topk) * lora_layer.proj.scaling
                            original_mha.out_proj.weight.data += lora_adjustment_proj

                        setattr(block, name, original_mha)
                        lora_layer_index += 1
                    if args.linear and isinstance(submodule, nn.Sequential):
                        for seq_name, seq_module in submodule.named_children():
                        
                            if isinstance(seq_module, LinearLoRA):
                                original_linear = nn.Linear(
                                    in_features=seq_module.in_features,
                                    out_features=seq_module.out_features
                                )
                                original_linear.weight.data.copy_(seq_module.ori.weight.data)
                                if seq_module.ori.bias is not None:
                                    original_linear.bias.data.copy_(seq_module.ori.bias.data)
                                    
                                lora_adjustment_li = seq_module.merge_BA('weight',topk) * seq_module.scaling
                                original_linear.weight.data += lora_adjustment_li
                        
                                mlp = getattr(block, 'mlp') 
                                setattr(mlp, seq_name, original_linear)
                                lora_layer_index += 1

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        lora_layer_index = index
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if args.at and isinstance(submodule, PlainMultiheadAttentionLoRA):
                        original_mha = nn.MultiheadAttention(
                            embed_dim=submodule.embed_dim,
                            num_heads=submodule.num_heads,
                            dropout=submodule.dropout,
                            bias=submodule.k_proj.bias is not None,
                            add_bias_kv=False,
                            add_zero_attn=False,
                            kdim=submodule.kdim,
                            vdim=submodule.vdim
                        )

                        q_weight = submodule.q_proj.ori.weight if 'q' in args.params else submodule.q_proj.weight
                        k_weight = submodule.k_proj.ori.weight if 'k' in args.params else submodule.k_proj.weight
                        v_weight = submodule.v_proj.ori.weight if 'v' in args.params else submodule.v_proj.weight
                        original_mha.in_proj_weight.data.copy_(torch.cat([
                            q_weight.data,
                            k_weight.data,
                            v_weight.data
                        ], dim=0))
                        # original_mha.in_proj_weight.data.copy_(torch.cat([
                        #     submodule.q_proj.weight.data,
                        #     submodule.k_proj.weight.data,
                        #     submodule.v_proj.weight.data
                        # ], dim=0))
                        if submodule.k_proj.bias is not None:
                            q_bias = submodule.q_proj.ori.bias if 'q' in args.params and hasattr(submodule.q_proj.ori, 'bias') else submodule.q_proj.bias
                            k_bias = submodule.k_proj.ori.bias if 'k' in args.params and hasattr(submodule.k_proj.ori, 'bias') else submodule.k_proj.bias
                            v_bias = submodule.v_proj.ori.bias if 'v' in args.params and hasattr(submodule.v_proj.ori, 'bias') else submodule.v_proj.bias

                            original_mha.in_proj_bias.data.copy_(torch.cat([
                                q_bias.data,
                                k_bias.data,
                                v_bias.data
                            ], dim=0))

                        o_weight = submodule.proj.ori.weight if 'o' in args.params else submodule.proj.weight
                        original_mha.out_proj.weight.data.copy_(o_weight.data)

                        o_bias = submodule.proj.ori.bias if 'o' in args.params else submodule.proj.bias
                        if submodule.proj.bias is not None:
                            original_mha.out_proj.bias.data.copy_(o_bias.data)

                        lora_layer = list_lora_layers[lora_layer_index]
                        if 'q' in args.params and isinstance(lora_layer.q_proj, LinearLoRA):
                            lora_adjustment_q = lora_layer.q_proj.merge_BA('weight',topk) * lora_layer.q_proj.scaling
                            original_mha.in_proj_weight.data[:submodule.embed_dim, :] += lora_adjustment_q
                        if 'k' in args.params and isinstance(lora_layer.k_proj, LinearLoRA):
                            lora_adjustment_k = lora_layer.k_proj.merge_BA('weight',topk) * lora_layer.k_proj.scaling
                            original_mha.in_proj_weight.data[submodule.embed_dim:2*submodule.embed_dim, :] += lora_adjustment_k
                        if 'v' in args.params and isinstance(lora_layer.v_proj, LinearLoRA):
                            lora_adjustment_v = lora_layer.v_proj.merge_BA('weight',topk) * lora_layer.v_proj.scaling
                            original_mha.in_proj_weight.data[2*submodule.embed_dim:, :] += lora_adjustment_v
                        if 'o' in args.params and isinstance(lora_layer.proj, LinearLoRA):
                            lora_adjustment_proj = lora_layer.proj.merge_BA('weight',topk) * lora_layer.proj.scaling
                            original_mha.out_proj.weight.data += lora_adjustment_proj

                        setattr(block, name, original_mha)
                        lora_layer_index += 1
                    if args.linear and isinstance(submodule, nn.Sequential):
                        for seq_name, seq_module in submodule.named_children():
                            if isinstance(seq_module, LinearLoRA):
                                original_linear = nn.Linear(
                                    in_features=seq_module.in_features,
                                    out_features=seq_module.out_features
                                )
                                #original_linear.load_state_dict(seq_module.ori.state_dict())
                                original_linear.weight.data.copy_(seq_module.ori.weight.data)
                                if seq_module.ori.bias is not None:
                                    original_linear.bias.data.copy_(seq_module.ori.bias.data)
                                lora_adjustment_li = seq_module.merge_BA('weight',topk) * seq_module.scaling
                                original_linear.weight.data += lora_adjustment_li  
                                
                                mlp = getattr(block, 'mlp')  
                                setattr(mlp, seq_name, original_linear)
                                lora_layer_index += 1

    return clip_model

def merge_lora_back_to_original_qkv(args, clip_model, list_lora_layers):
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        lora_layer_index = 0
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                   
                    if isinstance(submodule, PlainMultiheadAttentionLoRA):
                        original_mha = nn.MultiheadAttention(
                            embed_dim=submodule.embed_dim,
                            num_heads=submodule.num_heads,
                            dropout=submodule.dropout,
                            bias=submodule.k_proj.bias is not None,
                            add_bias_kv=False,
                            add_zero_attn=False,
                            kdim=submodule.kdim,
                            vdim=submodule.vdim
                        )

                        q_weight = submodule.q_proj.ori.weight if 'q' in args.params else submodule.q_proj.weight
                        k_weight = submodule.k_proj.ori.weight if 'k' in args.params else submodule.k_proj.weight
                        v_weight = submodule.v_proj.ori.weight if 'v' in args.params else submodule.v_proj.weight
                        original_mha.in_proj_weight.data.copy_(torch.cat([
                            q_weight.data,
                            k_weight.data,
                            v_weight.data
                        ], dim=0))
                        # original_mha.in_proj_weight.data.copy_(torch.cat([
                        #     submodule.q_proj.weight.data,
                        #     submodule.k_proj.weight.data,
                        #     submodule.v_proj.weight.data
                        # ], dim=0))
                        if submodule.k_proj.bias is not None:
                            q_bias = submodule.q_proj.ori.bias if 'q' in args.params and hasattr(submodule.q_proj.ori, 'bias') else submodule.q_proj.bias
                            k_bias = submodule.k_proj.ori.bias if 'k' in args.params and hasattr(submodule.k_proj.ori, 'bias') else submodule.k_proj.bias
                            v_bias = submodule.v_proj.ori.bias if 'v' in args.params and hasattr(submodule.v_proj.ori, 'bias') else submodule.v_proj.bias

                            original_mha.in_proj_bias.data.copy_(torch.cat([
                                q_bias.data,
                                k_bias.data,
                                v_bias.data
                            ], dim=0))

                        o_weight = submodule.proj.ori.weight if 'o' in args.params else submodule.proj.weight
                        original_mha.out_proj.weight.data.copy_(o_weight.data)

                        o_bias = submodule.proj.ori.bias if 'o' in args.params else submodule.proj.bias
                        if submodule.proj.bias is not None:
                            original_mha.out_proj.bias.data.copy_(o_bias.data)

                        # 合并 LoRA 调整量
                        lora_layer = list_lora_layers[lora_layer_index]
                        if 'q' in args.params and isinstance(lora_layer.q_proj, LinearLoRA):
                            lora_adjustment_q = lora_layer.q_proj.merge_BA('weight') * lora_layer.q_proj.scaling
                            original_mha.in_proj_weight.data[:submodule.embed_dim, :] += lora_adjustment_q
                        if 'k' in args.params and isinstance(lora_layer.k_proj, LinearLoRA):
                            lora_adjustment_k = lora_layer.k_proj.merge_BA('weight') * lora_layer.k_proj.scaling
                            original_mha.in_proj_weight.data[submodule.embed_dim:2*submodule.embed_dim, :] += lora_adjustment_k
                        if 'v' in args.params and isinstance(lora_layer.v_proj, LinearLoRA):
                            lora_adjustment_v = lora_layer.v_proj.merge_BA('weight') * lora_layer.v_proj.scaling
                            original_mha.in_proj_weight.data[2*submodule.embed_dim:, :] += lora_adjustment_v
                        if 'o' in args.params and isinstance(lora_layer.proj, LinearLoRA):
                            lora_adjustment_proj = lora_layer.proj.merge_BA('weight') * lora_layer.proj.scaling
                            original_mha.out_proj.weight.data += lora_adjustment_proj

                        setattr(block, name, original_mha)
                        lora_layer_index += 1

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        lora_layer_index = 12
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, PlainMultiheadAttentionLoRA):
                        original_mha = nn.MultiheadAttention(
                            embed_dim=submodule.embed_dim,
                            num_heads=submodule.num_heads,
                            dropout=submodule.dropout,
                            bias=submodule.k_proj.bias is not None,
                            add_bias_kv=False,
                            add_zero_attn=False,
                            kdim=submodule.kdim,
                            vdim=submodule.vdim
                        )

                        q_weight = submodule.q_proj.ori.weight if 'q' in args.params else submodule.q_proj.weight
                        k_weight = submodule.k_proj.ori.weight if 'k' in args.params else submodule.k_proj.weight
                        v_weight = submodule.v_proj.ori.weight if 'v' in args.params else submodule.v_proj.weight
                        original_mha.in_proj_weight.data.copy_(torch.cat([
                            q_weight.data,
                            k_weight.data,
                            v_weight.data
                        ], dim=0))
                        # original_mha.in_proj_weight.data.copy_(torch.cat([
                        #     submodule.q_proj.weight.data,
                        #     submodule.k_proj.weight.data,
                        #     submodule.v_proj.weight.data
                        # ], dim=0))
                        if submodule.k_proj.bias is not None:
                            q_bias = submodule.q_proj.ori.bias if 'q' in args.params and hasattr(submodule.q_proj.ori, 'bias') else submodule.q_proj.bias
                            k_bias = submodule.k_proj.ori.bias if 'k' in args.params and hasattr(submodule.k_proj.ori, 'bias') else submodule.k_proj.bias
                            v_bias = submodule.v_proj.ori.bias if 'v' in args.params and hasattr(submodule.v_proj.ori, 'bias') else submodule.v_proj.bias

                            original_mha.in_proj_bias.data.copy_(torch.cat([
                                q_bias.data,
                                k_bias.data,
                                v_bias.data
                            ], dim=0))

                        o_weight = submodule.proj.ori.weight if 'o' in args.params else submodule.proj.weight
                        original_mha.out_proj.weight.data.copy_(o_weight.data)

                        o_bias = submodule.proj.ori.bias if 'o' in args.params else submodule.proj.bias
                        if submodule.proj.bias is not None:
                            original_mha.out_proj.bias.data.copy_(o_bias.data)

                        # 合并 LoRA 调整量
                        lora_layer = list_lora_layers[lora_layer_index]
                        if 'q' in args.params and isinstance(lora_layer.q_proj, LinearLoRA):
                            lora_adjustment_q = lora_layer.q_proj.merge_BA('weight') * lora_layer.q_proj.scaling
                            original_mha.in_proj_weight.data[:submodule.embed_dim, :] += lora_adjustment_q
                        if 'k' in args.params and isinstance(lora_layer.k_proj, LinearLoRA):
                            lora_adjustment_k = lora_layer.k_proj.merge_BA('weight') * lora_layer.k_proj.scaling
                            original_mha.in_proj_weight.data[submodule.embed_dim:2*submodule.embed_dim, :] += lora_adjustment_k
                        if 'v' in args.params and isinstance(lora_layer.v_proj, LinearLoRA):
                            lora_adjustment_v = lora_layer.v_proj.merge_BA('weight') * lora_layer.v_proj.scaling
                            original_mha.in_proj_weight.data[2*submodule.embed_dim:, :] += lora_adjustment_v
                        if 'o' in args.params and isinstance(lora_layer.proj, LinearLoRA):
                            lora_adjustment_proj = lora_layer.proj.merge_BA('weight') * lora_layer.proj.scaling
                            original_mha.out_proj.weight.data += lora_adjustment_proj

                        setattr(block, name, original_mha)
                        lora_layer_index += 1

    return clip_model




def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    trainable_params = []
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        if p.requires_grad:
            trainable_params.append(n)

    if bias == 'none':
        pass
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
                if p.requires_grad:
                    trainable_params.append(n)
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
                if m.bias.requires_grad:
                    trainable_params.append(f"{type(m).__name__}.bias")
    else:
        raise NotImplementedError

