#  ------------------------------------------------------------------------------------------
#  This code is reconstructed based on loralib (https://github.com/microsoft/LoRA) by Baijiong Lin.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

def set_param(curr_mod, name, param=None, mode='update'):
    r"""Refer to https://github.com/Baijiong-Lin/MOML/blob/main/MTL/utils.py"""
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p
import torch
import torch.nn as nn

class LoRALayer:
    def __init__(self, r, lora_alpha, fan_in_fan_out=False, dropout_rate=0):
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout_rate
        if self.r > 0:
            self.scaling = self.lora_alpha / self.r
        self.merged = False
        self.fan_in_fan_out = fan_in_fan_out
        self.params_with_lora = {}
        self.rank_activation_counts = torch.zeros(r)
        self.lora_router = None

    def register_lora_param(self, user_adaw, in_features):
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f'self.{param_name}').size()) == 2
            self.register_parameter(f'{lora_name}_lora_A',
                                    nn.Parameter(eval(f'self.{param_name}').new_zeros(
                                        (self.r, eval(f'self.{param_name}').size()[1])))
                                    )
            self.register_parameter(f'{lora_name}_lora_B',
                                    nn.Parameter(eval(f'self.{param_name}').new_zeros(
                                        (eval(f'self.{param_name}').size()[0], self.r)))
                                    )
            eval(f'self.{param_name}').requires_grad = False

        if self.r > 0:
            self.lora_router = nn.Linear(in_features, self.r)

    def init_lora_param(self, use_pissa=False, use_adaw=0):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f'{lora_name}_lora_A'):
                if use_pissa:
                    pass
                else:
                    nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A'), a=math.sqrt(5))
                    nn.init.zeros_(eval(f'self.{lora_name}_lora_B'))

    def transpose(self, w: torch.Tensor):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge_BA(self, param_name: str,merge_k=4):
        lora_name = self.params_with_lora[param_name]
        A = eval(f'self.{lora_name}_lora_A')
        B = eval(f'self.{lora_name}_lora_B')
        _, topk_indices = torch.topk(self.rank_activation_counts, merge_k)
        topk_mask = torch.zeros_like(self.rank_activation_counts, dtype=torch.bool)
        topk_mask[topk_indices] = True
        A_active = A[topk_mask]
        B_active = B[:, topk_mask]
        adjusted_A=A_active
        delta_W = B_active @ adjusted_A
        full_delta_W = torch.zeros_like(eval(f'self.{param_name}'))
        full_delta_W= delta_W

        return self.transpose(full_delta_W.view(eval(f'self.{param_name}').shape))
    def merge_lora_param(self, activation_threshold=0):
        for param_name, lora_name in self.params_with_lora.items():
            p = getattr(self, param_name)
            p_new = p.detach() + self.merge_BA(param_name, activation_threshold=activation_threshold) * self.scaling
            setattr(self, param_name, p_new)

    def add_lora_data(self, activation_threshold=0):
        for param_name, lora_name in self.params_with_lora.items():
            getattr(self, param_name).data += self.merge_BA(param_name, activation_threshold=activation_threshold) * self.scaling

    def sub_lora_data(self, activation_threshold=0):
        for param_name, lora_name in self.params_with_lora.items():
            getattr(self, param_name).data -= self.merge_BA(param_name, activation_threshold=activation_threshold) * self.scaling

    def lora_train(self, mode: bool = True):
        if mode:
            if self.merged and self.r > 0:
                self.sub_lora_data()
            self.merged = False
        else:
            if not self.merged and self.r > 0:
                self.add_lora_data()
            self.merged = True


class LinearLoRA(nn.Linear, LoRALayer):
    def __init__(
            self,
            existing_linear: nn.Linear,
            r: int = 0,
            lora_alpha: int = 1,
            fan_in_fan_out: bool = False,
            dropout_rate=0.,
            use_pissa: bool = 0,
            use_adaw: bool = 1,
            router_threshold=0.05,
            **kwargs
    ):
        super().__init__(
            in_features=existing_linear.in_features,
            out_features=existing_linear.out_features
        )
        self.ori = nn.Linear(
            in_features=existing_linear.in_features,
            out_features=existing_linear.out_features
        )
        self.ori.weight.data.copy_(existing_linear.weight.data.clone())
        if existing_linear.bias is not None:
            self.ori.bias.data.copy_(existing_linear.bias.data.clone())
        for param in self.ori.parameters():
            param.requires_grad = False
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, fan_in_fan_out=fan_in_fan_out,
                           dropout_rate=dropout_rate)
        self.use_pissa = use_pissa
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.register_lora_param(use_adaw, existing_linear.in_features)
        self.init_lora_param(use_pissa=use_pissa, use_adaw=use_adaw)
        self.weight.data = self.transpose(self.weight.data)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.router_threshold = router_threshold

    def train(self, mode: bool = True):
        super().train(mode)
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):
        original_output = nn.Linear.forward(self.ori, x)
        flag=0

        if x.dim()==2:
            len_=x.shape[0]
            if len_%197==0:
                bs=len_//197
            else:
                bs=len_//77

            flag=1
            fea=x.shape[1]
            x = x.view(-1, bs, fea)


        if self.training and self.dropout.p > 0:
            x = self.dropout(x)
        if self.r > 0 and not self.merged:
            lora_name = self.params_with_lora['weight']
            cls_output = x[0].clone()  # (batch_size, feature)
            router_scores = self.lora_router(cls_output)
            router_probs = torch.softmax(router_scores, dim=-1)
            #print(router_probs)
            #============================
            k = 8
            _, topk_indices = torch.topk(router_probs, k=k, dim=-1)
            active_ranks = torch.zeros_like(router_probs, dtype=torch.bool)
            batch_indices = torch.arange(router_probs.shape[0]).unsqueeze(-1)
            active_ranks[batch_indices,topk_indices] = True
            batch_active_counts = active_ranks.sum(dim=0)
            batch_active_counts = batch_active_counts.to(self.rank_activation_counts.device)
            assert batch_active_counts.shape == self.rank_activation_counts.shape, f"Shape mismatch: {batch_active_counts.shape} vs {self.rank_activation_counts.shape}"
            self.rank_activation_counts += batch_active_counts
             
            # total_activations = batch_active_counts.sum()
            # frequencies = batch_active_counts / total_activations

            # sorted_frequencies, sorted_indices = torch.sort(frequencies, descending=True)
            # #print(sorted_frequencies)
            # cumulative_frequency = 0
            # selected_count = 0
            # for i in range(len(sorted_frequencies)):
            #     cumulative_frequency += sorted_frequencies[i]
            #     selected_count += 1
            #     if cumulative_frequency >= 0.9:
            #         break

            # topk_indices = sorted_indices[:selected_count]
            topk_mask = torch.zeros_like(batch_active_counts, dtype=torch.bool)
            topk_mask[topk_indices] = True
            #============================================================



            
            _, topk_indices = torch.topk(batch_active_counts,k)
            topk_mask = torch.zeros_like(batch_active_counts, dtype=torch.bool)
            topk_mask[topk_indices] = True

            current_active_ranks = topk_mask
            A_active = eval(f'self.{lora_name}_lora_A')[current_active_ranks]
            B_active = eval(f'self.{lora_name}_lora_B')[:, current_active_ranks]

            adjusted_A=A_active
            delta_W = B_active @ adjusted_A
            full_delta_W = torch.zeros_like(eval(f'self.weight'))
            full_delta_W = delta_W

            current_lora_adjustment = torch.matmul(x, self.transpose(full_delta_W).transpose(0, 1)) * self.scaling

                
            lora_adjustment=current_lora_adjustment
            if flag:
                lora_adjustment=current_lora_adjustment.view(-1,current_lora_adjustment.shape[2])

            result = original_output + lora_adjustment
        else:
            result = original_output
        return result

class PlainMultiheadAttentionLoRA(nn.Module):
    def __init__(
            self,
            existing_mha: nn.MultiheadAttention,
            enable_lora: list = ['q', 'k', 'v', 'o'],
            r: int = 0, 
            lora_alpha: int = 1, 
            dropout_rate:float = 0.,
            **kwargs
        ):
        super().__init__()
        
        self.dropout = 0 # this module is not used to retrain the main block
        self.embed_dim = existing_mha.embed_dim
        self.kdim = existing_mha.kdim
        self.vdim = existing_mha.vdim
        self._qkv_same_embed_dim = existing_mha._qkv_same_embed_dim
        self.num_heads = existing_mha.num_heads
        self.batch_first = existing_mha.batch_first
        self.head_dim = existing_mha.head_dim
        #self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=existing_mha.in_proj_bias is not None)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None)

        # Initialize parameters
        with torch.no_grad():
            
            # Extract the existing weights and biases
            existing_weight = existing_mha.in_proj_weight.data
            existing_bias = existing_mha.in_proj_bias.data if existing_mha.in_proj_bias is not None else None

            # Initialize q_proj
            self.q_proj.weight.data.copy_(existing_weight[:self.embed_dim, :])
            if existing_bias is not None:
                self.q_proj.bias.data.copy_(existing_bias[:self.embed_dim])

            # Initialize k_proj
            self.k_proj.weight.data.copy_(existing_weight[self.embed_dim:2*self.embed_dim, :])
            if existing_bias is not None:
                self.k_proj.bias.data.copy_(existing_bias[self.embed_dim:2*self.embed_dim])

            # Initialize v_proj
            self.v_proj.weight.data.copy_(existing_weight[2*self.embed_dim:, :])
            if existing_bias is not None:
                self.v_proj.bias.data.copy_(existing_bias[2*self.embed_dim:])

            # Initialize proj
            self.proj.weight.data.copy_(existing_mha.out_proj.weight.data)
            if self.proj.bias is not None:
                self.proj.bias.data.copy_(existing_mha.out_proj.bias.data)

        self.scaled_dot_product_attention = F.scaled_dot_product_attention
        

        
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)
        # Init qkv as a new lora linear layer 
        for item in enable_lora:
            if item == 'q':
                self.q_proj = LinearLoRA(self.q_proj,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate)
            elif item == 'k':
                self.k_proj = LinearLoRA(self.k_proj,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate)
            elif item == 'v':
                self.v_proj = LinearLoRA(self.v_proj,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate)
            elif item == 'o':
                self.proj = LinearLoRA(self.proj,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate)



        
    def forward_module(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False):

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        """
        E = query.size(-1)
        qkv = self.qkv(query)
        qkv = qkv.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        """
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None  

    def train(self, mode: bool = True):
        super().train(mode)
        #self.lora_train(mode)  

    def forward(self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            **kwargs):
        

        return self.forward_module(query, key, value, **kwargs) 
        
