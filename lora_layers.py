#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from IPython import embed

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result

class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

# class Conv2d(ConvLoRA):
#     def __init__(self, *args, **kwargs):
#         super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # print("in init")
        # embed()
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True): # True for train and False for eval
 
        nn.Conv2d.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = False
        else:
            # print("test")
            # embed()
            if self.merge_weights and not self.merged:
                # print("merging")
                # embed()
                # Merge the weights and mark it
                self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):

        if self.r > 0 and not self.merged:

            return F.conv2d(
                x, 
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        
        return nn.Conv2d.forward(self, x)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# Can Extend to other ones like this

# class Conv3d(ConvLoRA):
#     def __init__(self, *args, **kwargs):
#         super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)

# class Conv3d(nn.Conv3d, LoRALayer):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size,
#         r: int = 0,
#         lora_alpha: int = 1,
#         lora_dropout: float = 0.,
#         merge_weights: bool = True,
#         **kwargs
#     ):
#         nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha,
#                            lora_dropout=lora_dropout,
#                            merge_weights=merge_weights)

#         # Normalize kernel_size to (kD, kH, kW)
#         if isinstance(kernel_size, int):
#             k_d = k_h = k_w = kernel_size
#         else:
#             # Expect a 3-tuple
#             k_d, k_h, k_w = kernel_size

#         k_total = k_d * k_h * k_w

#         if r > 0:
#             # Shapes analogous to your Conv2d, but with full 3D kernel flattened
#             self.lora_A = nn.Parameter(
#                 self.weight.new_zeros((r * k_total, in_channels * k_total))
#             )
#             self.lora_B = nn.Parameter(
#                 self.weight.new_zeros((out_channels * k_total, r * k_total))
#             )
#             self.scaling = self.lora_alpha / self.r
#             # Freeze base conv weights
#             self.weight.requires_grad = False

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.Conv3d.reset_parameters(self)
#         if hasattr(self, 'lora_A'):
#             nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B)

#     def train(self, mode: bool = True):
#         nn.Conv3d.train(self, mode)
#         if mode:
#             if self.merge_weights and self.merged:
#                 if self.r > 0:
#                     # Un-merge
#                     self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
#                 self.merged = False
#         else:
#             if self.merge_weights and not self.merged:
#                 self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
#                 self.merged = True

#     def forward(self, x: torch.Tensor):
#         if self.r > 0 and not self.merged:
#             return F.conv3d(
#                 x,
#                 self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
#                 self.bias,
#                 self.stride,
#                 self.padding,
#                 self.dilation,
#                 self.groups
#             )
#         return nn.Conv3d.forward(self, x)

class Conv3d(nn.Module, LoRALayer):
    """
    Proper 3D Conv LoRA:
    - Base conv: nn.Conv3d(in_channels, out_channels, kernel_size, ...)
    - LoRA applied to the flattened kernel: W ∈ R[out, in * kT * kH * kW]
    - ΔW = B @ A with A ∈ R[r, in*kT*kH*kW], B ∈ R[out, r]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Module.__init__(self)
        # Base conv layer
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)

        # LoRA metadata
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        # Only create LoRA params if r > 0
        if r > 0:
            # Get spatial kernel size from weight shape to support any (kT, kH, kW)
            kT, kH, kW = self.conv.weight.shape[2:]
            # print("kT, kH, kW:", kT, kH, kW)
            patch_size = kT * kH * kW

            in_eff = in_channels * patch_size   # flattened input per filter
            out_eff = out_channels              # output channels

            # A: [r, in_eff], B: [out_eff, r]
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r, in_eff))
            )
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros((out_eff, r))
            )

            self.scaling = self.lora_alpha / self.r

            # Freeze the pretrained conv weights
            self.conv.weight.requires_grad = False

        self.reset_parameters()
        # Track whether we have merged ΔW into conv.weight
        self.merged = False

    def reset_parameters(self):
        # Reset base conv
        self.conv.reset_parameters()
        # Reset LoRA params if they exist
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def _delta_weight(self):
        """
        Compute ΔW with shape matching self.conv.weight: [out, in, kT, kH, kW].
        """
        if self.r == 0:
            return 0.0

        kT, kH, kW = self.conv.weight.shape[2:]
        patch_size = kT * kH * kW

        in_eff = self.conv.in_channels * patch_size
        out_eff = self.conv.out_channels

        # B @ A → [out_eff, in_eff]
        delta_flat = self.lora_B @ self.lora_A       # (out, in*K)
        # Reshape to conv kernel shape
        delta_w = delta_flat.view(
            out_eff,
            self.conv.in_channels,
            kT,
            kH,
            kW,
        )
        return delta_w

    def train(self, mode: bool = True):
        """
        Handles weight merging / unmerging just like the Linear/Conv2d LoRA:
        - In eval mode and merge_weights=True → merge ΔW into conv.weight once
        - In train mode → unmerge if previously merged
        """
        super(Conv3d, self).train(mode)
        if mode:
            # Switch to train: unmerge if needed
            if self.merge_weights and self.merged and self.r > 0:
                self.conv.weight.data -= self._delta_weight() * self.scaling
                self.merged = False
        else:
            # Switch to eval: merge if needed
            if self.merge_weights and not self.merged and self.r > 0:
                self.conv.weight.data += self._delta_weight() * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            # On-the-fly LoRA without permanently modifying conv.weight
            delta_w = self._delta_weight() * self.scaling
            weight = self.conv.weight + delta_w
            return F.conv3d(
                x,
                weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )
        else:
            # Either no LoRA or already merged
            return self.conv(x)