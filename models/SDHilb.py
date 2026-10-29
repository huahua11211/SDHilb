import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiScaleHilbert(nn.Module):
    def __init__(self, channels, kernel_sizes, conv_trainable=True):
        super().__init__()
        self.channels = channels
        self.conv_trainable = conv_trainable
        self.conv_layers = nn.ModuleList()
        self.kernel_sizes = kernel_sizes
        self.kernel_weights = nn.Parameter(torch.randn(channels, len(kernel_sizes)))
        
        for ks in kernel_sizes:
            conv = self._create_hilbert_conv(ks)
            self.conv_layers.append(conv)

    def _create_hilbert_conv(self, ks):
       
        ks = int(ks)
        ks = ks + 1 if ks % 2 == 0 else ks
        half = ks // 2
        
        kernel = torch.zeros(ks)
        for i in range(-half, half+1):
            if i != 0:
                kernel[i+half] = 1.0 / (np.pi * i)
        kernel = kernel * torch.hann_window(ks)
        
        conv = nn.Conv1d(
            self.channels, self.channels,
            kernel_size=ks,
            padding=half,
            groups=self.channels,  
            bias=False
        )
        conv.weight.data.copy_(kernel.view(1, 1, -1).repeat(self.channels, 1, 1))
        conv.weight.requires_grad_(self.conv_trainable)
        return conv

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_norm = x - x_mean
        x_var = torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
        x_norm = x_norm / torch.sqrt(x_var)
        
        self.rin_params = (x_mean, x_var)
        
        x_perm = x_norm.permute(0, 2, 1)  # [B, C, L]
        
        weights = F.gumbel_softmax(self.kernel_weights, tau=0.3, dim=-1)  # [C, K]
        
        conv_outputs = []
        for conv in self.conv_layers:
            output = conv(x_perm)  # [B, C, L]
            conv_outputs.append(output.unsqueeze(-1))  # [B, C, L, 1]
        
        all_outputs = torch.cat(conv_outputs, dim=-1)  # [B, C, L, K]
        fused_imag = torch.einsum('bclk,ck->bcl', all_outputs, weights)
        
        return x_norm, fused_imag.permute(0, 2, 1) 


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.enc_in
        self.kernel_sizes = args.kernel_sizes
        
        kernel_sizes = [int(k) for k in args.kernel_sizes.split('_')]
        self.decomp = MultiScaleHilbert(
            channels=self.channels,
            kernel_sizes=kernel_sizes,
            conv_trainable=getattr(args, "conv_trainable", True)
        )
        
        self.Linear_A = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_adj = nn.Linear(self.seq_len, self.pred_len)
        
        nn.init.constant_(self.Linear_A.weight, 1/self.seq_len)  
        nn.init.constant_(self.Linear_adj.weight, 1/self.seq_len)
        

    def forward(self, x):
        real, imag = self.decomp(x)  
        
        real_perm = real.permute(0,2,1)  # [B, C, L]
        imag_perm = imag.permute(0,2,1)   # [B, C, L]

        real_final = self.Linear_adj(real_perm)  + self.Linear_A(imag_perm)  # [B, C, P]
        real_final = real_final.permute(0,2,1)  # [B, P, C]
        x_mean, x_var = self.decomp.rin_params
        final_output = real_final * torch.sqrt(x_var) + x_mean
        
        return final_output