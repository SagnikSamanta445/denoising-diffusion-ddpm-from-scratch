"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    betas = torch.linspace(beta_start, beta_end, steps = T, dtype=torch.float32)

    return betas

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    alphas = 1 - betas

    return alphas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    cumprod = torch.cumprod(alphas, dim=0)
    return cumprod

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    k = torch.gather(a, dim = 0, index=t).reshape(-1,1,1,1)
    return k

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    bar_alpha_t = extract_into_batch(alphas_cumprod, t, x0)

    x_t = torch.sqrt(bar_alpha_t)*x0 + torch.sqrt(1- bar_alpha_t)*noise

    return x_t

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    
    betas = linear_beta_schedule(T, beta_start, beta_end)
    alphas = alphas_from_betas(betas)
    alphas_cumprod = cumprod_alphas(alphas)

    s = {
        "betas" : betas,
        "alphas" : alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod" : torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod" :  torch.sqrt(1 - alphas_cumprod),
        "T" : T
    } 
    return s

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    diff = noise - noise_pred
    return torch.square(diff).mean()

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
     
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    noise_pred = model(x_t, t)
    loss = noise_prediction_loss(noise_pred, noise)

    return loss

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    
    half = dim // 2

    i = torch.arange(half, dtype=float)

    freqs = 1.0 / (10000 ** (i/half))

    #                 [B x 1] * [1, half]      
    args = t.float()[:, None] * freqs[None, :]

    emb_sin = torch.sin(args)
    emb_cos = torch.cos(args)

    #   [B X dim]
    emb = torch.cat([emb_sin, emb_cos], dim = 1)

    return emb

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    
    torch.manual_seed(seed)

    def init_weight(shape):
        w = torch.empty(shape)
        torch.nn.init.normal_(w,0.0,0.02)
        return w.requires_grad_()

    def init_bias(shape):
        return torch.zeros(shape, requires_grad = True)
    
    conv_in_w = init_weight([hidden,in_ch,3,3])
    conv_in_b = init_bias([hidden])
    
    time_mlp_w = init_weight([hidden,time_dim])
    time_mlp_b = init_bias([hidden])

    conv_mid_w = init_weight([hidden,hidden,3,3])
    conv_mid_b = init_bias([hidden])

    conv_out_w = init_weight([in_ch,hidden,3,3])
    conv_out_b = init_bias([in_ch])

    return {
        "conv_in_w" : conv_in_w,
        "conv_in_b" : conv_in_b,
        "time_mlp_w" : time_mlp_w,
        "time_mlp_b" : time_mlp_b,
        "conv_mid_w" : conv_mid_w,
        "conv_mid_b" : conv_mid_b,
        "conv_out_w" : conv_out_w,
        "conv_out_b" : conv_out_b
    }

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
   
    # step 1 (h : (B, hidden, H, W))
    h = F.conv2d(x, params['conv_in_w'], params['conv_in_b'],padding=1)

    # step 2 
    temb = timestep_embedding(t, params['time_mlp_w'].shape[1])
    temb = temb.to(dtype=params['time_mlp_w'].dtype)
    temb = F.relu(F.linear(temb, params['time_mlp_w'], params['time_mlp_b']))
    
    #broadcasting(temp : (B, hidden, 1, 1))
    h = h + temb[:, :, None, None]

    # step 3
    h = F.relu(h)
    h = F.relu(F.conv2d(h, params['conv_mid_w'], params['conv_mid_b'],padding=1))

    # step 4
    return F.conv2d(h, params['conv_out_w'], params['conv_out_b'],padding=1)

# Step 12 - make_blob_dataset
import torch


def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    x = torch.zeros(n, 1, size, size)

    radius = size // 4

    torch.manual_seed(seed)

    for i in range(n):
        cy, cx = torch.randint(radius, size - radius, (2,))

        # Create a grid of pixel coordinates
        y, x_coord = torch.meshgrid(
            torch.arange(size),
            torch.arange(size),
            indexing="ij"
        )

        dist_sq = (y - cy) ** 2 + (x_coord - cx) ** 2

        x[i, 0] = (dist_sq <= radius ** 2).float()

    return x

# Step 13 - ddpm_train_step (not yet solved)
# TODO: implement

# Step 14 - train_ddpm (not yet solved)
# TODO: implement

# Step 15 - predict_x0_from_eps (not yet solved)
# TODO: implement

# Step 16 - ddpm_p_mean_variance (not yet solved)
# TODO: implement

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

