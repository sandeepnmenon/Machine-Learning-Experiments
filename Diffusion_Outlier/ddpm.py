import copy
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules import EMA
from modules import UNet_conditional as UNet
from utils import get_data, save_images, setup_logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s: %(message)s', 
    level=logging.INFO, 
    datefmt='%I:%M:%S'
)

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.01, img_size=64, device='cuda') -> None:
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
         
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e
    
    def sample_timesteps(self, n):
        return torch.randint(0, self.noise_steps, (n,), device=self.device)
    
    def sample(self, model, n, labels, cgf_scale=3):
        logging.info(f'Sampling {n} new images...')
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n)*i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cgf_scale > 0:
                    unconditional_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(unconditional_predicted_noise, predicted_noise, cgf_scale)

                alpha = self.alpha[i][:, None, None, None]
                alpha_hat = self.alpha_hat[i][:, None, None, None]
                beta = self.beta[i][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = 1.0 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + \
                    torch.sqrt(beta) * noise
                    
            model.train()
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
            
            return x
        
        
def train(args):
    setup_logging(args.run_name)
    device = torch.device(args.device)
    dataloader = get_data(args)
    model = UNet(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join('runs', args.run_name))
    data_len = len(dataloader)
    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model.eval().requires_grad_(False))
    
    for epoch in range(args.epochs):
        logging.info(f'Epoch {epoch+1}/{args.epochs}')
        pbar = tqdm(dataloader, position=0)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.rand() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar('MSE', loss.item(), global_step=epoch*data_len + i)
        
         
        if epoch % 10 == 0:
            labels = torch.arange(args.num_classes).long().to(device)
            sampled_images = diffusion.sample(model, n = images.shape[0], labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n = images.shape[0], labels=labels)
            save_images(sampled_images, os.path.join('results', args.run_name, f'{epoch+1}.png'), nrow=8)
            save_images(ema_sampled_images, os.path.join('results', args.run_name, f'ema_{epoch+1}.png'), nrow=8)
            torch.save(model.state_dict(), os.path.join('models', args.run_name, f'ckpt_{epoch+1}.pt'))
            torch.save(ema_model.state_dict(), os.path.join('models', args.run_name, f'ema_ckpt_{epoch+1}.pt'))
        
def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = 'ddpm_unconditional'
    args.epoch = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = 'data'
    args.device = 'cuda'
    args.lr = 1e-4
    train(args)
    
if __name__ == '__main__':
    launch()