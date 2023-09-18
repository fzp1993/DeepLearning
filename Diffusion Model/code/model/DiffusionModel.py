import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model.Unet import UNetModel
from tqdm.auto import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def linear_beta_schedule(beta_range, timesteps):
    beta_start = beta_range[0]
    beta_end = beta_range[1]
    return np.linspace(beta_start, beta_end, timesteps)

class Diffusion_init(nn.Module):
    def __init__(self, 
                 timesteps, 
                 beta_range = [0.0001, 0.02],
                 noise_info = [0, 1]):
        super().__init__()

        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timesteps = timesteps
        self.beta_range = list(map(float,eval(beta_range)))
        self.noise_info = list(map(int,eval(noise_info)))
        self.betas = linear_beta_schedule(self.beta_range, self.timesteps)

        alphas = 1. - self.betas
        alphas_cumprod = np.cumprod(alphas, axis=0) # 不同时间步alpha的连乘
        alphas_cumprod_prev = np.pad(alphas_cumprod[:-1], (1, 0), constant_values=1)

        self.sqrt_recip_alphas = np.sqrt(1. / alphas)
        self.sqrt_alphas_cumprod = torch.from_numpy(np.sqrt(alphas_cumprod)).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.from_numpy(np.sqrt(1. - alphas_cumprod)).to(self.device)
        
        # 方差
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.p2_loss_weight = (1 + alphas_cumprod / (1 - alphas_cumprod)) ** -0.

    def extract(self, a, t, input_shape):
        if torch.is_tensor(a) == False:
            a = torch.from_numpy(a)
        a = a.to(self.device)
        batch = t.shape[0]
        out = a.gather(-1, t)
        x_noisy = out.reshape(batch, *((1,) * (len(input_shape) - 1)))
        return x_noisy
    
    def noise_gen(self, input, noise_info):
        mean = noise_info[0]
        std = noise_info[1]
        return torch.normal(mean, std, size=input.shape)

class Diffusion(Diffusion_init):
    def __init__(self, 
                 timesteps, 
                 beta_range=[0.0002, 0.02], 
                 noise_info=[0, 1]):
        super().__init__(timesteps, 
                         beta_range, 
                         noise_info)

    def diffusion_process(self, origin_input, t, noise = None):
        '''
        加噪过程：
        $\sqrt{\Pord_{t=0}^{t=T}\alpha_t}x_0+\sqrt{\Pord_{t=0}^{t=T}1-\alpha_t}z_t$
        origin_input: 原始的输入，不含有噪声
        t: 第t个时间步
        noise: 噪声数据
        '''
        if noise is None:
            noise = super().noise_gen(origin_input, self.noise_info)
        return (super().extract(self.sqrt_alphas_cumprod, 
                             t, 
                             origin_input.shape) 
                             * origin_input.to(self.device) +
                super().extract(self.sqrt_one_minus_alphas_cumprod, 
                             t, 
                             origin_input.shape) 
                             * noise.to(self.device))
    
    def loss_function(self, denoise_model, origin_input, t, noise=None):
        if noise is None:
            noise = super().noise_gen(origin_input, self.noise_info).to(self.device)
        x_noisy = self.diffusion_process(origin_input=origin_input, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t).to(self.device)

        loss = nn.MSELoss(reduction='none')(noise, predicted_noise)# todo
        # loss = nn.SmoothL1Loss(reduction='none')(noise, predicted_noise)# todo
        # loss = nn.CrossEntropyLoss()(predicted_noise, noise)
        loss = loss.reshape(loss.shape[0], -1)
        loss = loss * super().extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, denoise_model, origin_input, t, noise=None):
        return self.loss_function(denoise_model, origin_input, t, noise=None)
    
class Sampling(Diffusion_init):
    def __init__(self, 
                 timesteps, 
                 beta_range=[0.0001, 0.02], 
                 noise_info=[0, 1]):
        super().__init__(timesteps, 
                         beta_range, 
                         noise_info)
        
    def sampling(self, denoise_model, input_t, t, t_index):
        '''
        噪声预测：
        $x_{t-1}=1/\sqrt{\alpha_t} (x_t-\farc{1-\alpha_t}{\sqrt{1-\Pord_{t=0}^{t=T}\alpha_t}}\epsilon_\theta (x_t,t))+\sigma_t z$
        denoise_model: 噪声预测模型，即Unet
        input_t: 第t时刻的加噪数据
        '''
        t = torch.tensor(t, dtype=torch.int64).to(self.device)
        input_t = input_t.to(self.device)
        betas_t = super().extract(self.betas, t, input_t.shape)
        sqrt_one_minus_alphas_cumprod_t = super().extract(
            self.sqrt_one_minus_alphas_cumprod, 
            t, 
            input_t.shape)
        sqrt_recip_alphas_t = super().extract(self.sqrt_recip_alphas, 
                                           t, 
                                           input_t.shape)
        model_mean = sqrt_recip_alphas_t * (input_t - betas_t 
                                            * denoise_model(input_t, t)
                                            / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        posterior_variance_t = super().extract(self.posterior_variance, 
                                            t, 
                                            input_t.shape)
        noise = super().noise_gen(input_t, self.noise_info)
        return model_mean + torch.sqrt(posterior_variance_t) * noise.to(self.device)

    def forward(self, denoise_model, image_size, batch_size=16, channels=3):
        batch = batch_size
        # 从纯噪声开始
        input = torch.empty((batch_size, channels, image_size[0], image_size[1]))
        img = super().noise_gen(input, self.noise_info)
        imgs = [] # 保存每个时间步的去噪过程

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.sampling(denoise_model, 
                                img, 
                                np.full((batch,), i, dtype='int32'), 
                                i)
            imgs.append(img)
        return imgs

if __name__ == '__main__':
    batch = 2
    channel = 1
    height = 30
    width = 30
    n_feature = 11
    dim_context = 10
    input = torch.rand(batch,channel,height,width)
    # context = torch.rand(batch,n_feature,dim_context)
    print("origin input's shape: \n batch: %d\n channel: %d\n height: %d\n width: %d\n" 
          %(input.shape))
    # print("origin context's shape: \n batch: %d\n n_feature: %d\n dim_context: %d\n" 
    #       %(context.shape))
    model_channels = 64
    num_res_blocks = 1
    attention_resolutions = [1]
    num_heads = 8
    n_embed = 1
    denoise_model = UNetModel(in_channels = channel,
                              model_channels = model_channels,
                              num_res_blocks = num_res_blocks,
                              attention_resolutions = attention_resolutions,
                              dropout=0,
                              channel_mult=(1, 2, 4, 8),
                              conv_resample=True,
                              dims=2,
                              num_classes=0,
                              num_heads=num_heads,
                              num_head_channels=-1,
                              use_scale_shift_norm=False,
                              resblock_updown=True,
                              transformer_depth=4,
                              context_dim=None,
                              n_embed=n_embed
                              )
    model = Sampling(timesteps=200)
    t = np.random.randint(0,200,(batch,))
    t = torch.from_numpy(t)
    y = model(denoise_model,[32,32],16,3)
    print(y)