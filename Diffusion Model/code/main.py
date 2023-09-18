import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from model.DiffusionModel import Diffusion, Sampling
from model.Unet import UNetModel
from utils.params import Params
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run(params):
    train_set = torchvision.datasets.FashionMNIST(
        root = params.dataset_path,  # 将数据保存在本地什么位置
        train=True,  # 我们希望数据用于训练集，其中6万张图片用作训练数据，1万张图片用于测试数据
        download=True,  # 如果目录下没有文件，则自动下载
        transform=transforms.Compose([
            # 改变图片大小，长和宽能被2连续整除，不然Unet会出维度错误
            transforms.Resize((params.input_height,params.input_width)),
            # 将图片转为Tensor类型
            transforms.ToTensor()
        ])  
    )
    train_loader = torch.utils.data.DataLoader(dataset = train_set, 
                                               batch_size = params.batch_size,
                                               shuffle = True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_unet = UNetModel(
        in_channels = params.channel,
        model_channels = params.model_channels,
        num_res_blocks = params.num_res_blocks,
        attention_resolutions = params.attention_resolutions,
        dropout = params.dropout,
        channel_mult = params.channel_mult,
        conv_resample = params.conv_resample,
        dims = params.dims,
        num_classes = params.num_classes,
        num_heads = params.num_heads,
        num_head_channels = params.num_head_channels,
        use_scale_shift_norm = params.use_scale_shift_norm,
        resblock_updown = params.resblock_updown,
        transformer_depth = params.transformer_depth,
        context_dim = params.context_dim,
        n_embed = params.n_embed
    )
    model_diffusion = Diffusion(
        timesteps = params.timesteps,
        beta_range = params.beta_range,
        noise_info = params.noise_info
    )
    model_sampling = Sampling(
        timesteps = params.timesteps,
        beta_range = params.beta_range,
        noise_info = params.noise_info
    )
    optimizer = torch.optim.Adam(model_unet.parameters(), lr=params.learning_rate)
    for epoch in range(params.epochs):
        #=====================train============================
        model_unet.to(device)
        model_unet = torch.nn.DataParallel(model_unet)
        model_unet.train()
        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        for step, (input, label) in loop:
            # input = torch.nn.functional.pad(input,(2,2,2,2),mode='constant', value=0)
            # input_size = input.shape
            input.to(device)
            optimizer.zero_grad()
            t = np.random.randint(0, params.timesteps, (params.batch_size,))
            t = torch.from_numpy(t).to(device)
            loss = model_diffusion(model_unet, input, t)
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{params.epochs}]')
            loop.set_postfix(loss = loss.item())

        #=====================eval============================
        with torch.no_grad():
            model_unet.eval()
            output = model_sampling(
                model_unet,
                [params.input_height,params.input_width],
                params.batch_size,
                params.channel
            )
            path = '{}/{}_gray_image.png'.format(params.save_path,epoch)
            torchvision.utils.save_image(
                output[-1],
                path, 
                nrow=1,
                normalize=True,
                value_range=(0, 1),
                cmap='gray'
            )


if __name__=="__main__":
  
    params = Params()
    config_file = 'config/config_SD.ini'
    params.parse_config(config_file)
    run(params)