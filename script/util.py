import os
from omegaconf import OmegaConf
import math
import numpy as np
import cv2
from einops import rearrange
import torch
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as SSIM
import json
import glob

def get_immediate_subfolders(directory):
    # 列出一级子文件夹
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    subfolders.sort()
    return subfolders

def get_input_ndarray(image_path):
    original_ndarray = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # 0~255 HWX UINT8
    if len(original_ndarray.shape) == 2:
        gray_ndarray = original_ndarray
        bgr_ndarray = cv2.cvtColor(gray_ndarray, cv2.COLOR_GRAY2BGR) # 0~255 HWC BGR UINT8
        rgb_ndarray = cv2.cvtColor(gray_ndarray, cv2.COLOR_GRAY2RGB) # 0~255 HWC RGB UINT8
    else:
        bgr_ndarray = original_ndarray # 0~255 HWC BGR UINT8
        rgb_ndarray = cv2.cvtColor(bgr_ndarray, cv2.COLOR_BGR2RGB) # 0~255 HWC RGB UINT8
        gray_ndarray = cv2.cvtColor(bgr_ndarray, cv2.COLOR_BGR2GRAY) # 0~255 HW1 GRAY UINT8
    return rgb_ndarray, gray_ndarray

def get_files_with_format(directory='', file_format='.jpg', drop_first=False):
    subfolder_paths = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    subfolder_paths.sort()
    file_paths = []
    for subfolder_path in subfolder_paths:
        file_paths_ = glob.glob(os.path.join(directory, subfolder_path, 'rgb', '*' + file_format))
        file_paths_.sort()
        if drop_first == True:
            file_paths_ = file_paths_[1:]
        file_paths = file_paths + file_paths_
    return file_paths

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(imgA, imgB, gray_scale=False):
    if len(imgA.shape) == 2:
        score, diff = SSIM(imgA, imgB, full=True, channel_axis=None)
    elif len(imgA.shape) == 3:
        score, diff = SSIM(imgA, imgB, full=True, channel_axis=2)
    return score

def save_image_from_chw_tensor(tensor=None, path='./test.png'):
    tensor = (((tensor + 1.0)/2.0).clamp(0, 1))*255.0
    image = np.transpose(tensor.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    cv2.imwrite(path, image[:, :, ::-1])

def create_decay_tensor(N, S, small_value=1e-6):
    # 计算lambda，使得在S个点之后值接近small_value
    T = S - 1
    lambda_val = -torch.log(torch.tensor(small_value / N)) / T

    # 创建时间步的张量
    t = torch.arange(0, S)

    # 计算指数衰减值
    decay_values = N * torch.exp(-lambda_val * t)

    # 将最后一个值设为0
    decay_values = torch.cat((decay_values, torch.tensor([0.0])))

    return decay_values

def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)
    return conf

def dump_config(path, config):
    with open(path, 'w') as fp:
        OmegaConf.save(config=config, f=fp)

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents

def reconstruct_R(D, x, R_x, ref, F):
    # 初始化R序列
    R = ref.clone()
    
    # 将已知的R[:, x, :, :, :]的值赋给正确的索引位置
    R[:, x, :, :, :] = R_x
    
    # 向前递推
    for i in range(x-1, -1, -1):
        R[:, i, :, :, :] = R[:, i+1, :, :, :] - D[:, i+1, :, :, :]
    
    # 向后递推
    for i in range(x+1, F):
        R[:, i, :, :, :] = R[:, i-1, :, :, :] + D[:, i, :, :, :]
    
    return R

def normalize_and_save_tensor(tensor, save_path):
    assert tensor.ndimension() == 2, "输入张量必须是尺寸为 (h, w) 的二维张量"

    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    
    to_pil = transforms.ToPILImage()
    image = to_pil(normalized_tensor.unsqueeze(0))
    
    image.save(save_path)

def feature2image(feature):
    return (255 * (0.5 + feature.detach().cpu().numpy().clip(-1, 1)/2)).astype(np.uint8)

def tensor2image(tensor, t_min=None, t_max=None):
    if t_min == None and t_max == None:
        t_min = tensor.min()
        t_max = tensor.max()
    normalized_tensor = (tensor - t_min) / (t_max - t_min)
    normalized_tensor = (normalized_tensor * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return normalized_tensor

def encode_and_decode(values, vae, config, clamp_range=[]):
    num = values.shape[1]
    values = rearrange(values, "b f c h w -> (b f) c h w")
    # 
    posteriors = vae.encode(values).latent_dist
    if config.vae_mode == 'sample':
        latents = posteriors.sample()
    elif config.vae_mode == 'mode':
        latents = posteriors.mode()
    else:
        raise ValueError
    posteriors.logvar = rearrange(posteriors.logvar, "(b f) c h w -> b f c h w", f=num)
    posteriors.mean = rearrange(posteriors.mean, "(b f) c h w -> b f c h w", f=num)
    posteriors.parameters = rearrange(posteriors.parameters, "(b f) c h w -> b f c h w", f=num)
    posteriors.std = rearrange(posteriors.std, "(b f) c h w -> b f c h w", f=num)
    posteriors.var = rearrange(posteriors.var, "(b f) c h w -> b f c h w", f=num)
    decodes = []
    for i in range(0, latents.shape[0], config.decode_chunk_size):
        decodes.append(vae.decode(latents[i:i + config.decode_chunk_size, :, :, :], num_frames=latents[i:i + config.decode_chunk_size, :, :, :].shape[0]).sample)
    decodes = torch.cat(decodes, dim=0)
    decodes = rearrange(decodes, "(b f) c h w -> b f c h w", f=num)
    if len(clamp_range) == 2:
        decodes = torch.clamp(decodes, min=clamp_range[0], max=clamp_range[1])
    # 
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=num)
    values = rearrange(values, "(b f) c h w -> b f c h w", f=num)
    return values, posteriors, latents, decodes

def encode(values, vae, config):
    num = values.shape[1]
    values = rearrange(values, "b f c h w -> (b f) c h w")
    # 
    posteriors = vae.encode(values).latent_dist
    if config.vae_mode == 'sample':
        latents = posteriors.sample()
    elif config.vae_mode == 'mode':
        latents = posteriors.mode()
    else:
        raise ValueError
    posteriors.logvar = rearrange(posteriors.logvar, "(b f) c h w -> b f c h w", f=num)
    posteriors.mean = rearrange(posteriors.mean, "(b f) c h w -> b f c h w", f=num)
    posteriors.parameters = rearrange(posteriors.parameters, "(b f) c h w -> b f c h w", f=num)
    posteriors.std = rearrange(posteriors.std, "(b f) c h w -> b f c h w", f=num)
    posteriors.var = rearrange(posteriors.var, "(b f) c h w -> b f c h w", f=num)
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=num)
    values = rearrange(values, "(b f) c h w -> b f c h w", f=num)
    return values, posteriors, latents

def decode(latents, vae, config, clamp_range=[]):
    num = latents.shape[1]
    latents = rearrange(latents, "b f c h w -> (b f) c h w")
    decodes = []
    for i in range(0, latents.shape[0], config.decode_chunk_size): 
        decodes.append(vae.decode(latents[i:i + config.decode_chunk_size, :, :, :], num_frames=latents[i:i + config.decode_chunk_size, :, :, :].shape[0]).sample)
    decodes = torch.cat(decodes, dim=0)
    decodes = rearrange(decodes, "(b f) c h w -> b f c h w", f=num)
    if len(clamp_range) == 2:
        decodes = torch.clamp(decodes, min=clamp_range[0], max=clamp_range[1])
    # 
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=num)
    return latents, decodes

def get_feature_from_encoder_forward(encoder, sample):
    frame_num = sample.shape[1]
    sample = rearrange(sample, "b f c h w -> (b f) c h w")
    feature_list = []
    feature_list.append(rearrange(sample, "(b f) c h w -> b f c h w", f=frame_num))

    sample = encoder.conv_in(sample)
    feature_list.append(rearrange(sample, "(b f) c h w -> b f c h w", f=frame_num))

    for down_block in encoder.down_blocks:
        sample = down_block(sample)
        feature_list.append(rearrange(sample, "(b f) c h w -> b f c h w", f=frame_num))

    # middle
    sample = encoder.mid_block(sample)
    feature_list.append(rearrange(sample, "(b f) c h w -> b f c h w", f=frame_num))

    # post-process
    sample = encoder.conv_norm_out(sample)
    feature_list.append(rearrange(sample, "(b f) c h w -> b f c h w", f=frame_num))
    sample = encoder.conv_act(sample)
    feature_list.append(rearrange(sample, "(b f) c h w -> b f c h w", f=frame_num))
    sample = encoder.conv_out(sample)
    feature_list.append(rearrange(sample, "(b f) c h w -> b f c h w", f=frame_num))
    
    return feature_list

def process_event(event_values, network, config):
    if config.event_processing_pipeline == 'default':
        event_values, event_posteriors, event_latents, event_decodes = encode_and_decode(values=event_values, vae=network['vae_event'], config=config)
    elif config.event_processing_pipeline == 'pure_vit':
        event_num = event_values.shape[1]
        event_values = rearrange(event_values, "b f c h w -> (b f) c h w")
        event_latents = network['vit_event'](event_values)
        event_latents = rearrange(event_latents, "(b f) c h w -> b f c h w", f=event_num)
        event_values = rearrange(event_values, "(b f) c h w -> b f c h w", f=event_num)
        event_decodes = event_values.clone()
        event_posteriors = None
    elif config.event_processing_pipeline == 'pre_vit':
        event_num = event_latents.shape[1]
        event_latents = rearrange(event_latents, "b f c h w -> (b f) c h w")
        event_latents = network['vit_event'](event_latents)
        event_latents = rearrange(event_latents, "(b f) c h w -> b f c h w", f=event_num)
        event_latents, event_decodes = decode(latents=event_latents, vae=network['vae_event'], config=config)
    return event_values, event_posteriors, event_latents, event_decodes

def process_pipeline(config, inputs, networks):
    losses = {}
    tensors = {}
    
    for pipeline in config.pipelinechain:
        if pipeline['name'] in ['event_to_redisual_rgb_latent', 'redisual_rgb_latent_prediction', 'residual_of_latent_residual', 'residual_of_latent_residual_with_kld', 'residual_of_latent_residual_with_kld_with_stochastic']:
            with torch.no_grad():
                rgb_values, rgb_posteriors, rgb_latents = encode(values=inputs['rgb_values'], vae=networks['vae_rgb'], config=config)
                gray_values, gray_posteriors, gray_latents = encode(values=inputs['gray_values'], vae=networks['vae_rgb'], config=config)
                event_values, event_posteriors, event_latents = encode(values=inputs['event_values'], vae=networks['vae_event'], config=config)
        elif pipeline['name'] in ['event_to_redisual_rgb_latent_with_resnet']:
            with torch.no_grad():
                rgb_values, rgb_posteriors, rgb_latents = encode(values=inputs['rgb_values'], vae=networks['vae_rgb'], config=config)
        
        if pipeline['name'] in ['event_to_redisual_rgb_latent', 'redisual_rgb_latent_prediction']:
            start_rgb_latents = rgb_latents[:, :-1, :, :, :]
            end_rgb_latents = rgb_latents[:, 1:, :, :, :]
            if pipeline['name'] == ['event_to_redisual_rgb_latent']:
                input_tensor = inputs['event_values']
            elif pipeline['name'] == ['redisual_rgb_latent_prediction']:
                input_tensor = torch.cat([start_rgb_latents, event_latents], dim=2)
            frame_num = input_tensor.shape[1]
            input_tensor = rearrange(input_tensor, "b f c h w -> (b f) c h w")
            output_tensor = networks['vit'](input_tensor)
            output_tensor = rearrange(output_tensor, "(b f) c h w -> b f c h w", f=frame_num)
            pd_residual_rgb_latents = output_tensor
            gt_residual_rgb_latents = end_rgb_latents - start_rgb_latents
            
            for criterion in pipeline['criterionchain']:
                if criterion['name'] == 'l1_loss':
                    criterion_value = torch.mean(torch.abs(pd_residual_rgb_latents - gt_residual_rgb_latents))
                elif criterion['name'] == 'l2_loss':
                    criterion_value = torch.nn.functional.mse_loss(pd_residual_rgb_latents, gt_residual_rgb_latents, reduction="mean")
                losses[pipeline['name'] + '/' + criterion['name']] = {
                    'value': criterion_value,
                    'lambda': criterion['lambda']
                }
            tensors.update(
                {
                    'start_rgb_latents': start_rgb_latents,
                    'end_rgb_latents': end_rgb_latents,
                    'event_latents': event_latents,
                    'pd_residual_rgb_latents': pd_residual_rgb_latents,
                    'gt_residual_rgb_latents': gt_residual_rgb_latents,
                }
            )
        elif pipeline['name'] in ['residual_of_latent_residual']:
            start_rgb_latents = rgb_latents[:, :-1, :, :, :]
            end_rgb_latents = rgb_latents[:, 1:, :, :, :]
            start_gray_latents = gray_latents[:, :-1, :, :, :]
            end_gray_latents = gray_latents[:, 1:, :, :, :]
            residual_rgb_latents = end_rgb_latents - start_rgb_latents
            residual_gray_latents = end_gray_latents - start_gray_latents
            gt_residual_of_latent_residual = residual_rgb_latents - residual_gray_latents
            # 
            input_tensor = torch.cat([start_gray_latents, end_gray_latents, residual_gray_latents, event_latents], dim=2)
            frame_num = input_tensor.shape[1]
            input_tensor = rearrange(input_tensor, "b f c h w -> (b f) c h w")
            output_tensor = networks['vit'](input_tensor)
            output_tensor = rearrange(output_tensor, "(b f) c h w -> b f c h w", f=frame_num)
            pd_residual_of_latent_residual = output_tensor.contiguous()
            # 
            for criterion in pipeline['criterionchain']:
                if criterion['name'] == 'l1_loss':
                    criterion_value = torch.mean(torch.abs(pd_residual_of_latent_residual - gt_residual_of_latent_residual))
                elif criterion['name'] == 'l2_loss':
                    criterion_value = torch.nn.functional.mse_loss(pd_residual_of_latent_residual, gt_residual_of_latent_residual, reduction="mean")
                elif criterion['name'] == 'sigmoid_l1_loss':
                    criterion_value = torch.nn.functional.mse_loss(pd_residual_of_latent_residual, torch.sigmoid(gt_residual_of_latent_residual), reduction="mean")
                losses[pipeline['name'] + '/' + criterion['name']] = {
                    'value': criterion_value,
                    'lambda': criterion['lambda']
                }
            # 
            tensors.update(
                {
                    'residual_rgb_latents_&_residual_gray_latents_&_gt_residual_of_latent_residual': torch.cat(
                        [
                            residual_rgb_latents.view(-1, residual_rgb_latents.shape[-1]), 
                            residual_gray_latents.view(-1, residual_gray_latents.shape[-1]),
                            gt_residual_of_latent_residual.view(-1, gt_residual_of_latent_residual.shape[-1]), 
                        ], dim=-1),
                    'gt_residual_of_latent_residual_&_pd_residual_of_latent_residual': torch.cat(
                        [
                            gt_residual_of_latent_residual.view(-1, gt_residual_of_latent_residual.shape[-1]), 
                            pd_residual_of_latent_residual.view(-1, pd_residual_of_latent_residual.shape[-1]),
                        ], dim=-1),
                    'sigmoid_gt_residual_of_latent_residual_&_pd_residual_of_latent_residual': torch.cat(
                        [
                            torch.sigmoid(gt_residual_of_latent_residual).view(-1, gt_residual_of_latent_residual.shape[-1]), 
                            pd_residual_of_latent_residual.view(-1, pd_residual_of_latent_residual.shape[-1]),
                        ], dim=-1),
                }
            )
        elif pipeline['name'] in ['residual_of_latent_residual_with_kld']:
            start_rgb_latents = rgb_latents[:, :-1, :, :, :]
            end_rgb_latents = rgb_latents[:, 1:, :, :, :]
            start_gray_latents = gray_latents[:, :-1, :, :, :]
            end_gray_latents = gray_latents[:, 1:, :, :, :]
            residual_rgb_latents = end_rgb_latents - start_rgb_latents
            residual_gray_latents = end_gray_latents - start_gray_latents
            gt_residual_of_latent_residual = residual_rgb_latents - residual_gray_latents
            # 
            input_tensor = torch.cat([start_gray_latents, end_gray_latents, residual_gray_latents, event_latents], dim=2)
            frame_num = input_tensor.shape[1]
            input_tensor = rearrange(input_tensor, "b f c h w -> (b f) c h w")
            output_tensor = networks['vit'](input_tensor)
            output_tensor = rearrange(output_tensor, "(b f) c h w -> b f c h w", f=frame_num)
            output_tensor = output_tensor.contiguous()
            pd_residual_of_latent_residual_mean, pd_residual_of_latent_residual_logvar = torch.chunk(output_tensor, 2, dim=2)
            pd_residual_of_latent_residual_logvar = torch.clamp(pd_residual_of_latent_residual_logvar, -30.0, 20.0)
            pd_residual_of_latent_residual_std = torch.exp(0.5 * pd_residual_of_latent_residual_logvar)
            pd_residual_of_latent_residual_var = torch.exp(pd_residual_of_latent_residual_logvar)
            pd_residual_of_latent_residual = pd_residual_of_latent_residual_mean + pd_residual_of_latent_residual_std * torch.randn_like(pd_residual_of_latent_residual_mean)

            for criterion in pipeline['criterionchain']:
                if criterion['name'] == 'l1_loss':
                    criterion_value = torch.mean(torch.abs(pd_residual_of_latent_residual - gt_residual_of_latent_residual))
                elif criterion['name'] == 'l2_loss':
                    criterion_value = torch.nn.functional.mse_loss(pd_residual_of_latent_residual, gt_residual_of_latent_residual, reduction="mean")
                elif criterion['name'] == 'kld_loss':
                    criterion_value = torch.mean(0.5 * torch.sum(torch.pow(pd_residual_of_latent_residual_mean, 2) + pd_residual_of_latent_residual_var - 1.0 - pd_residual_of_latent_residual_logvar, dim=[2, 3, 4]))
                losses[pipeline['name'] + '/' + criterion['name']] = {
                    'value': criterion_value,
                    'lambda': criterion['lambda']
                }
            # 
            tensors.update(
                {
                    'residual_rgb_latents_&_residual_gray_latents_&_gt_residual_of_latent_residual': torch.cat(
                        [
                            residual_rgb_latents.view(-1, residual_rgb_latents.shape[-1]), 
                            residual_gray_latents.view(-1, residual_gray_latents.shape[-1]),
                            gt_residual_of_latent_residual.view(-1, gt_residual_of_latent_residual.shape[-1]), 
                        ], dim=-1),
                    'gt_residual_of_latent_residual_&_pd_residual_of_latent_residual': torch.cat(
                        [
                            gt_residual_of_latent_residual.view(-1, gt_residual_of_latent_residual.shape[-1]), 
                            pd_residual_of_latent_residual.view(-1, pd_residual_of_latent_residual.shape[-1]),
                        ], dim=-1),
                    'sigmoid_gt_residual_of_latent_residual_&_sigmoid_pd_residual_of_latent_residual': torch.cat(
                        [
                            torch.sigmoid(gt_residual_of_latent_residual).view(-1, gt_residual_of_latent_residual.shape[-1]), 
                            torch.sigmoid(pd_residual_of_latent_residual).view(-1, pd_residual_of_latent_residual.shape[-1]),
                        ], dim=-1),
                    'sigmoid_gt_residual_of_latent_residual_&_sigmoid_pd_residual_of_latent_residual_mean': torch.cat(
                        [
                            torch.sigmoid(gt_residual_of_latent_residual).view(-1, gt_residual_of_latent_residual.shape[-1]), 
                            torch.sigmoid(pd_residual_of_latent_residual_mean).view(-1, pd_residual_of_latent_residual.shape[-1]),
                        ], dim=-1),
                }
            )
        elif pipeline['name'] in ['residual_of_latent_residual_with_kld_with_stochastic']:
            start_rgb_latents = rgb_latents[:, :-1, :, :, :]
            start_rgb_posteriors_mean = rgb_posteriors.mean[:, :-1, :, :, :]
            start_rgb_posteriors_var = rgb_posteriors.var[:, :-1, :, :, :]
            # 
            end_rgb_latents = rgb_latents[:, 1:, :, :, :]
            end_rgb_posteriors_mean = rgb_posteriors.mean[:, 1:, :, :, :]
            end_rgb_posteriors_var = rgb_posteriors.var[:, 1:, :, :, :]
            # 
            residual_rgb_latents = end_rgb_latents - start_rgb_latents
            residual_rgb_posteriors_mean = end_rgb_posteriors_mean - start_rgb_posteriors_mean
            residual_rgb_posteriors_var = end_rgb_posteriors_var + start_rgb_posteriors_var

            start_gray_latents = gray_latents[:, :-1, :, :, :]
            start_gray_posteriors_mean = gray_posteriors.mean[:, :-1, :, :, :]
            start_gray_posteriors_var = gray_posteriors.var[:, :-1, :, :, :]
            # 
            end_gray_latents = gray_latents[:, 1:, :, :, :]
            end_gray_posteriors_mean = gray_posteriors.mean[:, 1:, :, :, :]
            end_gray_posteriors_var = gray_posteriors.var[:, 1:, :, :, :]
            # 
            residual_gray_latents = end_gray_latents - start_gray_latents
            residual_gray_posteriors_mean = end_gray_posteriors_mean - start_gray_posteriors_mean
            residual_gray_posteriors_var = end_gray_posteriors_var + start_gray_posteriors_var
            
            gt_residual_of_latent_residual = residual_rgb_latents - residual_gray_latents
            gt_residual_of_latent_residual_mean = residual_rgb_posteriors_mean - residual_gray_posteriors_mean
            gt_residual_of_latent_residual_var  = residual_rgb_posteriors_var + residual_gray_posteriors_var
            gt_residual_of_latent_residual_logvar = torch.log(gt_residual_of_latent_residual_var)
            
            input_tensor = torch.cat([start_gray_latents, end_gray_latents, residual_gray_latents, event_latents], dim=2)
            frame_num = input_tensor.shape[1]
            input_tensor = rearrange(input_tensor, "b f c h w -> (b f) c h w")
            output_tensor = networks['vit'](input_tensor)
            output_tensor = rearrange(output_tensor, "(b f) c h w -> b f c h w", f=frame_num)
            output_tensor = output_tensor.contiguous()
            pd_residual_of_latent_residual_mean, pd_residual_of_latent_residual_logvar = torch.chunk(output_tensor, 2, dim=2)
            pd_residual_of_latent_residual_mean = pd_residual_of_latent_residual_mean.contiguous()
            pd_residual_of_latent_residual_logvar = torch.clamp(pd_residual_of_latent_residual_logvar, -30.0, 20.0).contiguous()
            pd_residual_of_latent_residual_std = torch.exp(0.5 * pd_residual_of_latent_residual_logvar).contiguous()
            pd_residual_of_latent_residual_var = torch.exp(pd_residual_of_latent_residual_logvar).contiguous()
            pd_residual_of_latent_residual = (pd_residual_of_latent_residual_mean + pd_residual_of_latent_residual_std * torch.randn_like(pd_residual_of_latent_residual_mean)).contiguous()

            for criterion in pipeline['criterionchain']:
                if criterion['name'] == 'l1_loss':
                    criterion_value = torch.mean(torch.abs(pd_residual_of_latent_residual - gt_residual_of_latent_residual))
                elif criterion['name'] == 'l2_loss':
                    criterion_value = torch.nn.functional.mse_loss(pd_residual_of_latent_residual, gt_residual_of_latent_residual, reduction="mean")
                elif criterion['name'] == 'kld_loss':
                    criterion_value = torch.mean(0.5 * torch.sum(
                        torch.pow(pd_residual_of_latent_residual_mean - gt_residual_of_latent_residual_mean, 2) / gt_residual_of_latent_residual_var
                        + pd_residual_of_latent_residual_var / gt_residual_of_latent_residual_var
                        - 1.0
                        - pd_residual_of_latent_residual_logvar
                        + gt_residual_of_latent_residual_logvar,
                        dim=[2, 3, 4]
                    ))

                losses[pipeline['name'] + '/' + criterion['name']] = {
                    'value': criterion_value,
                    'lambda': criterion['lambda']
                }
            # 
            tensors.update(
                {
                    'gt_residual_of_latent_residual_&_pd_residual_of_latent_residual': torch.cat(
                        [
                            gt_residual_of_latent_residual.view(-1, gt_residual_of_latent_residual.shape[-1]), 
                            pd_residual_of_latent_residual.view(-1, pd_residual_of_latent_residual.shape[-1]), 
                        ], dim=-1),
                    'gt_residual_of_latent_residual_mean_&_pd_residual_of_latent_residual_mean': torch.cat(
                        [
                            gt_residual_of_latent_residual_mean.view(-1, gt_residual_of_latent_residual_mean.shape[-1]), 
                            pd_residual_of_latent_residual_mean.view(-1, pd_residual_of_latent_residual_mean.shape[-1]),
                        ], dim=-1),
                    'gt_residual_of_latent_residual_var_&_pd_residual_of_latent_residual_var': torch.cat(
                        [
                            gt_residual_of_latent_residual_var.view(-1, gt_residual_of_latent_residual_var.shape[-1]), 
                            pd_residual_of_latent_residual_var.view(-1, pd_residual_of_latent_residual_var.shape[-1]),
                        ], dim=-1),
                }
            )
        elif pipeline['name'] in ['event_to_redisual_rgb_value']:
            start_rgb_values = inputs['rgb_values'][:, :-1, :, :, :]
            end_rgb_values = inputs['rgb_values'][:, 1:, :, :, :]
            event_values = inputs['event_values']
            gt_residual_rgb_values = end_rgb_values - start_rgb_values
            # 
            frame_num = event_values.shape[1]
            event_values = rearrange(event_values, "b f c h w -> (b f) c h w")
            pd_residual_rgb_values = networks['resnet'](event_values)
            pd_residual_rgb_values = rearrange(pd_residual_rgb_values, "(b f) c h w -> b f c h w", f=frame_num)
            pd_residual_rgb_values = pd_residual_rgb_values.contiguous()

            for criterion in pipeline['criterionchain']:
                if criterion['name'] == 'l1_loss':
                    criterion_value = torch.mean(torch.abs(pd_residual_rgb_values - gt_residual_rgb_values))
                elif criterion['name'] == 'l2_loss':
                    criterion_value = torch.nn.functional.mse_loss(pd_residual_rgb_values, gt_residual_rgb_values, reduction="mean")
                losses[pipeline['name'] + '/' + criterion['name']] = {
                    'value': criterion_value,
                    'lambda': criterion['lambda']
                }
            tensors.update(
                {
                    'gt_residual_rgb_values_&_pd_residual_rgb_values': torch.cat(
                        [
                            gt_residual_rgb_values.view(-1, gt_residual_rgb_values.shape[-1]), 
                            pd_residual_rgb_values.view(-1, pd_residual_rgb_values.shape[-1]),
                        ], dim=-1),
                }
            )
        elif pipeline['name'] in ['obtain_latent']:
            with torch.no_grad():
                rgb_values, rgb_posteriors, rgb_latents = encode(values=inputs['rgb_values'], vae=networks['vae_rgb'], config=config)
                gray_values, gray_posteriors, gray_latents = encode(values=inputs['gray_values'], vae=networks['vae_rgb'], config=config)
                event_values, event_posteriors, event_latents = encode(values=inputs['event_values'], vae=networks['vae_event'], config=config)
            batch_idx = inputs['batch_idx']
            rgb_latent_residuals = rgb_latents[:, 1:, :, :, :] - rgb_latents[:, :-1, :, :, :]
            torch.save(rgb_latent_residuals, os.path.join(pipeline['latent_folder'], f'batch@{batch_idx:04d}_type@rgb_latent_residuals.pth'))
            torch.save(rgb_latents, os.path.join(pipeline['latent_folder'], f'batch@{batch_idx:04d}_type@rgb_latents.pth'))
        elif pipeline['name'] in ['quantized_latent_residual_prediction']:
            with torch.no_grad():
                rgb_values, rgb_posteriors, rgb_latents = encode(values=inputs['rgb_values'], vae=networks['vae_rgb'], config=config)
                gray_values, gray_posteriors, gray_latents = encode(values=inputs['gray_values'], vae=networks['vae_rgb'], config=config)
                event_values, event_posteriors, event_latents = encode(values=inputs['event_values'], vae=networks['vae_event'], config=config)
            # 
            start_rgb_latents = rgb_latents[:, :-1, :, :, :]
            end_rgb_latents = rgb_latents[:, 1:, :, :, :]
            start_gray_latents = gray_latents[:, :-1, :, :, :]
            end_gray_latents = gray_latents[:, 1:, :, :, :]
            gray_latent_residuals = end_gray_latents - start_gray_latents
            rgb_latent_residuals = end_rgb_latents - start_rgb_latents
            # 
            quantized_point_values = torch.load(pipeline['quantized_point_value'])
            differences = torch.abs(rgb_latent_residuals.unsqueeze(-1) - quantized_point_values)
            # 
            # 找到每个位置对应的最小差值的索引
            quantized_rgb_latent_residuals = torch.argmin(differences, dim=-1)
            # print(quantized_rgb_latent_residuals.max().item())
            # print(quantized_rgb_latent_residuals.min().item())
            # 
            # 使用这些索引从A中提取出对应的量化值
            # quantized_rgb_latent_residuals = quantized_point_values[closest_indices]
            # 
            batch_num, frame_num, channel_num, height_num, width_num = event_latents.shape
            input_tensor = torch.cat([start_gray_latents, end_gray_latents, gray_latent_residuals, event_latents], dim=2)
            input_tensor = rearrange(input_tensor, "b f c h w -> (b f) c h w")
            output_tensor = networks['resnet'](input_tensor)
            output_tensor = rearrange(output_tensor, "(b f) c h w -> b f c h w", f=frame_num)
            output_tensor = output_tensor.view(batch_num, frame_num, -1, channel_num, height_num, width_num).permute(0, 2, 1, 3, 4, 5)
            logited_rgb_latent_residuals = output_tensor.contiguous()
            # 
            for criterion in pipeline['criterionchain']:
                if criterion['name'] == 'l1_loss':
                    criterion_value = torch.mean(torch.abs(pd_residual_of_latent_residual - gt_residual_of_latent_residual))
                elif criterion['name'] == 'l2_loss':
                    criterion_value = torch.nn.functional.mse_loss(pd_residual_of_latent_residual, gt_residual_of_latent_residual, reduction="mean")
                elif criterion['name'] == 'cross_entropy_loss':
                    criterion_value = torch.nn.functional.cross_entropy(logited_rgb_latent_residuals, quantized_rgb_latent_residuals, reduction='mean')
                elif criterion['name'] == 'accuracy':
                    correct_num = (torch.argmax(logited_rgb_latent_residuals, dim=1) == quantized_rgb_latent_residuals).sum().item()
                    total_num = quantized_rgb_latent_residuals.numel()
                    accuracy = correct_num / total_num
                    criterion_value = torch.tensor(accuracy).to(device=quantized_rgb_latent_residuals.device)
                elif criterion['name'] == 'psnr':
                    mse = torch.mean((torch.argmax(logited_rgb_latent_residuals, dim=1).to(dtype=torch.float32) - quantized_rgb_latent_residuals.to(dtype=torch.float32)) ** 2)
                    criterion_value = 20 * torch.log10(criterion['peak_value'] / torch.sqrt(mse))
                losses[pipeline['name'] + '/' + criterion['name']] = {
                    'value': criterion_value,
                    'lambda': criterion['lambda']
                }
            # 
            tensors.update(
                {}
            )
        elif pipeline['name'] in ['spatial_quantized_latent_residual_prediction']:
            with torch.no_grad():
                rgb_values, rgb_posteriors, rgb_latents = encode(values=inputs['rgb_values'], vae=networks['vae_rgb'], config=config)
                gray_values, gray_posteriors, gray_latents = encode(values=inputs['gray_values'], vae=networks['vae_rgb'], config=config)
                event_values, event_posteriors, event_latents = encode(values=inputs['event_values'], vae=networks['vae_event'], config=config)
            # 
            batch_num, frame_num, channel_num, height_num, width_num = event_latents.shape
            # 
            start_rgb_latents = rgb_latents[:, :-1, :, :, :]
            end_rgb_latents = rgb_latents[:, 1:, :, :, :]
            start_gray_latents = gray_latents[:, :-1, :, :, :]
            end_gray_latents = gray_latents[:, 1:, :, :, :]
            gray_latent_residuals = end_gray_latents - start_gray_latents
            rgb_latent_residuals = end_rgb_latents - start_rgb_latents
            # 
            spatial_quantized_codebook = torch.load(pipeline['spatial_quantized_codebook'])

            rgb_latent_residuals_ = rgb_latent_residuals.permute(0, 1, 3, 4, 2).reshape(-1, channel_num)
            dists = torch.cdist(rgb_latent_residuals_.unsqueeze(0), spatial_quantized_codebook.unsqueeze(0), p=2).squeeze(0)
            spatial_quantized_rgb_latent_residuals = torch.argmin(dists, dim=-1)
            spatial_quantized_rgb_latent_residuals = spatial_quantized_rgb_latent_residuals.reshape(batch_num, frame_num, height_num, width_num)

            if inputs['stage'] == 'test':
                if inputs['batch_idx'] == 0:
                    codebook_statistics_dict = {f'{i:04d}': 0 for i in range(pipeline['codebook_num'])}
                else:
                    with open(pipeline['codebook_statistics'], 'r') as file:
                        codebook_statistics_dict = json.load(file)
                unique_elements, element_counts = torch.unique(spatial_quantized_rgb_latent_residuals, return_counts=True)
                for elem, count in zip(unique_elements.tolist(), element_counts.tolist()):
                    key = f'{elem:04d}'
                    if key in codebook_statistics_dict:
                        codebook_statistics_dict[key] += count
                    else:
                        codebook_statistics_dict[key] = count
                with open(pipeline['codebook_statistics'], 'w') as file:
                    json.dump(codebook_statistics_dict, file, ensure_ascii=False, indent=4)

            input_tensor = torch.cat([start_gray_latents, end_gray_latents, gray_latent_residuals, event_latents], dim=2)
            input_tensor = rearrange(input_tensor, "b f c h w -> (b f) c h w")
            output_tensor = networks['resnet'](input_tensor)
            output_tensor = rearrange(output_tensor, "(b f) c h w -> b f c h w", f=frame_num)
            logited_rgb_latent_residuals = output_tensor.permute(0, 2, 1, 3, 4).contiguous()
            # 
            for criterion in pipeline['criterionchain']:
                if criterion['name'] == 'l1_loss':
                    criterion_value = torch.mean(torch.abs(pd_residual_of_latent_residual - gt_residual_of_latent_residual))
                elif criterion['name'] == 'l2_loss':
                    criterion_value = torch.nn.functional.mse_loss(pd_residual_of_latent_residual, gt_residual_of_latent_residual, reduction="mean")
                elif criterion['name'] == 'cross_entropy_loss':
                    criterion_value = torch.nn.functional.cross_entropy(logited_rgb_latent_residuals, spatial_quantized_rgb_latent_residuals, reduction='mean')
                elif criterion['name'] == 'accuracy':
                    correct_num = (torch.argmax(logited_rgb_latent_residuals, dim=1) == spatial_quantized_rgb_latent_residuals).sum().item()
                    total_num = spatial_quantized_rgb_latent_residuals.numel()
                    accuracy = correct_num / total_num
                    criterion_value = torch.tensor(accuracy).to(device=spatial_quantized_rgb_latent_residuals.device)
                elif criterion['name'] == 'psnr':
                    mse = torch.mean((torch.argmax(logited_rgb_latent_residuals, dim=1).to(dtype=torch.float32) - spatial_quantized_rgb_latent_residuals.to(dtype=torch.float32)) ** 2)
                    criterion_value = 20 * torch.log10(criterion['peak_value'] / torch.sqrt(mse))
                elif criterion['name'] == 'quantized_loss':
                    spatial_dequantized_rgb_latent_residuals = spatial_quantized_codebook[spatial_quantized_rgb_latent_residuals.view(-1)].view(batch_num, frame_num, height_num, width_num, channel_num)
                    spatial_dequantized_rgb_latent_residuals = spatial_dequantized_rgb_latent_residuals.permute(0, 1, 4, 2, 3)
                    criterion_value = torch.mean(torch.abs(spatial_dequantized_rgb_latent_residuals - rgb_latent_residuals))
                elif criterion['name'] == 'predicted_quantized_loss':
                    predicted_dequantized_rgb_latent_residuals = spatial_quantized_codebook[torch.argmax(logited_rgb_latent_residuals, dim=1).view(-1)].view(batch_num, frame_num, height_num, width_num, channel_num)
                    predicted_dequantized_rgb_latent_residuals = predicted_dequantized_rgb_latent_residuals.permute(0, 1, 4, 2, 3)
                    criterion_value = torch.mean(torch.abs(predicted_dequantized_rgb_latent_residuals - rgb_latent_residuals))
                elif criterion['name'] == 'baseline_mae_between_gray_and_rgb_residuals':
                    criterion_value = torch.mean(torch.abs(gray_latent_residuals - rgb_latent_residuals))
                losses[pipeline['name'] + '/' + criterion['name']] = {
                    'value': criterion_value,
                    'lambda': criterion['lambda']
                }
            # 
            tensors.update(
                {
                    'logited_rgb_latent_residuals_&_spatial_quantized_rgb_latent_residuals': torch.cat(
                        [
                            torch.argmax(logited_rgb_latent_residuals, dim=1).view(-1, torch.argmax(logited_rgb_latent_residuals, dim=1).shape[-1]), 
                            spatial_quantized_rgb_latent_residuals.view(-1, spatial_quantized_rgb_latent_residuals.shape[-1]),
                        ], dim=-1),
                }
            )
        elif pipeline['name'] in ['multi_scale_residual_of_latent_residual']:
            with torch.no_grad():
                rgb_values, rgb_posteriors, rgb_latents = encode(values=inputs['rgb_values'], vae=networks['vae_rgb'], config=config)
                rgb_features = get_feature_from_encoder_forward(encoder=networks['vae_rgb'].encoder, sample=inputs['rgb_values'])
                # 
                gray_values, gray_posteriors, gray_latents = encode(values=inputs['gray_values'], vae=networks['vae_rgb'], config=config)
                gray_features = get_feature_from_encoder_forward(encoder=networks['vae_rgb'].encoder, sample=inputs['gray_values'])
                # 
                event_values, event_posteriors, event_latents = encode(values=inputs['event_values'], vae=networks['vae_event'], config=config)
                event_features = get_feature_from_encoder_forward(encoder=networks['vae_event'].encoder, sample=inputs['event_values'])
            
            # batch_num, frame_num, channel_num, height_num, width_num = rgb_features[0].shape
            gt_rgb_feature_residuals = []
            gt_gray_feature_residuals = []
            input_features = []
            for idx in pipeline['feature_layer_idx']:
                gt_rgb_feature_residuals.append(rgb_features[idx][:, 1:, :, :, :] - rgb_features[idx][:, :-1, :, :, :])
                gt_gray_feature_residuals.append(gray_features[idx][:, 1:, :, :, :] - gray_features[idx][:, :-1, :, :, :])
                input_features.append(torch.cat([gray_features[idx][:, 1:, :, :, :], gray_features[idx][:, :-1, :, :, :], event_features[idx]], dim=2))
            gt_rgb_latent_residual = rgb_latents[:, 1:, :, :, :] - rgb_latents[:, :-1, :, :, :]
            gt_gray_latent_residual = gray_latents[:, 1:, :, :, :] - gray_latents[:, :-1, :, :, :]
            pd_final, pd_tails = networks['multi_scale_resnet'](features=input_features)

            for criterion in pipeline['criterionchain']:
                if criterion['name'] == 'l1_loss':
                    criterion_value = torch.mean(torch.abs(pd_final + gt_gray_latent_residual - gt_rgb_latent_residual))
                elif criterion['name'] == 'l1_loss_feature_sclae@00':
                    criterion_value = torch.mean(torch.abs(pd_tails[0] + gt_gray_feature_residuals[0] - gt_rgb_feature_residuals[0]))
                elif criterion['name'] == 'l1_loss_feature_sclae@01':
                    criterion_value = torch.mean(torch.abs(pd_tails[1] + gt_gray_feature_residuals[1]- gt_rgb_feature_residuals[1]))
                elif criterion['name'] == 'l1_loss_feature_sclae@02':
                    criterion_value = torch.mean(torch.abs(pd_tails[2] + gt_gray_feature_residuals[2] - gt_rgb_feature_residuals[2]))
                elif criterion['name'] == 'l1_loss_feature_sclae@03':
                    criterion_value = torch.mean(torch.abs(pd_tails[3] + gt_gray_feature_residuals[3] - gt_rgb_feature_residuals[3]))
                elif criterion['name'] == 'l1_loss_baseline':
                    criterion_value = torch.mean(torch.abs(gt_gray_latent_residual - gt_rgb_latent_residual))
                elif criterion['name'] == 'l1_loss_baseline_feature_sclae@00':
                    criterion_value = torch.mean(torch.abs(gt_gray_feature_residuals[0] - gt_rgb_feature_residuals[0]))
                elif criterion['name'] == 'l1_loss_baseline_feature_sclae@01':
                    criterion_value = torch.mean(torch.abs(gt_gray_feature_residuals[1] - gt_rgb_feature_residuals[1]))
                elif criterion['name'] == 'l1_loss_baseline_feature_sclae@02':
                    criterion_value = torch.mean(torch.abs(gt_gray_feature_residuals[2] - gt_rgb_feature_residuals[2]))
                elif criterion['name'] == 'l1_loss_baseline_feature_sclae@03':
                    criterion_value = torch.mean(torch.abs(gt_gray_feature_residuals[3] - gt_rgb_feature_residuals[3]))
                losses[pipeline['name'] + '/' + criterion['name']] = {
                    'value': criterion_value,
                    'lambda': criterion['lambda']
                }
            # 
            pd_rgb_latent_residual = pd_final + gt_gray_latent_residual
            pd_rgb_feature_residuals = [
                pd_tails[0] + gt_gray_feature_residuals[0],
                pd_tails[1] + gt_gray_feature_residuals[1],
                pd_tails[2] + gt_gray_feature_residuals[2],
                pd_tails[3] + gt_gray_feature_residuals[3],
            ]
            # 
            tensors.update(
                {
                    'pd_rgb_latent_residual_&_gt_rgb_latent_residual': torch.cat(
                        [
                            pd_rgb_latent_residual.view(-1, pd_rgb_latent_residual.shape[-1]), 
                            gt_rgb_latent_residual.view(-1, gt_rgb_latent_residual.shape[-1]),
                        ], dim=-1),
                    'pd_rgb_feature_residuals[0]_&_gt_rgb_feature_residuals[0]': torch.cat(
                        [
                            pd_rgb_feature_residuals[0][:, :, :4, :, :].contiguous().view(-1, pd_rgb_feature_residuals[0][:, :, :4, :, :].shape[-1]), 
                            gt_rgb_feature_residuals[0][:, :, :4, :, :].contiguous().view(-1, gt_rgb_feature_residuals[0][:, :, :4, :, :].shape[-1]),
                        ], dim=-1),
                    'pd_rgb_feature_residuals[1]_&_gt_rgb_feature_residuals[1]': torch.cat(
                        [
                            pd_rgb_feature_residuals[1][:, :, :4, :, :].contiguous().view(-1, pd_rgb_feature_residuals[1][:, :, :4, :, :].shape[-1]), 
                            gt_rgb_feature_residuals[1][:, :, :4, :, :].contiguous().view(-1, gt_rgb_feature_residuals[1][:, :, :4, :, :].shape[-1]),
                        ], dim=-1),
                    'pd_rgb_feature_residuals[2]_&_gt_rgb_feature_residuals[2]': torch.cat(
                        [
                            pd_rgb_feature_residuals[2][:, :, :4, :, :].contiguous().view(-1, pd_rgb_feature_residuals[2][:, :, :4, :, :].shape[-1]), 
                            gt_rgb_feature_residuals[2][:, :, :4, :, :].contiguous().view(-1, gt_rgb_feature_residuals[2][:, :, :4, :, :].shape[-1]),
                        ], dim=-1),
                    'pd_rgb_feature_residuals[3]_&_gt_rgb_feature_residuals[3]': torch.cat(
                        [
                            pd_rgb_feature_residuals[3][:, :, :4, :, :].contiguous().view(-1, pd_rgb_feature_residuals[3][:, :, :4, :, :].shape[-1]), 
                            gt_rgb_feature_residuals[3][:, :, :4, :, :].contiguous().view(-1, gt_rgb_feature_residuals[3][:, :, :4, :, :].shape[-1]),
                        ], dim=-1),
                }
            )
        elif pipeline['name'] in ['vae_event_training']:
            event_values, event_posteriors, event_latents, event_decodes = encode_and_decode(values=inputs['event_values'], vae=networks['vae_event'], config=config, clamp_range=[])
            event_latents = event_latents.contiguous()
            for criterion in pipeline['criterionchain']:
                if criterion['name'] == 'l1_loss':
                    criterion_value = torch.mean(torch.abs(event_decodes - event_values))
                elif criterion['name'] == 'kld_loss':
                    criterion_value = torch.mean(
                        0.5 * torch.sum(torch.pow(event_posteriors.mean, 2) + event_posteriors.var - 1.0 - event_posteriors.logvar, dim=[2, 3, 4])
                    )
                elif criterion['name'] == 'latent_std':
                    criterion_value = event_latents.std()
                losses[pipeline['name'] + '/' + criterion['name']] = {
                    'value': criterion_value,
                    'lambda': criterion['lambda']
                }
            tensors.update(
                {
                    'event_decodes_&_event_values': torch.cat(
                        [
                            event_decodes.view(-1, event_decodes.shape[-1]), 
                            event_values.view(-1, event_values.shape[-1]),
                        ], dim=-1),
                    'event_latents': torch.cat(
                        [
                            event_latents.view(-1, event_latents.shape[-1]),
                        ], dim=-1),
                }
            )
        elif pipeline['name'] in ['event_latent_prediction']:
            with torch.no_grad():
                rgb_values, rgb_posteriors, rgb_latents = encode(values=inputs['rgb_values'], vae=networks['vae_rgb'], config=config)
                event_values, event_posteriors, event_latents = encode(values=inputs['event_values'], vae=networks['vae_event'], config=config)
            batch_num, frame_num, channel_num, height_num, width_num = event_latents.shape

            end_rgb_latents = rgb_latents[:, 1:, :, :, :]
            start_rgb_latents = rgb_latents[:, :-1, :, :, :]
            rgb_latent_residuals = end_rgb_latents - start_rgb_latents
            gt_event_latents = event_latents.contiguous()
            
            input_tensor = torch.cat([end_rgb_latents, start_rgb_latents, rgb_latent_residuals], dim=2)
            input_tensor = rearrange(input_tensor, "b f c h w -> (b f) c h w")
            output_tensor = networks['resnet'](input_tensor)
            output_tensor = rearrange(output_tensor, "(b f) c h w -> b f c h w", f=frame_num).contiguous()
            pd_event_latents = output_tensor

            for criterion in pipeline['criterionchain']:
                if criterion['name'] == 'l1_loss':
                    criterion_value = torch.mean(torch.abs(pd_event_latents - gt_event_latents))
                elif criterion['name'] == 'l2_loss':
                    criterion_value = torch.nn.functional.mse_loss(pd_event_latents, gt_event_latents, reduction="mean")
                elif criterion['name'] == 'exp_loss':
                    criterion_value = torch.mean(torch.exp(torch.abs(pd_event_latents - gt_event_latents)) - 1)
                elif criterion['name'] == 'event_latent_std':
                    criterion_value = gt_event_latents.std()
                losses[pipeline['name'] + '/' + criterion['name']] = {
                    'value': criterion_value,
                    'lambda': criterion['lambda']
                }
            tensors.update(
                {
                    'pd_event_latents_&_gt_event_latents': torch.cat(
                        [
                            pd_event_latents.view(-1, pd_event_latents.shape[-1]), 
                            gt_event_latents.view(-1, gt_event_latents.shape[-1]),
                        ], dim=-1),
                }
            )


        elif pipeline['name'] in ['event_to_redisual_rgb_latent_with_resnet']:
            start_rgb_latents = rgb_latents[:, :-1, :, :, :]
            end_rgb_latents = rgb_latents[:, 1:, :, :, :]
            input_tensor = inputs['event_values']
            frame_num = input_tensor.shape[1]
            input_tensor = rearrange(input_tensor, "b f c h w -> (b f) c h w")
            output_tensor = networks['resnet_with_downscale'](input_tensor)
            output_tensor = rearrange(output_tensor, "(b f) c h w -> b f c h w", f=frame_num)
            pd_residual_rgb_latents = output_tensor
            gt_residual_rgb_latents = end_rgb_latents - start_rgb_latents
            # print(rgb_latents.std())
            
            for criterion in pipeline['criterionchain']:
                if criterion['name'] == 'l1_loss':
                    criterion_value = torch.mean(torch.abs(pd_residual_rgb_latents - gt_residual_rgb_latents))
                elif criterion['name'] == 'l2_loss':
                    criterion_value = torch.nn.functional.mse_loss(pd_residual_rgb_latents, gt_residual_rgb_latents, reduction="mean")
                losses[pipeline['name'] + '/' + criterion['name']] = {
                    'value': criterion_value,
                    'lambda': criterion['lambda']
                }
            start_rgb_latents = start_rgb_latents.contiguous()
            end_rgb_latents = end_rgb_latents.contiguous()
            pd_residual_rgb_latents = pd_residual_rgb_latents.contiguous()
            gt_residual_rgb_latents = gt_residual_rgb_latents.contiguous()
            tensors.update(
                {
                    'start_rgb_latents': start_rgb_latents.view(-1, start_rgb_latents.shape[-1]),
                    'end_rgb_latents': end_rgb_latents.view(-1, end_rgb_latents.shape[-1]),
                    'pd_residual_rgb_latents': pd_residual_rgb_latents.view(-1, pd_residual_rgb_latents.shape[-1]),
                    'gt_residual_rgb_latents': gt_residual_rgb_latents.view(-1, gt_residual_rgb_latents.shape[-1]),
                }
            )

    return tensors, losses