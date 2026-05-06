import torch
import os
import PIL.Image
import copy
import numpy as np
import cv2
import shutil
import glob
import logging
import argparse
import copy
from einops import rearrange

from diffusers import UNetSpatioTemporalConditionModel, StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif
from script.dataset import EventAndVideoDataset
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import retrieve_timesteps, _append_dims, StableVideoDiffusionPipelineOutput
from script.simple_vit import SimpleViT
from script.util import load_config, dump_config, normalize_and_save_tensor, feature2image, tensor2image, encode_and_decode, encode, decode, process_event, process_pipeline, reconstruct_R, save_image_from_chw_tensor
from script.resnet import ResNet, ResNetWithDownscale

def parse_args():
    parser = argparse.ArgumentParser(description="Script to Infer Stable Video Diffusion.")
    parser.add_argument("--refresh", default=True, action="store_true")
    parser.add_argument("--decode_chunk_size", default=4, type=int)
    parser.add_argument("--num_videos_per_prompt", default=1, type=int)
    parser.add_argument("--motion_bucket_id", default=127, type=int)
    parser.add_argument("--fps", default=7, type=int)
    parser.add_argument("--num_inference_steps", default=30, type=int)
    parser.add_argument("--noise_aug_strength", default=0.02, type=float)
    parser.add_argument("--config", default='experiments.svd/2025120201_LongSequenceGeneration/old.yaml', type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    config.output_folder = os.path.join(os.path.dirname(args.config), os.path.basename(args.config).split('.')[0])
    config.parameter_folder = os.path.join(os.path.dirname(args.config), os.path.basename(args.config).split('.')[0], 'parameter')
    config.log_folder = os.path.join(os.path.dirname(args.config), os.path.basename(args.config).split('.')[0], 'log')
    config.visualization_folder = os.path.join(os.path.dirname(args.config), os.path.basename(args.config).split('.')[0], 'visualization')

    return args, config

args, config = parse_args()

generator = torch.manual_seed(-1)
time_per_frame = None
sigmas = None
output_type = 'pil'
return_dict = True

if os.path.exists(config.log_folder) == False:
    os.makedirs(config.log_folder)

if os.path.exists(os.path.join(config.log_folder, 'inference.log')):
    with open(os.path.join(config.log_folder, 'inference.log'), 'w'):
        pass
logging.basicConfig(filename=os.path.join(config.log_folder, 'inference.log'), level=logging.INFO)
logger = logging.getLogger()

test_dataset = EventAndVideoDataset(
    folder_path=config.data.test_folder_path,
    mode='test',
    event_num=config.data.event_num,
    img_format=config.data.img_format,
    event_suffix=config.data.event_suffix,
    random_crop=[],
    condition_mode=[condition['name'] for condition in config.data.condition],
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config.data.test_batch_size,
    shuffle=False,
)

unet = UNetSpatioTemporalConditionModel.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    subfolder="unet",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)
unet.load_state_dict(torch.load(config.inference.unet_param_path), strict=False)
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    unet=unet,
    low_cpu_mem_usage=False,
    torch_dtype=torch.float16, variant="fp16", local_files_only=True,
)
pipe.to("cuda:0")
vae = pipe.vae

networks = {}
for network in config.network:
    if network['name'] in ['vae_rgb', 'vae_event']:
        net = copy.deepcopy(vae)
        net.config.scaling_factor = network['scaling_factor']
    elif network['name'] in ['vae_rgb_fp16']:
        net = copy.deepcopy(vae).to(dtype=torch.float16)
        net.config.scaling_factor = network['scaling_factor']
    elif network['name'] == 'resnet_with_downscale':
        net = ResNetWithDownscale(config=network)
    elif network['name'] == 'resnet':
        net = ResNet(config=network)
    if len(network['pretrained_path']) > 0:
        net.load_state_dict(torch.load(network['pretrained_path']), strict=True)
    net.requires_grad_(network['trainable'])
    net.to("cuda:0")
    networks[network['name']] = net

idx = 0
with torch.no_grad():
    for condition in config.data.condition:
        for batch_idx, batch in enumerate(test_dataloader):
            device = pipe._execution_device
            # 
            target_rgb_paths = [item[0] for item in batch['target_rgb_paths']]
            target_event_paths = [item[0] for item in batch['target_event_paths']]
            # 
            if condition['name'] == 'N-E-N':
                target_rgb_paths = target_rgb_paths[1:]
            # 
            output_rgb_paths = [item.replace(config.data.test_folder_path, os.path.join(config.visualization_folder, condition['name'])).replace('.jpg', '.png') for item in target_rgb_paths]
            output_dir_path = os.path.dirname(output_rgb_paths[0])
            if os.path.exists(output_dir_path):
                if args.refresh == True:
                    shutil.rmtree(output_dir_path)
                    os.makedirs(output_dir_path)
                else:
                    if len(glob.glob(os.path.join(output_dir_path, "*"))) == len(output_rgb_paths):
                        continue
                    else:
                        shutil.rmtree(output_dir_path)
                        os.makedirs(output_dir_path)
            else:
                os.makedirs(output_dir_path)
            # 
            rgb_values = batch["rgb_raw"].to(device)
            event_values = batch["event"].to(device)

            B_, F_e, C_, H_, W_ = event_values.shape
            if hasattr(config.data, 'test_at_dynamic_resolution') == True and config.data.test_at_dynamic_resolution == True:
                inference_height = (H_ + 63) // 64 * 64
                inference_width = (W_ + 63) // 64 * 64
            else:
                inference_height = config.data.height
                inference_width = config.data.width
            rgb_values = rgb_values.view(B_ * (F_e + 1), C_, H_, W_)
            event_values = event_values.view(B_ * F_e, C_, H_, W_)
            rgb_values_upsampled = torch.nn.functional.interpolate(rgb_values, size=(inference_height, inference_width), mode='bicubic', align_corners=False)
            event_values_upsampled = torch.nn.functional.interpolate(event_values, size=(inference_height, inference_width), mode='bicubic', align_corners=False)
            rgb_values = rgb_values_upsampled.view(B_, F_e + 1, C_, inference_height, inference_width)
            event_values = event_values_upsampled.view(B_, F_e, C_, inference_height, inference_width)
            # 
            if condition['name'] == 'R-E-R':
                encoder_hidden_pixel_values = rgb_values[:, 0, :, :, :]
                reference_values = torch.cat([rgb_values[:, 0:1, :, :, :], rgb_values[:, -1:, :, :, :]], dim=1)
                pixel_values = torch.cat([rgb_values, torch.zeros_like(rgb_values[:, 0:1, :, :, :])], dim=1)
                num_frames = config.data.event_num + 2
            elif condition['name'] == 'R-E-N':
                encoder_hidden_pixel_values = rgb_values[:, 0, :, :, :]
                reference_values = torch.cat([rgb_values[:, 0:1, :, :, :], torch.zeros_like(rgb_values[:, -1:, :, :, :])], dim=1)
                pixel_values = torch.cat([rgb_values, torch.zeros_like(rgb_values[:, 0:1, :, :, :])], dim=1)
                num_frames = config.data.event_num + 2
            elif condition['name'] == 'N-E-R':
                encoder_hidden_pixel_values = rgb_values[:, -1, :, :, :]
                reference_values = torch.cat([torch.zeros_like(rgb_values[:, 0:1, :, :, :]), rgb_values[:, -1:, :, :, :]], dim=1)
                pixel_values = torch.cat([rgb_values, torch.zeros_like(rgb_values[:, 0:1, :, :, :])], dim=1)
                num_frames = config.data.event_num + 2
            elif condition['name'] == 'N-E-N':
                encoder_hidden_pixel_values = event_values[:, 0, :, :, :]
                reference_values = None
                pixel_values = rgb_values[:, 1:, :, :, :]
                num_frames = config.data.event_num
            else:
                raise ValueError

            if hasattr(condition, 'min_guidance_scale') == False or hasattr(condition, 'max_guidance_scale') == False:
                condition['min_guidance_scale'] = 1.0
                condition['max_guidance_scale'] = 3.0

            pipe._guidance_scale = condition['max_guidance_scale']

            image_embeddings = pipe._encode_image_modified(encoder_hidden_pixel_values, device, args.num_videos_per_prompt, pipe.do_classifier_free_guidance)

            fps = args.fps - 1
            
            if hasattr(config.inference, 'adaptive_event_noise_coefficient'):
                sigma_E = event_values.std()
                alpha = config.inference.adaptive_event_noise_coefficient
                noise = torch.randn_like(event_values) * (alpha * sigma_E)
                event_values = event_values + noise
            else:
                event_values = event_values + args.noise_aug_strength * torch.randn_like(event_values)

            if condition['name'] == 'N-E-N':
                pass
            else:
                reference_values = reference_values + args.noise_aug_strength * torch.randn_like(reference_values)

            if config.inference.pipeline['name'] in ['image_space_customized_guidence_from_support']:
                if config.inference.pipeline['support_type'] == 'event':
                    frame_num = event_values.shape[1]
                    residual_rgb_values = networks['resnet'](rearrange(event_values, "b f c h w -> (b f) c h w"))
                    residual_rgb_values = rearrange(residual_rgb_values, "(b f) c h w -> b f c h w", f=frame_num)
                    residual_rgb_values[:, 0, :, :, :] = 0
                elif config.inference.pipeline['support_type'] == 'ground_truth':
                    residual_rgb_values = torch.zeros_like(pixel_values)
                    start_rgb_values = pixel_values[:, :-1, :, :, :]
                    end_rgb_values = pixel_values[:, 1:, :, :, :]
                    residual_rgb_values[:, 1:, :, :, :] = end_rgb_values - start_rgb_values
            

            needs_upcasting_rgb = networks['vae_rgb'].dtype == torch.float16 and networks['vae_rgb'].config.force_upcast
            if needs_upcasting_rgb:
                networks['vae_rgb'].to(dtype=torch.float32)
            needs_upcasting_event = networks['vae_event'].dtype == torch.float16 and networks['vae_event'].config.force_upcast
            if needs_upcasting_event:
                networks['vae_event'].to(dtype=torch.float32)

            if config.inference.pipeline['name'] in ['image_space_customized_guidence_from_support']:
                if hasattr(config.inference.pipeline, 'reference_mode'):
                    reference_start_rgb_latent = pipe._encode_vae_image_modified(
                        image=pixel_values[:, 0:1, :, :, :],
                        vae=networks['vae_rgb'],
                        device=device,
                        num_videos_per_prompt=args.num_videos_per_prompt,
                        do_classifier_free_guidance=False,
                        scaling=True,
                    )
                    reference_end_rgb_latent = pipe._encode_vae_image_modified(
                        image=pixel_values[:, -1:, :, :, :],
                        vae=networks['vae_rgb'],
                        device=device,
                        num_videos_per_prompt=args.num_videos_per_prompt,
                        do_classifier_free_guidance=False,
                        scaling=True,
                    )

            pixel_latents = pipe._encode_vae_image_modified(
                image=pixel_values,
                vae=networks['vae_rgb'],
                device=device,
                num_videos_per_prompt=args.num_videos_per_prompt,
                do_classifier_free_guidance=False,
                scaling=False,
            )
            noise_for_each_sample = torch.randn_like(pixel_latents)
            event_latents = pipe._encode_vae_image_modified(
                image=event_values,
                vae=networks['vae_event'],
                device=device,
                num_videos_per_prompt=args.num_videos_per_prompt,
                do_classifier_free_guidance=pipe.do_classifier_free_guidance,
                scaling=False,
            )
            
            if condition['name'] == 'N-E-N':
                pass
            else:
                reference_latents = pipe._encode_vae_image_modified(
                    image=reference_values,
                    vae=networks['vae_rgb'],
                    device=device,
                    num_videos_per_prompt=args.num_videos_per_prompt,
                    do_classifier_free_guidance=pipe.do_classifier_free_guidance,
                    scaling=False,
                )
            
            if condition['name'] == 'N-E-N':
                image_latents = event_latents
            else:
                image_latents = torch.cat([reference_latents, event_latents], dim=1)
            
            image_latents = image_latents.to(image_embeddings.dtype)

            added_time_ids = pipe._get_add_time_ids(
                fps,
                args.motion_bucket_id,
                args.noise_aug_strength,
                image_embeddings.dtype,
                config.data.test_batch_size,
                args.num_videos_per_prompt,
                pipe.do_classifier_free_guidance,
            )
            added_time_ids = added_time_ids.to(device)

            sigmas = None
            num_inference_steps_ = config.inference.step_num if hasattr(config.inference, 'step_num') else args.num_inference_steps
            timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps_, device, None, sigmas)

            num_channels_latents = pipe.unet.config.in_channels
            latents = None
            latents = pipe.prepare_latents(
                config.data.test_batch_size * args.num_videos_per_prompt,
                num_frames,
                num_channels_latents,
                inference_height,
                inference_width,
                image_embeddings.dtype,
                device,
                generator,
                latents,
            )

            guidance_scale = torch.linspace(condition['min_guidance_scale'], condition['max_guidance_scale'], num_frames).unsqueeze(0)
            guidance_scale = guidance_scale.to(device, latents.dtype)
            guidance_scale = guidance_scale.repeat(config.data.test_batch_size * args.num_videos_per_prompt, 1)
            guidance_scale = _append_dims(guidance_scale, latents.ndim)

            pipe._guidance_scale = guidance_scale

            num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
            pipe._num_timesteps = len(timesteps)
            with pipe.progress_bar(total=num_inference_steps) as progress_bar:
                for t_idx, t in enumerate(timesteps):
                    
                    latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
                    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                    
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                    noise_pred = pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]

                    
                    if pipe.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    latents = latents.to(torch.float32)
                    noise_pred = noise_pred.to(torch.float32)
                    sigmas = pipe.scheduler.sigmas[pipe.scheduler.step_index]
                    c_out = - sigmas / ((sigmas**2 + 1)**0.5)
                    c_skip = 1 / (sigmas**2 + 1)
                    weighing = (1 + sigmas ** 2) * (sigmas**-2.0)
                    pred_target_latents = noise_pred * c_out + c_skip * latents
                     
                    B, F, C, H, W = pred_target_latents.shape
                    event_latents_uncond, event_latents_cond = event_latents.chunk(2)
                    scaled_rgb_latents = networks['vae_rgb'].config.scaling_factor * pixel_latents
                    scaled_event_latents = networks['vae_event'].config.scaling_factor * event_latents_cond
                    rgb_latents = pred_target_latents
                    
                    if config.inference.pipeline['name'] in ['image_space_customized_guidence_from_support']:
                        print(t_idx)
                        if hasattr(config.inference.pipeline, 'reference_mode'):
                            # print(sigmas)
                            if hasattr(config.inference.pipeline, 'reference_weight_strategy'):
                                if config.inference.pipeline['reference_weight_strategy'] == 'linear':
                                    lambda_shift = config.inference.pipeline['reference_weight_start'] - (config.inference.pipeline['reference_weight_start'] - config.inference.pipeline['reference_weight_end']) * (t_idx / (len(timesteps) - 1))
                                    print('-----↓')
                                    print(lambda_shift)
                                    print('-----↑')
                                elif config.inference.pipeline['reference_weight_strategy'] == 'nonlinear':
                                    if config.inference.pipeline['reference_weight_order'] == 'descending':
                                        lambda_shift = 1 - torch.exp(-sigmas)
                                    elif config.inference.pipeline['reference_weight_order'] == 'aescending':
                                        lambda_shift = torch.exp(-sigmas)
                            else:
                                lambda_shift = 1 - torch.exp(-sigmas)
                            # 
                            diff_start_rgb_latent = reference_start_rgb_latent - rgb_latents[:, 0:1, :, :, :]
                            diff_start_rgb_latent = diff_start_rgb_latent.repeat(1, rgb_latents.shape[1], 1, 1, 1)
                            shift_start_rgb_latents = diff_start_rgb_latent + rgb_latents
                            # 
                            diff_end_rgb_latent = reference_end_rgb_latent - rgb_latents[:, -1:, :, :, :]
                            diff_end_rgb_latent = diff_end_rgb_latent.repeat(1, rgb_latents.shape[1], 1, 1, 1)
                            shift_end_rgb_latents = diff_end_rgb_latent + rgb_latents
                            # 
                            if config.inference.pipeline['reference_mode'] == 'vfi':
                                shift_rgb_latents = 0.5 * lambda_shift * shift_start_rgb_latents + 0.5 * lambda_shift * shift_end_rgb_latents + (1 - lambda_shift) * rgb_latents
                            elif config.inference.pipeline['reference_mode'] == 'vfp':
                                shift_rgb_latents = lambda_shift * shift_start_rgb_latents + (1 - lambda_shift) * rgb_latents
                                constraint_rgb_latents = rgb_latents
                            rgb_latents = shift_rgb_latents

                        if t_idx in config.inference.pipeline['guidence_t_idx']:
                            if hasattr(config.inference.pipeline, 'learning_rate_strategy') == False:
                                learning_rate = config.inference.pipeline['learning_rate']
                            elif config.inference.pipeline['learning_rate_strategy'] == 'linear':
                                learning_rate = config.inference.pipeline['learning_rate_start'] - (config.inference.pipeline['learning_rate_start'] - config.inference.pipeline['learning_rate_end']) * (config.inference.pipeline['guidence_t_idx'].index(t_idx) / (len(config.inference.pipeline['guidence_t_idx']) - 1))
                            elif config.inference.pipeline['learning_rate_strategy'] == 'exponential':
                                learning_rate = config.inference.pipeline['learning_rate_start'] * (config.inference.pipeline['learning_rate_end'] / config.inference.pipeline['learning_rate_start']) ** (config.inference.pipeline['guidence_t_idx'].index(t_idx) / (len(config.inference.pipeline['guidence_t_idx']) - 1))
                            print(f'{learning_rate}')
                            fp16_rgb_latents = rgb_latents.to(dtype=torch.float16)
                            fp16_rgb_latents = fp16_rgb_latents.flatten(0, 1)
                            fp16_rgb_latents = 1 / networks['vae_rgb_fp16'].config.scaling_factor * fp16_rgb_latents
                            window_size = config.inference.pipeline['windows_size']
                            backward_num = 1 if hasattr(config.inference.pipeline, 'backward_num') == False else config.inference.pipeline['backward_num']
                            with torch.set_grad_enabled(True):
                                fp16_rgb_latents.requires_grad_(True)
                                for backward_idx in range(backward_num):
                                    fp16_rgb_latent_grads = torch.zeros_like(fp16_rgb_latents)
                                    for i in range(0, fp16_rgb_latents.shape[0] - 1, window_size - 1):
                                        print(f'[{i:d}->{i+window_size-1:d}]')
                                        decode_rgb_values = networks['vae_rgb_fp16'].decode(fp16_rgb_latents[i : i + window_size], num_frames=fp16_rgb_latents[i : i + window_size].shape[0]).sample
                                        decode_residual_rgb_values = decode_rgb_values[1:, :, :, :] - decode_rgb_values[:-1, :, :, :]
                                        residual_loss = torch.sum(torch.abs(decode_residual_rgb_values - residual_rgb_values[0, i+1:i+window_size, :, :, :]))
                                        mae_value = torch.mean(torch.abs(decode_residual_rgb_values - residual_rgb_values[0, i+1:i+window_size, :, :, :])).item()
                                        mse_value = torch.nn.functional.mse_loss(decode_residual_rgb_values, residual_rgb_values[0, i+1:i+window_size, :, :, :], reduction="mean").item()
                                        logger.info(f'batch_idx@{batch_idx}_range@[{i:d}->{i+window_size-1:d}]_t_idx@{t_idx}_mae@{mae_value:04.4f}_mse@{mse_value:04.4f}')
                                        grad = torch.autograd.grad(residual_loss, fp16_rgb_latents, grad_outputs=torch.ones_like(residual_loss), create_graph=False, retain_graph=False, only_inputs=True, allow_unused=True)[0]
                                        fp16_rgb_latent_grads += grad
                                    fp16_rgb_latents = fp16_rgb_latents - learning_rate * fp16_rgb_latent_grads
                            
                            constraint_rgb_latents = networks['vae_rgb_fp16'].config.scaling_factor * fp16_rgb_latents.to(dtype=torch.float32) 
                            constraint_rgb_latents = rearrange(constraint_rgb_latents, "(b f) c h w -> b f c h w", f=F)
                        else:
                            constraint_rgb_latents = rgb_latents

                    target_latents = constraint_rgb_latents
                    derivative = (latents - target_latents)/sigmas
                    dt = pipe.scheduler.sigmas[pipe.scheduler.step_index + 1] - sigmas
                    prev_sample = latents + derivative * dt
                    
                    latents = prev_sample.to(dtype=torch.float16)
                    pipe.scheduler._step_index += 1

                    if t_idx == len(timesteps) - 1 or ((t_idx + 1) > num_warmup_steps and (t_idx + 1) % pipe.scheduler.order == 0):
                        progress_bar.update()


            if condition['name'] == 'R-E-R':
                raise ValueError
            elif condition['name'] == 'R-E-N':
                raise ValueError
            elif condition['name'] == 'N-E-R':
                raise ValueError
            elif condition['name'] == 'N-E-N':
                latents = latents[:, :, :, :, :]
            else:
                raise ValueError

            latents = latents.to(dtype=torch.float32)
            assert config.data.test_batch_size == 1
            
            frames = pipe._decode_vae_image_modified(
                latents=latents,
                vae=networks['vae_rgb'],
                num_frames=latents.shape[1],
                decode_chunk_size=args.decode_chunk_size,
                scaling=True,
            )
            for i in range(latents.shape[1]):
                frame = frames[0, i, :, :, :]
                save_image_from_chw_tensor(tensor=frame, path=output_rgb_paths[i])
            idx += 1