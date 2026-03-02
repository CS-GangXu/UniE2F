from skimage.metrics import structural_similarity as SSIM
import os
import torch
import csv
import os
import argparse
import torch
import glob
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance, calculate_activation_statistics
from script.util import load_config, get_files_with_format

import numpy as np
import math
import lpips
import cv2

import os
import torch
import glob
import time
import csv
from script.util import calculate_psnr, calculate_ssim, get_files_with_format, get_input_ndarray


parser = argparse.ArgumentParser(description="Script to Infer Stable Video Diffusion.")
parser.add_argument("--batch_size", default=50, type=int)
parser.add_argument("--dims", default=2048, type=int)
parser.add_argument("--num_workers", default=8, type=int)
# 
parser.add_argument("--config", default='experiments.svd/2025021401_FinalVersionExtension/aaai_type@vfp.yaml', type=str)

args = parser.parse_args()
config = load_config(args.config)
config.output_folder = os.path.join(os.path.dirname(args.config), os.path.basename(args.config).split('.')[0])
config.log_folder = os.path.join(os.path.dirname(args.config), os.path.basename(args.config).split('.')[0], 'log')
config.visualization_folder = os.path.join(os.path.dirname(args.config), os.path.basename(args.config).split('.')[0], 'visualization')
if os.path.exists(config.log_folder) == False:
    os.makedirs(config.log_folder)

loss_fn_vgg = lpips.LPIPS(net='vgg').to("cuda")



for condition in config.data.condition:
    pd_folder_path = os.path.join(config.visualization_folder, condition['name'])
    gt_folder_path = config.data.test_folder_path
    overall_performance_path = os.path.join(config.log_folder, 'overall_performance_' + condition['name'] + '.csv')
    clip_performance_path = os.path.join(config.log_folder, 'clip_performance_' + condition['name'] + '.csv')
    individual_performance_path = os.path.join(config.log_folder, 'individual_performance_' + condition['name'] + '.csv')
    if condition['name'] == 'N-E-N':
        drop_first_gt = True
    
    pd_file_paths = get_files_with_format(pd_folder_path, '.png', drop_first=False)
    pd_file_paths.sort()
    gt_file_paths = []
    for pd_file_path in pd_file_paths:
        gt_file_path = pd_file_path.replace(pd_folder_path, gt_folder_path).replace('.png', '.' + config['data']['img_format'])
        gt_file_paths.append(gt_file_path)

    # gt_file_paths = get_files_with_format(gt_folder_path, '.' + config['data']['img_format'], drop_first=drop_first_gt)
    # pd_file_paths = get_files_with_format(pd_folder_path, '.png', drop_first=False)
    # gt_file_paths.sort()
    # pd_file_paths.sort()
    # assert len(gt_file_paths) == len(pd_file_paths)
    # for i in range(len(gt_file_paths)):
    #     if gt_file_paths[i].replace(gt_folder_path, '').replace('.' + config['data']['img_format'], '.format') == pd_file_paths[i].replace(pd_folder_path, '').replace('.png', '.format'):
    #         pass
    #     else:
    #         raise ValueError
    
    avg_psnr_rgb = 0.0
    avg_psnr_gray = 0.0
    avg_ssim_rgb = 0.0
    avg_ssim_gray = 0.0
    avg_mse_rgb = 0.0
    avg_mse_gray = 0.0
    avg_lpips_rgb = 0.0
    avg_lpips_gray = 0.0
    overall_performances = []
    individual_performances = []
    clip_performance_dict = {}
    clip_performances = []
    idx = 0

    for i in range(len(gt_file_paths)):
        print(f'processing the {i:07d}-th object in {pd_folder_path:s}')

        gt_rgb_ndarray, gt_gray_ndarray = get_input_ndarray(gt_file_paths[i])
        pd_rgb_ndarray, pd_gray_ndarray = get_input_ndarray(pd_file_paths[i])
        
        if gt_rgb_ndarray.shape != pd_rgb_ndarray.shape:
            pd_rgb_ndarray = cv2.resize(pd_rgb_ndarray, (gt_rgb_ndarray.shape[1], gt_rgb_ndarray.shape[0]), interpolation=cv2.INTER_AREA)
            pd_gray_ndarray = cv2.resize(pd_gray_ndarray, (gt_gray_ndarray.shape[1], gt_gray_ndarray.shape[0]), interpolation=cv2.INTER_AREA)

        gt_rgb_tensor = torch.from_numpy(np.transpose(gt_rgb_ndarray/255, (2, 0, 1))).unsqueeze(0) # 0~1 1CHW RGB FLOAT32
        gt_rgb_tensor = (gt_rgb_tensor * 2) - 1 # -1~1 1CHW RGB FLOAT32
        pd_rgb_tensor = torch.from_numpy(np.transpose(pd_rgb_ndarray/255, (2, 0, 1))).unsqueeze(0) # 0~1 1CHW RGB FLOAT32
        pd_rgb_tensor = (pd_rgb_tensor * 2) - 1 # -1~1 1CHW RGB FLOAT32


        gt_gray_tensor = torch.from_numpy(gt_gray_ndarray/255).unsqueeze(0).unsqueeze(0) # 0~1 11HW GRAY FLOAT32
        gt_gray_tensor = (gt_gray_tensor * 2) - 1 # -1~1 11HW GRAY FLOAT32
        gt_gray_tensor = gt_gray_tensor.repeat(1, 3, 1, 1) # -1~1 1CHW GRAY FLOAT32
        pd_gray_tensor = torch.from_numpy(pd_gray_ndarray/255).unsqueeze(0).unsqueeze(0) # 0~1 11HW GRAY FLOAT32
        pd_gray_tensor = (pd_gray_tensor * 2) - 1 # -1~1 11HW GRAY FLOAT32
        pd_gray_tensor = pd_gray_tensor.repeat(1, 3, 1, 1) # -1~1 1CHW GRAY FLOAT32

        mse_rgb = np.mean(((pd_rgb_ndarray.astype(np.float64)/255) - (gt_rgb_ndarray.astype(np.float64)/255)) ** 2)
        mse_gray = np.mean(((pd_gray_ndarray.astype(np.float64)/255) - (gt_gray_ndarray.astype(np.float64)/255)) ** 2)
        psnr_rgb = calculate_psnr(pd_rgb_ndarray, gt_rgb_ndarray) # 0~255 HWC RGB UINT8
        psnr_gray = calculate_psnr(pd_gray_ndarray, gt_gray_ndarray) # 0~255 HWC RGB UINT8
        ssim_rgb = calculate_ssim(pd_rgb_ndarray, gt_rgb_ndarray) # 0~255 HWC RGB UINT8
        ssim_gray = calculate_ssim(pd_gray_ndarray, gt_gray_ndarray) # 0~255 HWC RGB UINT8
        
        lpips_rgb = loss_fn_vgg(pd_rgb_tensor.to(torch.float32).to("cuda"), gt_rgb_tensor.to(torch.float32).to("cuda")).item() # -1~1 1CHW RGB
        lpips_gray = loss_fn_vgg(pd_gray_tensor.to(torch.float32).to("cuda"), gt_gray_tensor.to(torch.float32).to("cuda")).item() # -1~1 1CHW GRAY
        # 
        individual_performances.append(
            {
                "gt_file_path": gt_file_paths[i],
                "pd_file_path": pd_file_paths[i],
                "psnr_rgb": psnr_rgb,
                "psnr_gray": psnr_gray, 
                "ssim_rgb": ssim_rgb, 
                "ssim_gray": ssim_gray, 
                "mse_rgb": mse_rgb,
                "mse_gray": mse_gray,
                "lpips_rgb": lpips_rgb,
                "lpips_gray": lpips_gray,
            }
        )
        clip_name = pd_file_paths[i].split('/')[-3]
        if clip_name not in clip_performance_dict.keys():
            clip_performance_dict[clip_name] = {
                'psnr_rgb': [],
                'psnr_gray': [],
                'ssim_rgb': [],
                'ssim_gray': [],
                'mse_rgb': [],
                'mse_gray': [],
                'lpips_rgb': [],
                'lpips_gray': [],
            }
        clip_performance_dict[clip_name]['psnr_rgb'].append(psnr_rgb)
        clip_performance_dict[clip_name]['psnr_gray'].append(psnr_gray)
        clip_performance_dict[clip_name]['ssim_rgb'].append(ssim_rgb)
        clip_performance_dict[clip_name]['ssim_gray'].append(ssim_gray)
        clip_performance_dict[clip_name]['mse_rgb'].append(mse_rgb)
        clip_performance_dict[clip_name]['mse_gray'].append(mse_gray)
        clip_performance_dict[clip_name]['lpips_rgb'].append(lpips_rgb)
        clip_performance_dict[clip_name]['lpips_gray'].append(lpips_gray)
        # 
        avg_psnr_rgb += psnr_rgb
        avg_psnr_gray += psnr_gray
        avg_ssim_rgb += ssim_rgb
        avg_ssim_gray += ssim_gray
        avg_mse_rgb += mse_rgb
        avg_mse_gray += mse_gray 
        avg_lpips_rgb += lpips_rgb
        avg_lpips_gray += lpips_gray
        idx += 1
    
    avg_psnr_rgb = avg_psnr_rgb / idx
    avg_psnr_gray = avg_psnr_gray / idx
    avg_ssim_rgb = avg_ssim_rgb / idx
    avg_ssim_gray = avg_ssim_gray / idx
    avg_mse_rgb = avg_mse_rgb / idx
    avg_mse_gray = avg_mse_gray / idx
    avg_lpips_rgb = avg_lpips_rgb / idx
    avg_lpips_gray = avg_lpips_gray / idx

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block_idx]).to(torch.device("cuda"))
    m_pd, s_pd = calculate_activation_statistics(pd_file_paths, model, args.batch_size, args.dims, torch.device("cuda"), args.num_workers)
    m_gt, s_gt = calculate_activation_statistics(gt_file_paths, model, args.batch_size, args.dims, torch.device("cuda"), args.num_workers)
    fid_value = calculate_frechet_distance(m_pd, s_pd, m_gt, s_gt)
    
    for clip_name, clip_performance in clip_performance_dict.items():
        clip_performances.append(
            {
                'clip': clip_name,
                'psnr_rgb': np.mean(clip_performance['psnr_rgb']),
                'psnr_gray': np.mean(clip_performance['psnr_gray']),
                'ssim_rgb': np.mean(clip_performance['ssim_rgb']),
                'ssim_gray': np.mean(clip_performance['ssim_gray']),
                'mse_rgb': np.mean(clip_performance['mse_rgb']),
                'mse_gray': np.mean(clip_performance['mse_gray']),
                'lpips_rgb': np.mean(clip_performance['lpips_rgb']),
                'lpips_gray': np.mean(clip_performance['lpips_gray']),
            }
        )

    overall_performances.append(
        {
            "avg_mse_gray": round(avg_mse_gray, 4),
            "avg_ssim_gray": round(avg_ssim_gray, 3), 
            "avg_lpips_gray": round(avg_lpips_gray, 3),
            "avg_lpips_rgb": round(avg_lpips_rgb, 3),
            "fid": fid_value,
            "avg_psnr_rgb": round(avg_psnr_rgb, 2),
            "avg_psnr_gray": round(avg_psnr_gray, 2),
            "avg_ssim_rgb": round(avg_ssim_rgb, 3),
            "avg_mse_rgb": round(avg_mse_rgb, 4)
        }
    )

    overall_fields = overall_performances[0].keys()
    with open(overall_performance_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=overall_fields)
        writer.writeheader()
        writer.writerows(overall_performances)

    clip_fields = clip_performances[0].keys()
    with open(clip_performance_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=clip_fields)
        writer.writeheader()
        writer.writerows(clip_performances)

    individual_fields = individual_performances[0].keys()
    with open(individual_performance_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=individual_fields)
        writer.writeheader()
        writer.writerows(individual_performances)


# start_time = time.time()
# current_time = time.time()
# elapsed_time = current_time - start_time
# hours = int(elapsed_time // 3600)
# minutes = int((elapsed_time % 3600) // 60)
# seconds = int(elapsed_time % 60)
# print(f'Time difference: {hours:02}h-{minutes:02}m-{seconds:02}s ###: {idx:05d}')