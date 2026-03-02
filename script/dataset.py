from torch.utils.data import Dataset
import torch
import os
import glob
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from pathlib import Path
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class EventAndVideoDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, mode, event_num=None, img_mode='rgb', img_format='png', event_suffix='event_corrected', random_crop=[], time_per_frame=None, condition_mode=["R-E-R"]):
        try:
            items = os.listdir(folder_path)
            subdirectories = [os.path.join(folder_path, item) for item in items if os.path.isdir(os.path.join(folder_path, item))]
        except Exception as e:
            print(f"Error: {e}")
            exit()
        self.clips = subdirectories
        self.img_format = img_format
        self.mode = mode
        self.event_num = event_num
        self.time_per_frame = time_per_frame
        self.random_crop = random_crop
        self.condition_modes = condition_mode
        self.img_mode = img_mode
        self.event_suffix = event_suffix
    
    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        example = {}
        target_rgb_paths = []
        target_gray_paths = []
        target_event_paths = []
        # 
        clip_path = self.clips[idx]
        rgb_paths = glob.glob(os.path.join(clip_path, 'rgb', '*.' + self.img_format))
        rgb_paths.sort()
        gray_paths = glob.glob(os.path.join(clip_path, 'gray', '*.' + self.img_format))
        gray_paths.sort()
        if len(gray_paths) == 0:
            # print('WARNING: gray_paths is blank!')
            gray_paths = rgb_paths
        # 
        if self.mode == 'train':
            start_idx = random.randint(1, (len(rgb_paths) - self.event_num)) # event is less than rgb by 1
        elif self.mode == 'test':
            start_idx = 1
        else:
            raise ValueError("Error")
        # 
        event_paths = glob.glob(os.path.join(clip_path, self.event_suffix, '*.npz'))
        event_paths.sort()
        # event_data = np.load(os.path.join(clip_path, 'event', 'data.npz'))
        # ts = event_data['t'].astype(np.float32)
        # xs = event_data['x'].astype(np.float32)
        # ys = event_data['y'].astype(np.float32)
        # ps = event_data['p'].astype(np.float32)
        # 
        event_datas = []
        rgb_raws = []
        gray_raws = []
        # rgb_vaes = []
        # rgb_clips = []
        # 
        temp_raw, _, _ = self.read_rgb(path=rgb_paths[0], device=torch.device('cpu'))
        _, original_height, original_width = temp_raw.shape
        if len(self.random_crop) == 2:
            crop_height, crop_width = self.random_crop
        else:
            crop_height = original_height
            crop_width = original_width

        # Ensure crop size is divisible by 8 (VAE downsampling factor)
        # If crop > original, use the largest multiple of 8 that fits within original
        if crop_height > original_height:
            crop_height = (original_height // 64) * 64
        if crop_width > original_width:
            crop_width = (original_width // 64) * 64

        top, left = self.get_random_crop_coord(original_height, original_width, crop_height, crop_width)
        # 
        for i in range(start_idx, start_idx + self.event_num):
            # start_frame_index = np.searchsorted(ts, (i - 1) * self.time_per_frame + 1)
            # end_frame_index = np.searchsorted(ts, i * self.time_per_frame + 1)
            # event_data = self.events_to_triple_grid_pytorch(ts=ts[start_frame_index:end_frame_index], xs=xs[start_frame_index:end_frame_index], ys=ys[start_frame_index:end_frame_index], ps=ps[start_frame_index:end_frame_index], width=original_width, height=original_height, device=torch.device("cpu")).to(torch.float32)
            # event_datas.append(event_data[:, top:top + crop_height, left:left + crop_width])
            event_data = np.load(event_paths[i - 1])['data'] # [-N, N] (3, H, W)
            event_data = torch.from_numpy(event_data).to(torch.float32)
            event_datas.append(event_data[:, top:top + crop_height, left:left + crop_width])
            target_event_paths.append(event_paths[i - 1])
            if i == start_idx:
                rgb_raw, _, _ = self.read_rgb(path=rgb_paths[i - 1], device=torch.device('cpu'))
                rgb_raws.append(rgb_raw[:, top:top + crop_height, left:left + crop_width])
                target_rgb_paths.append(rgb_paths[i - 1])
                # 
                gray_raw, _, _ = self.read_rgb(path=gray_paths[i - 1], device=torch.device('cpu'))
                gray_raws.append(gray_raw[:, top:top + crop_height, left:left + crop_width])
                target_gray_paths.append(gray_paths[i - 1])
                # 
                # rgb_vaes.append(rgb_vae[:, top:top + crop_height, left:left + crop_width])
                # rgb_clips.append(rgb_clip)
            rgb_raw, _, _ = self.read_rgb(path=rgb_paths[i], device=torch.device('cpu'))
            rgb_raws.append(rgb_raw[:, top:top + crop_height, left:left + crop_width])
            target_rgb_paths.append(rgb_paths[i])
            # 
            gray_raw, _, _ = self.read_rgb(path=gray_paths[i], device=torch.device('cpu'))
            gray_raws.append(gray_raw[:, top:top + crop_height, left:left + crop_width])
            target_gray_paths.append(gray_paths[i])
            # rgb_vaes.append(rgb_vae[:, top:top + crop_height, left:left + crop_width])
            # rgb_clips.append(rgb_clip)
        condition_modes = random.choice(self.condition_modes)
        example = {'event': torch.stack(event_datas, dim=0).to("cuda"), 'rgb_raw': torch.stack(rgb_raws, dim=0).to("cuda"), 'gray_raw': torch.stack(gray_raws, dim=0).to("cuda"), 'condition_mode': condition_modes, 'target_rgb_paths': target_rgb_paths, 'target_gray_paths': target_gray_paths, 'target_event_paths': target_event_paths, 'folder_path': clip_path}
        # example = {'event': torch.stack(event_datas, dim=0).to("cuda"), 'rgb_raw': torch.stack(rgb_raws, dim=0).to("cuda"), 'rgb_vae': torch.stack(rgb_vaes, dim=0).to("cuda"), 'rgb_clip': torch.stack(rgb_clips, dim=0).to("cuda")}
        return example

    def get_random_crop_coord(self, original_height, original_width, crop_height, crop_width):
        if crop_height > original_height or crop_width > original_width:
            raise ValueError("Crop size should be smaller than the original tensor size")
        top = torch.randint(0, original_height - crop_height + 1, (1,)).item()
        left = torch.randint(0, original_width - crop_width + 1, (1,)).item()
        return top, left

    def random_crop_tensor(self, tensor, random_crop):
        if len(random_crop) == 0:
            return tensor
        
        _, height, width = tensor.shape
        
        crop_height, crop_width = random_crop
        
        if crop_height > height or crop_width > width:
            raise ValueError("Crop size should be smaller than the original tensor size")

        top = torch.randint(0, height - crop_height + 1, (1,)).item()
        left = torch.randint(0, width - crop_width + 1, (1,)).item()
        
        cropped_tensor = tensor[:, top:top + crop_height, left:left + crop_width]

        return cropped_tensor

    def read_rgb(
            self, 
            path, 
            device, 
            noise_aug_strength=0.02,
        ):
        output = {}
        img = Image.open(path)
        if img.mode == 'L':
            img = img.convert('RGB')

        rgb_data_per_frame_ndarray = np.array(img).astype(np.float32) / 255.0  
        rgb_data_per_frame_tensor = torch.from_numpy(rgb_data_per_frame_ndarray.transpose(2, 0, 1))
        rgb_data_per_frame_tensor = 2.0 * rgb_data_per_frame_tensor - 1.0
        rgb_raw = rgb_data_per_frame_tensor


        # temp = _resize_with_antialiasing(rgb_data_per_frame_tensor.unsqueeze(0), (224, 224))
        rgb_clip = None # (temp.squeeze(0) + 1.0) / 2.0

        # noise = self.randn_tensor(rgb_data_per_frame_tensor.shape, generator=torch.manual_seed(42), device=rgb_data_per_frame_tensor.device, dtype=rgb_data_per_frame_tensor.dtype)
        rgb_vae = None # rgb_data_per_frame_tensor + noise_aug_strength * noise
        

        return rgb_raw, rgb_vae, rgb_clip

    def randn_tensor(
            self,
            shape,
            generator = None,
            device = None,
            dtype = None,
            layout = None,
        ):
        """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
        passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
        is always created on the CPU.
        """
        # device on which tensor is created defaults to device
        rand_device = device
        batch_size = shape[0]

        layout = layout or torch.strided
        device = device or torch.device("cpu")

        if generator is not None:
            gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
            if gen_device_type != device.type and gen_device_type == "cpu":
                rand_device = "cpu"
                if device != "mps":
                    raise ValueError(
                        f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                        f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                        f" slighly speed up this function by passing a generator that was created on the {device} device."
                    )
            elif gen_device_type != device.type and gen_device_type == "cuda":
                raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

        # make sure generator list of length 1 is treated like a non-list
        if isinstance(generator, list) and len(generator) == 1:
            generator = generator[0]

        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            latents = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
                for i in range(batch_size)
            ]
            latents = torch.cat(latents, dim=0).to(device)
        else:
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

        return latents

    def events_to_triple_grid_pytorch(self, ts, xs, ys, ps, width, height, device):
        def accumulate_events(xs, ys, ts, ps, H, W):
            out = np.zeros((H, W))
            out_positive = np.zeros((H, W))
            out_negative = np.zeros((H, W))
            for i in range(len(xs)):
                x, y, t, p = xs[i], ys[i], ts[i], ps[i]
                out[int(y), int(x)] += p
                if p == 1.0:
                    out_positive[int(y), int(x)] += p
                elif p == -1.0:
                    out_negative[int(y), int(x)] += p
                else:
                    raise ValueError("The value does not meet condition A or B")
            output = np.concatenate((out[None, ...], out_positive[None, ...], out_positive[None, ...]), axis=0)
            return output

        ps[ps == 0] = -1

        frame_event = accumulate_events(
            xs=xs,
            ys=ys, 
            ts=ts, 
            ps=ps, 
            H=height,
            W=width,
        )
        frame_event = torch.from_numpy(frame_event).to(torch.int8)
        return frame_event


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        mask_size=64,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        aug=True,
    ):
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        for concept in concepts_list:
            inst_img_path = [
                (x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()
            ]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = list(zip(class_images_path, class_prompt))
                self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top : top + inner, left : left + inner, :] = image
            mask[
                top // factor + 1 : (top + scale) // factor - 1, left // factor + 1 : (left + scale) // factor - 1
            ] = 1.0
        return instance_image, mask

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.flip(instance_image)

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"])
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example

# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output

def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out

def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)

def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: List[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out

def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding