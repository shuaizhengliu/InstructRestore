from diffusers import UniPCMultistepScheduler, DDIMScheduler

import os
import sys
from pipeline_instruct_restore import StableDiffusionControlNetPipeline 
from models.unet_2d_condition import UNet2DConditionModel
from models.controlnet_maskdecoder import ControlNetModel
from diffusers.utils import load_image
import torch
import numpy as np
import glob
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
import json
from PIL import Image
import argparse
import re


generator = torch.manual_seed(0)
parser = argparse.ArgumentParser(description='Inference control instruction with main scale and LRE.')
parser.add_argument('--cfg_scale', type=float, default=5.5, help='CFG scale')
parser.add_argument('--mask_init_step', type=int, default=0, help='Mask init step')
parser.add_argument('--align_method', type=str, default='wavelet', choices=['wavelet', 'adain', 'nofix'], help='Alignment method')
parser.add_argument('--start_point', type=str, default='lr', help='Start point')
parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
parser.add_argument('--instruction', type=str, required=True, help='instruction')
parser.add_argument('--save_folder', type=str, required=True, help='Folder to save the results')
parser.add_argument('--step', type=int, required=True, help='infer step')
parser.add_argument('--save_mask', action='store_true')
parser.add_argument('--controlnet_path', type=str)
parser.add_argument('--base_model_path', type=str)


args = parser.parse_args()

cfg_scale = args.cfg_scale
mask_init_step = args.mask_init_step
align_method = args.align_method
start_point = args.start_point
save_folder = args.save_folder
step = args.step
base_model_path = args.base_model_path

controlnet = ControlNetModel.from_pretrained(args.controlnet_path)
unet = UNet2DConditionModel.from_pretrained(os.path.join(base_model_path, 'unet'))
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, unet=unet
).to('cuda')

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()


save_folder = os.path.join(save_folder, 'image_result')
if args.save_mask:
    save_mask_folder = os.path.join(save_folder, 'predict_mask')
    if not os.path.exists(save_mask_folder):
        os.makedirs(save_mask_folder)
else:
    save_mask_folder = ''

if not os.path.exists(save_folder):
    os.makedirs(save_folder)


def process_single_image(image_path, main_prompt, control_prompt, mask_inner_scale, mask_outer_scale):
    cur_img_prefix = os.path.splitext(os.path.basename(image_path))[0]
    validation_image = Image.open(image_path).convert("RGB")
    control_image = load_image(image_path)
    control_prompt = f'make {main_prompt} clear'
    image,  _= pipe(
        control_prompt, main_prompt, main_scale=1, num_inference_steps=step, generator=generator, image=control_image, guidance_scale=cfg_scale, mask_inner_scale=mask_inner_scale, mask_outer_scale=mask_outer_scale, start_point=start_point, save_mask=args.save_mask, mask_init_step=mask_init_step, cur_img_prefix=cur_img_prefix, save_mask_folder=save_mask_folder)
    image = image[0]
    
    if align_method == 'nofix':
        image = image
    else:
        if align_method == 'wavelet':
            image = wavelet_color_fix(image, validation_image)
        elif align_method == 'adain':
            image = adain_color_fix(image, validation_image)
    return image

def parse_instruction(instruction):
    """
    Parse an instruction according to specified templates.
    Returns main_prompt, control_prompt, mask_inner_scale, mask_outer_scale.
    Raises ValueError if the instruction does not match any template.
    """

    templates = [
        # Template 1: make {main_prompt} clear with {mask_inner_scale} and keep other parts with {mask_outer_scale}
        re.compile(
            r'^make (.+?) clear with ([\d\.]+) and keep other parts clear with ([\d\.]+)$'
        ),
        # Template 2: make {main_prompt} clear with {mask_inner_scale} and keep other parts bokeh blur with {mask_outer_scale}
        re.compile(
            r'^make (.+?) clear with ([\d\.]+) and keep other parts bokeh blur with ([\d\.]+)$'
        )
    ]

    instruction = instruction.strip()
    for idx, pattern in enumerate(templates):
        match = pattern.fullmatch(instruction)
        if match:
            raw_main_prompt = match.group(1).strip()
            mask_inner_scale = float(match.group(2))
            mask_outer_scale = float(match.group(3))
            
            if idx == 1:
                # Second template: add "in front of bokeh background"
                main_prompt = f"{raw_main_prompt} in front of bokeh background"
                control_prompt = f"make {raw_main_prompt} clear and keep other part bokeh blur"
            else:
                main_prompt = raw_main_prompt
                control_prompt = f"make {raw_main_prompt} clear"
            return main_prompt, control_prompt, mask_inner_scale, mask_outer_scale

    raise ValueError(
        "Instruction does not match the required template. Please follow: "
        "'make {{main_prompt}} clear with {{mask_inner_scale}} and keep other parts clear with {{mask_outer_scale}}' "
        "or "
        "'make {{main_prompt}} clear with {{mask_inner_scale}} and keep other parts bokeh blur with {{mask_outer_scale}}'."
    )

image_path = args.image_path
instruction = args.instruction
main_prompt, control_prompt, mask_inner_scale, mask_outer_scale = parse_instruction(instruction)
result_image = process_single_image(image_path, main_prompt, control_prompt, mask_inner_scale, mask_outer_scale)
image_name = os.path.basename(image_path)
output_path = os.path.join(save_folder, image_name)
result_image.save(output_path)  

