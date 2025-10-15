import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import sys
from PIL import Image

from torchvision import transforms
from transformers import AutoTokenizer
import glob
import json
from PIL import ImageFilter

def generate_jsonlist(datafolder, deg_type_list):
    json_list = []
    for deg_type in deg_type_list:
        print('deg_type', deg_type)
        if deg_type == 'unideg':
            subfolder_name = ['unideg1', 'unideg2']
        elif deg_type == 'twodeg':
            subfolder_name = ['twodeg1', 'twodeg2']
        elif deg_type == 'orideg':
            subfolder_name = ['ori_deg']
        for subfolder in subfolder_name:
            json_list.extend(glob.glob(os.path.join(datafolder, subfolder, f"*.json")))
    return json_list

class Val_image(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
    ):

        self.data = []
        with open(args.val_json, 'r') as f:
            self.data.extend(json.load(f))

        self.image_transforms = transforms.Compose(
        [
            transforms.ToTensor()
        ])

        self.gt_normalize = transforms.Normalize([0.5], [0.5])             
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = dict()
        cur_item = self.data[index]
        gt_path = cur_item['gt_path']
        lr_path = cur_item['sr_bicubic_path']
        gt =  Image.open(gt_path).convert('RGB')
        lr = Image.open(lr_path).convert('RGB')
        
        gt_pt = self.image_transforms(gt)
        lr_pt = self.image_transforms(lr)
        
        gt = self.gt_normalize(gt_pt)
        control_image = lr_pt
        example["conditioning_pixel_values"] = control_image
        example["main_prompts"] = cur_item['main_prompt']
        example["control_prompts"] = cur_item['control_prompt']
        example["pixel_values"] = gt
        example['img_name'] = cur_item['img_name']
        return example


class Val_image_syntheticDIV2k(torch.utils.data.Dataset):
    # the control prompt is derived from main prompt,
    # since no gt path in json, the gt read from the lr path
    def __init__(
        self,
        args,
    ):

        self.data = []
        with open(args.val_json, 'r') as f:
            self.data.extend(json.load(f))

        self.image_transforms = transforms.Compose(
        [
            transforms.ToTensor()
        ])

        self.gt_normalize = transforms.Normalize([0.5], [0.5])             
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = dict()
        cur_item = self.data[index]
        gt_path = cur_item['input_path']
        lr_path = cur_item['input_path']
        img_name = os.path.basename(lr_path)
        img_name = os.path.splitext(img_name)[0]
        gt =  Image.open(gt_path).convert('RGB')
        lr = Image.open(lr_path).convert('RGB')
        
        gt_pt = self.image_transforms(gt)
        lr_pt = self.image_transforms(lr)
        
        gt = self.gt_normalize(gt_pt)
        control_image = lr_pt
        example["conditioning_pixel_values"] = control_image
        example["main_prompts"] = cur_item['main_prompt']
        example["control_prompts"] = "make " + example["main_prompts"] + " clear"
        example["pixel_values"] = gt
        example['img_name'] = img_name
        return example


class BIR_UniDeg2Clean_mainprompt_controlprompt_mask(torch.utils.data.Dataset):
    # The prompt in the main body is '' and the mask description
    # The prompt in the control branch is '' and "make {mask description} clear"
    def __init__(
        self,
        args
    ):

        self.jsons = generate_jsonlist(args.train_datafolder, args.deg_type_list)  # json list
        self.data = []
        for json_file in self.jsons:
            with open(json_file, 'r') as f:
                self.data.extend(json.load(f))

        self.resolution = args.resolution
        
        self.proportion_main_empty_prompts = args.proportion_main_empty_prompts
        self.proportion_control_empty_prompts = args.proportion_control_empty_prompts
         
        self.tokenizer = args.tokenizer

        self.image_transforms = transforms.Compose(
        [
            transforms.ToTensor()
        ])


        self.gt_normalize = transforms.Normalize([0.5], [0.5])
        

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = dict()
        cur_item = self.data[index]
        gt_path = cur_item['gt_path']
        lr_path = cur_item['sr_bicubic_path']
        mask_path = cur_item['mask_path']
        gt =  Image.open(gt_path).convert('RGB')
        lr = Image.open(lr_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        mask_pt = self.image_transforms(mask)
        gt_pt = self.image_transforms(gt)
        lr_pt = self.image_transforms(lr)

        # sub_bs = float(cur_item['sub_bs'])
        # sub_ns = float(cur_item['sub_ns'])
        # sub_bs_str = f"{sub_bs:.2f}"
        # sub_ns_str = f"{sub_ns:.2f}"
        

        main_prompt = cur_item['description']
        control_prompt = "make " + main_prompt[0] + " clear"
        if random.random() < self.proportion_main_empty_prompts:
            main_prompt = ''
            control_prompt = ''
            mask_pt = torch.ones(gt_pt.shape[-2:]).unsqueeze(0)
        
        mask_pt = mask_pt.unsqueeze(0)
        mask_pt = F.interpolate(mask_pt, scale_factor=1/8, mode='nearest')
        mask_pt = mask_pt.squeeze(0)

        gt = self.gt_normalize(gt_pt)
        
        example["conditioning_pixel_values"] = lr_pt
        example["control_input_ids"] = self.tokenize_caption(control_prompt)
        example["main_input_ids"] = self.tokenize_caption(main_prompt)
        example["pixel_values"] = gt
        example['mask'] = mask_pt

        return example
        

    def tokenize_caption(self, caption):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


class BIR_UniDeg2Clean_mainprompt_controlprompt_mask_addbokeh_ratioargs(torch.utils.data.Dataset):
    # The prompt in the main body is '' and the mask description
    # The prompt in the control branch is '' and "make {mask description} clear"
    # bokeh gt can change the ratio
    def __init__(
        self,
        args
    ):

        self.jsons = generate_jsonlist(args.train_datafolder, args.deg_type_list)  # json list
        self.bokeh_jsons = generate_jsonlist(args.train_bokeh_datafolder, args.deg_type_list)  # json list
        self.data = []
        self.bokeh_data = []
        for json_file in self.jsons:
            with open(json_file, 'r') as f:
                self.data.extend(json.load(f))

        for json_file in self.bokeh_jsons:
            with open(json_file, 'r') as f:
                self.bokeh_data.extend(json.load(f))

        self.resolution = args.resolution
        
        self.proportion_main_empty_prompts = args.proportion_main_empty_prompts
        self.proportion_control_empty_prompts = args.proportion_control_empty_prompts
         
        self.tokenizer = args.tokenizer

        self.image_transforms = transforms.Compose(
        [
            transforms.ToTensor()
        ])


        self.gt_normalize = transforms.Normalize([0.5], [0.5])
        self.bokeh_ratio = args.bokeh_ratio
        self.data_ratio = 1 - self.bokeh_ratio
        

        
    def __len__(self):
        return len(self.bokeh_data)

    def __getitem__(self, index):
        example = dict()
        if random.random() < self.data_ratio:
            cur_item = self.data[index]
            main_prompt = cur_item['description']
            control_prompt = "make " + main_prompt[0] + " clear"
        else:
            cur_item = self.bokeh_data[index]
            descript = cur_item['description']
            main_prompt = descript[0] + " in front of bokeh background" #modified in 20250220
            control_prompt = "make " + descript[0] + " clear and keep other part bokeh blur"

        gt_path = cur_item['gt_path']
        lr_path = cur_item['sr_bicubic_path']
        mask_path = cur_item['mask_path']
        gt =  Image.open(gt_path).convert('RGB')
        lr = Image.open(lr_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        mask_pt = self.image_transforms(mask)
        gt_pt = self.image_transforms(gt)
        lr_pt = self.image_transforms(lr)

        # sub_bs = float(cur_item['sub_bs'])
        # sub_ns = float(cur_item['sub_ns'])
        # sub_bs_str = f"{sub_bs:.2f}"
        # sub_ns_str = f"{sub_ns:.2f}"
        

        #main_prompt = cur_item['description']
        #control_prompt = "make " + main_prompt[0] + " clear"  #modified in 20250220
        if random.random() < self.proportion_main_empty_prompts:
            main_prompt = ''
            control_prompt = ''
            mask_pt = torch.ones(gt_pt.shape[-2:]).unsqueeze(0)
        
        mask_pt = mask_pt.unsqueeze(0)
        mask_pt = F.interpolate(mask_pt, scale_factor=1/8, mode='nearest')
        mask_pt = mask_pt.squeeze(0)

        gt = self.gt_normalize(gt_pt)
        
        example["conditioning_pixel_values"] = lr_pt
        example["control_input_ids"] = self.tokenize_caption(control_prompt)
        example["main_input_ids"] = self.tokenize_caption(main_prompt)
        example["pixel_values"] = gt
        example['mask'] = mask_pt

        return example
        

    def tokenize_caption(self, caption):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


class BIR_UniDeg2Clean_mainprompt_controlprompt_mask_addbokeh_v2(torch.utils.data.Dataset):
    # The prompt in the main body is '' and the mask description
    # The prompt in the control branch is '' and "make {mask description} clear"
    # 50% bokeh gt
    def __init__(
        self,
        args
    ):

        self.jsons = generate_jsonlist(args.train_datafolder, args.deg_type_list)  # json list
        self.bokeh_jsons = generate_jsonlist(args.train_bokeh_datafolder, args.deg_type_list)  # json list
        self.data = []
        self.bokeh_data = []
        for json_file in self.jsons:
            with open(json_file, 'r') as f:
                self.data.extend(json.load(f))

        for json_file in self.bokeh_jsons:
            with open(json_file, 'r') as f:
                self.bokeh_data.extend(json.load(f))

        self.resolution = args.resolution
        
        self.proportion_main_empty_prompts = args.proportion_main_empty_prompts
        self.proportion_control_empty_prompts = args.proportion_control_empty_prompts
         
        self.tokenizer = args.tokenizer

        self.image_transforms = transforms.Compose(
        [
            transforms.ToTensor()
        ])


        self.gt_normalize = transforms.Normalize([0.5], [0.5])
        

        
    def __len__(self):
        return len(self.bokeh_data)

    def __getitem__(self, index):
        example = dict()
        if random.random() < 0.5:
            cur_item = self.data[index]
            main_prompt = cur_item['description']
            control_prompt = "make " + main_prompt[0] + " clear"
        else:
            cur_item = self.bokeh_data[index]
            descript = cur_item['description']
            main_prompt = descript[0]  #modified in 20250221
            control_prompt = "make " + descript[0] + " clear and keep other part bokeh blur"

        gt_path = cur_item['gt_path']
        lr_path = cur_item['sr_bicubic_path']
        mask_path = cur_item['mask_path']
        gt =  Image.open(gt_path).convert('RGB')
        lr = Image.open(lr_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        mask_pt = self.image_transforms(mask)
        gt_pt = self.image_transforms(gt)
        lr_pt = self.image_transforms(lr)

        # sub_bs = float(cur_item['sub_bs'])
        # sub_ns = float(cur_item['sub_ns'])
        # sub_bs_str = f"{sub_bs:.2f}"
        # sub_ns_str = f"{sub_ns:.2f}"
        

        #main_prompt = cur_item['description']
        #control_prompt = "make " + main_prompt[0] + " clear"  #modified in 20250220
        if random.random() < self.proportion_main_empty_prompts:
            main_prompt = ''
            control_prompt = ''
            mask_pt = torch.ones(gt_pt.shape[-2:]).unsqueeze(0)
        
        mask_pt = mask_pt.unsqueeze(0)
        mask_pt = F.interpolate(mask_pt, scale_factor=1/8, mode='nearest')
        mask_pt = mask_pt.squeeze(0)

        gt = self.gt_normalize(gt_pt)
        
        example["conditioning_pixel_values"] = lr_pt
        example["control_input_ids"] = self.tokenize_caption(control_prompt)
        example["main_input_ids"] = self.tokenize_caption(main_prompt)
        example["pixel_values"] = gt
        example['mask'] = mask_pt

        return example
        

    def tokenize_caption(self, caption):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


if __name__ == "__main__":
    datafolder = '/home/notebook/data/personal/S9048593/LocalRestoration_Data'
    deg_type_list = ['unideg']
    json_list = generate_jsonlist(datafolder, deg_type_list)
    print(json_list)
