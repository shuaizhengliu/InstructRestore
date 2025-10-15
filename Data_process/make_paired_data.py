import os
import sys
sys.path.append(os.getcwd())
import cv2

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize, Compose
from pytorch_lightning import seed_everything

import argparse
from BIR_dataset import custom_collate_fn, BIRDataset_TwoDegradation
from mmrealsr.archs.mmrealsr_arch import MMRRDBNet_test_Score
from basicsr.utils.options import ordered_yaml

import yaml
import time
import json

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
#from basicsr.data.transforms import paired_random_crop, triplet_random_crop
#from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
import random
import torch.nn.functional as F


def realesrgan_degradation(batch,  args_degradation, use_usm=True, sf=4, resize_lq=True):
    jpeger = DiffJPEG(differentiable=False).cuda()
    usm_sharpener = USMSharp().cuda()  # do usm sharpening
    im_gt = batch['gt'].cuda()
    if use_usm:
        im_gt = usm_sharpener(im_gt)
    im_gt = im_gt.to(memory_format=torch.contiguous_format).float()

    lq_list = []
    for i in range(2):
        kernel1 = batch[f'kernel1_{i}'].cuda()
        kernel2 = batch[f'kernel2_{i}'].cuda()
        sinc_kernel = batch[f'sinc_kernel_{i}'].cuda()

        ori_h, ori_w = im_gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel1)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                args_degradation['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, args_degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(args_degradation['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = args_degradation['gray_noise_prob']
        if random.random() < args_degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=args_degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=args_degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if random.random() < args_degradation['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                args_degradation['resize_prob2'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, args_degradation['resize_range2'][1])
        elif updown_type == 'down':
            scale = random.uniform(args_degradation['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(int(ori_h / sf * scale),
                        int(ori_w / sf * scale)),
                mode=mode,
                )
        # add noise
        gray_noise_prob = args_degradation['gray_noise_prob2']
        if random.random() < args_degradation['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=args_degradation['noise_range2'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=args_degradation['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
                )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if random.random() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // sf,
                            ori_w // sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // sf,
                            ori_w // sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)

        # clamp and round
        im_lq = torch.clamp(out, 0, 1.0)
        lq_list.append(im_lq)
    
    '''
    # random crop (have been cropped in the Dataset, this step could be ignored)
    gt_size = args_degradation['gt_size']
    im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, sf)
    lq, gt = im_lq, im_gt


    gt = torch.clamp(gt, 0, 1)
    lq = torch.clamp(lq, 0, 1)
    '''
    lq_mask = batch['mask'].cuda()
    lq_mask = F.interpolate(lq_mask, size=(ori_h // sf, ori_w // sf), mode='nearest')

    return lq_list, im_gt, lq_mask


def postsave_batchitem(lr, sr_bicubic, lr_save_folder, sr_bicubic_save_folder, save_name, gt_path, mask_img_path, json_item):

    lr_path = os.path.join(lr_save_folder, save_name)
    sr_bicubic_path = os.path.join(sr_bicubic_save_folder, save_name)
    # save the gt, lr, sr_bicubic and mask
    
    cv2.imwrite(sr_bicubic_path, 255*sr_bicubic.detach().cpu().squeeze().permute(1,2,0).numpy()[..., ::-1])
    cv2.imwrite(lr_path, 255*lr.detach().cpu().squeeze().permute(1,2,0).numpy()[..., ::-1])
    

    # If the number of mask description is greater than 1, just copy the "subject" to the description
    mask_description = json_item['corrected_mask_description']
    if len(mask_description) > 1:
        mask_description = [json_item['subject']]

    cur_json_item = {'gt_path': gt_path,
     'lr_path': lr_path,
     'mask_path': mask_img_path,
     'sr_bicubic_path': sr_bicubic_path,
     'description': mask_description,
     'subject': json_item['subject']
     }
    
    return cur_json_item

def append_to_json_file(file_path, new_data):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if not isinstance(data, list):
        data = []

    data.extend(new_data)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument("--original_data_folder", type=str, default='', help='the path of the original data folder.')
parser.add_argument("--gt_type", type=str, default='Normal', help='the type of the gt data. Normal or Bokeh')
parser.add_argument("--save_dir", type=str, default='Training_data', help='the save path of the training dataset.')
parser.add_argument("--batch_size", type=int, default=1, help='smaller batch size means much time but more extensive degradation for making the training dataset.')  
parser.add_argument("--epoch", type=int, default=3, help='decide how many epochs to create for the dataset.')
parser.add_argument("--num_instances", type=int, default=1, help='decide how many instances to create for the dataset.')
parser.add_argument("--instance_id", type=int, default=0, help='decide which instance to create for the dataset.')
args = parser.parse_args()

print(f'====== START GPU: {args.instance_id} =========')
seed_everything(24 + 500 * (args.instance_id))  # set a different seed for each instance

args_training_dataset = {}

args_training_dataset['source_image_folder']= {'LSDIR': os.path.join(args.original_data_folder, 'Normal/LSDIR/GT'), 'Entity_Seg': os.path.join(args.original_data_folder, 'Normal/Entity_Seg/GT'), 'EBB':os.path.join(args.original_data_folder, 'Bokeh/EBB!/GT')}
args_training_dataset['source_mask_folder']= {'LSDIR': os.path.join(args.original_data_folder, 'Normal/LSDIR/masks'), 'Entity_Seg': os.path.join(args.original_data_folder, 'Normal/Entity_Seg/masks'), 'EBB': os.path.join(args.original_data_folder, 'Bokeh/EBB!/masks')}
args_training_dataset['json_folder'] = {'LSDIR': os.path.join(args.original_data_folder, 'Normal/LSDIR/annotated_jsons'), 'Entity_Seg': os.path.join(args.original_data_folder, 'Normal/Entity_Seg/annotated_jsons'), 'EBB': os.path.join(args.original_data_folder, 'Bokeh/EBB!/annotated_jsons')}


if args.gt_type == 'Normal':
    json_list = []
    json_list.append(os.listdir(args_training_dataset['json_folder']['LSDIR']))
    json_list.append(os.listdir(args_training_dataset['json_folder']['Entity_Seg']))
    args.save_dir = os.path.join(args.save_dir, 'Normal')
elif args.gt_type == 'Bokeh':
    json_list = os.listdir(args_training_dataset['json_folder']['EBB']) 
    args.save_dir = os.path.join(args.save_dir, 'Bokeh')
json_list.sort()  # Ensure the order is consistent

# Assign data to the current instance
total_files = len(json_list)
files_per_instance = total_files // args.num_instances
start_index = args.instance_id * files_per_instance
end_index = start_index + files_per_instance if args.instance_id < args.num_instances - 1 else total_files

# Assign the JSON file list to the current instance
args_training_dataset['json_list'] = json_list[start_index:end_index]

#################### REALESRGAN SETTING ###########################
args_training_dataset['queue_size'] = 160
args_training_dataset['crop_size'] =  512
args_training_dataset['io_backend'] = {}
args_training_dataset['io_backend']['type'] = 'disk'

args_training_dataset['blur_kernel_size'] = 21
args_training_dataset['kernel_list'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
args_training_dataset['kernel_prob'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
args_training_dataset['sinc_prob'] = 0.1
args_training_dataset['blur_sigma'] = [0.2, 1.5]
args_training_dataset['betag_range'] = [0.5, 2.0]
args_training_dataset['betap_range'] = [1, 1.5]

args_training_dataset['blur_kernel_size2'] = 11
args_training_dataset['kernel_list2'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
args_training_dataset['kernel_prob2'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
args_training_dataset['sinc_prob2'] = 0.1
args_training_dataset['blur_sigma2'] = [0.2, 1]
args_training_dataset['betag_range2'] = [0.5, 2.0]
args_training_dataset['betap_range2'] = [1, 1.5]

args_training_dataset['final_sinc_prob'] = 0.8

args_training_dataset['use_hflip'] = True
args_training_dataset['use_rot'] = False

train_dataset = BIRDataset_TwoDegradation(args_training_dataset)
batch_size = args.batch_size
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=11,
    drop_last=True,
    collate_fn=custom_collate_fn
)
print('length of training data', len(train_dataloader))

#################### REALESRGAN SETTING ###########################
args_degradation = {}
# the first degradation process
args_degradation['resize_prob'] = [0.2, 0.7, 0.1]  # up, down, keep
args_degradation['resize_range'] = [0.3, 1.5]
args_degradation['gaussian_noise_prob'] = 0.5
args_degradation['noise_range'] = [1, 15]
args_degradation['poisson_scale_range'] = [0.05, 2.0]
args_degradation['gray_noise_prob'] = 0.4
args_degradation['jpeg_range'] = [60, 95]

# the second degradation process
args_degradation['second_blur_prob'] = 0.5
args_degradation['resize_prob2'] = [0.3, 0.4, 0.3]  # up, down, keep
args_degradation['resize_range2'] = [0.6, 1.2]
args_degradation['gaussian_noise_prob2'] = 0.5
args_degradation['noise_range2'] = [1, 12]
args_degradation['poisson_scale_range2'] = [0.05, 1.0]
args_degradation['gray_noise_prob2'] = 0.4
args_degradation['jpeg_range2'] = [60, 100]

args_degradation['gt_size']= 512
args_degradation['no_degradation_prob']= 0.01


#################### Define the save path ###########################
data_root_path = args.save_dir 
gt_save_folder = os.path.join(data_root_path, 'gt')
mask_save_folder = os.path.join(data_root_path, 'mask')

unideg_lr_path = os.path.join(data_root_path, 'unideg1','lr')
unideg_sr_bicubic_path = os.path.join(data_root_path, 'unideg1','sr_bicubic')
 
unideg2_lr_path = os.path.join(data_root_path, 'unideg2','lr')
unideg2_sr_bicubic_path = os.path.join(data_root_path, 'unideg2','sr_bicubic')


# create the folder mentioned in before
os.makedirs(gt_save_folder, exist_ok=True)
os.makedirs(mask_save_folder, exist_ok=True)
os.makedirs(unideg_lr_path, exist_ok=True)
os.makedirs(unideg_sr_bicubic_path, exist_ok=True)
os.makedirs(unideg2_lr_path, exist_ok=True)
os.makedirs(unideg2_sr_bicubic_path, exist_ok=True)


# Define the path to save the JSON file
json_file_path_lr1 = os.path.join(data_root_path, 'unideg1', f'items_{args.instance_id}.json')
json_file_path_lr2 = os.path.join(data_root_path, 'unideg2', f'items_{args.instance_id}.json')


items_lr1 = []
items_lr2 = []
interval = 2000

epochs = args.epoch
step = 0
print('step', step)
start_time = time.time()

from tqdm import tqdm

with torch.no_grad():
    total_batches = epochs * len(train_dataloader)
    total_images = epochs * len(train_dataloader) * batch_size

    step = 0
    with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
        for epoch in range(epochs):
            batch_iter = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for num_batch, batch in batch_iter:
                #print('batch keys', batch.keys())
                
                lr_batch_list, gt_batch, lq_mask = realesrgan_degradation(batch, args_degradation=args_degradation)
                lr_batch1 = lr_batch_list[0]
                lr_batch2 = lr_batch_list[1]
                mask_batch = batch['mask']
                json_item_batch = batch['json_item']

                sr_bicubic_batch_unideg1 = F.interpolate(lr_batch1, size=(gt_batch.size(-2), gt_batch.size(-1)), mode='bicubic',)
                sr_bicubic_batch_unideg2 = F.interpolate(lr_batch2, size=(gt_batch.size(-2), gt_batch.size(-1)), mode='bicubic',)
                

                for i in range(batch_size):
                    step += 1
                    # print('process {} images...'.format(step))   # 用进度条替代此输出

                    gt = gt_batch[i, ...]
                    lr1 = lr_batch1[i, ...]
                    lr2 = lr_batch2[i, ...]
                    mask = mask_batch[i, ...]
                    json_item = json_item_batch[i]

                    sr_bicubic_unideg1 = sr_bicubic_batch_unideg1[i, ...]
                    sr_bicubic_unideg2 = sr_bicubic_batch_unideg2[i, ...]
                    # print('score batch1', score_batch1)
                    
                    # generate save name, all the images sharing the same gt would have the same name but in different folders
                    mask_img_name = json_item['mask_image_name']
                    content = os.path.splitext(mask_img_name)[0]
                    save_name = f'{content}_{epoch}.png'

                    # save the mask and gt
                    mask_img_path = os.path.join(mask_save_folder, save_name)           
                    gt_path = os.path.join(gt_save_folder, save_name) 
                    cv2.imwrite(mask_img_path, 255*mask.detach().cpu().squeeze().permute(1,2,0).numpy()[..., ::-1])
                    cv2.imwrite(gt_path, 255*gt.detach().cpu().squeeze().permute(1,2,0).numpy()[..., ::-1])

                    item_lr1 = postsave_batchitem(lr1, sr_bicubic_unideg1, unideg_lr_path, unideg_sr_bicubic_path, save_name, gt_path, mask_img_path, json_item)
                    item_lr2 = postsave_batchitem(lr2, sr_bicubic_unideg2, unideg2_lr_path, unideg2_sr_bicubic_path, save_name, gt_path, mask_img_path, json_item)
                    
                    items_lr1.append(item_lr1)
                    items_lr2.append(item_lr2)

                    if step % interval == 0:
                        append_to_json_file(json_file_path_lr1, items_lr1)
                        append_to_json_file(json_file_path_lr2, items_lr2)
                        items_lr1 = []
                        items_lr2 = []

                    pbar.update(1)  # 一张图片进度

                # del lr_batch1, lr_batch2, gt_batch, sr_bicubic_batch
                torch.cuda.empty_cache()
        
        # Process the remaining item (if any)
        if items_lr1:
            append_to_json_file(json_file_path_lr1, items_lr1)
        if items_lr2:
            append_to_json_file(json_file_path_lr2, items_lr2)


end_time = time.time()
print(f'For {step} steps, Processing time: {end_time-start_time} seconds')

