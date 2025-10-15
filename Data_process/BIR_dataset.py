import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data
from torch.utils.data import DataLoader

import glob
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

import json

def resize_image_and_mask(img_gt, mask, max_size=6000):
    # 获取图像的高度和宽度
    h, w = img_gt.shape[:2]
    
    # 找到最长边
    max_dim = max(h, w)
    
    # 如果最长边超过max_size，则进行缩放
    if max_dim > max_size:
        # 计算缩放比例
        scale = max_size / max_dim
        
        # 计算新的尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 使用保边最好的方法缩放图像
        img_gt_resized = cv2.resize(img_gt, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 使用最近邻插值方法缩放掩码
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        return img_gt_resized, mask_resized
    
    # 如果最长边不超过max_size，则返回原图像和掩码
    return img_gt, mask


def resize_two_image_and_mask(img_gt, sudo_gt, mask, max_size=6000):
    # 获取图像的高度和宽度
    h, w = img_gt.shape[:2]
    
    # 找到最长边
    max_dim = max(h, w)
    
    # 如果最长边超过max_size，则进行缩放
    if max_dim > max_size:
        # 计算缩放比例
        scale = max_size / max_dim
        
        # 计算新的尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 使用保边最好的方法缩放图像
        img_gt_resized = cv2.resize(img_gt, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        sudo_gt_resized = cv2.resize(sudo_gt, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 使用最近邻插值方法缩放掩码
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        return img_gt_resized, sudo_gt_resized, mask_resized
    
    # 如果最长边不超过max_size，则返回原图像和掩码
    return img_gt, sudo_gt, mask

@DATASET_REGISTRY.register(suffix='basicsr')
class BIRDataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(BIRDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512
        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        # Read the list of json files to get each item
        self.items = []
        for file in opt['json_list']:
            cur_list = json.load(open(os.path.join(opt['json_folder'],file)))
            self.items = self.items + cur_list
        

        # limit number of pictures for test
        if 'num_pic' in opt:
            if 'val' or 'test' in opt:
                random.shuffle(self.items)
                self.items = self.items[:opt['num_pic']]
            else:
                self.items = self.items[:opt['num_pic']]

        if 'mul_num' in opt:
            self.items = self.items * opt['mul_num']


        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        cur_item = self.items[index]
        print('cur_item', cur_item)
        cur_source = cur_item['img_source']
        #gt_path = os.path.join(self.opt['source_image_folder'][cur_source], cur_item['content_image_name'])
        if cur_source == 'LSDIR':
            content_image_name = cur_item['content_image_name']
            img_number = int(os.path.splitext(content_image_name)[0])
            sub_folder = str((img_number //1000+1)*1000).zfill(7)
            gt_path = os.path.join(self.opt['source_image_folder'][cur_source], sub_folder, content_image_name)
            
        else:
            gt_path = os.path.join(self.opt['source_image_folder'][cur_source], cur_item['content_image_name'])
        mask_path = os.path.join(self.opt['source_mask_folder'][cur_source], cur_item['mask_image_name'])
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                print('gt_path', gt_path)
                img_bytes = self.file_client.get(gt_path, 'gt')
                mask_bytes = self.file_client.get(mask_path, 'gt')
            except (IOError, OSError) as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                cur_item = self.items[index]
                cur_source = cur_item['img_source']
                gt_path = os.path.join(self.opt['source_image_folder'][cur_source], cur_item['content_image_name'])
                mask_path = os.path.join(self.opt['source_mask_folder'][cur_source], cur_item['mask_image_name'])
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)
        mask = imfrombytes(mask_bytes, float32=True) 

        '''
        # filter the dataset and remove images with too low quality
        img_size = os.path.getsize(gt_path)
        img_size = img_size/1024

        while img_gt.shape[0] * img_gt.shape[1] < 384*384 or img_size<100:
            index = random.randint(0, self.__len__()-1)
            gt_path = self.paths[index]

            time.sleep(0.1)  # sleep 1s for occasional server congestion
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_size = os.path.getsize(gt_path)
            img_size = img_size/1024
        '''

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt, mask = augment([img_gt, mask], self.opt['use_hflip'], self.opt['use_rot'])

        img_gt, mask = resize_image_and_mask(img_gt, mask) # if the max size of the image is larger than 6000, resize it

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.crop_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        # crop under the condition of forcing the crop center inside mask
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            img_gt, mask = self.dual_forcemask_crop(img_gt, mask, h, w, crop_pad_size)

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, sudo_gt, mask = img2tensor([img_gt, sudo_gt, mask], bgr2rgb=True, float32=True)
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'sudo_gt':sudo_gt, 'mask': mask, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'json_item': cur_item}
        return return_d

    def __len__(self):
        return len(self.items)


    def dual_forcemask_crop(self, img_gt, mask, h, w, patch_size):

        y_s = random.randint(0, h - patch_size)
        x_s = random.randint(0, w - patch_size)

        mask_pixs = np.stack(np.where(mask[patch_size // 2:-patch_size // 2, patch_size // 2:-patch_size // 2, 0]), 1)
        if len(mask_pixs) > 0:
            y_s, x_s = random.choice(mask_pixs)

        img_gt = img_gt[y_s:y_s + patch_size, x_s:x_s + patch_size, ...]
        mask = mask[y_s:y_s + patch_size, x_s:x_s + patch_size, ...]

        return img_gt, mask


class BIRDataset_TwoDegradation(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            source_image_folder (str): Data root path for gt.
            source_mask_folder (str): Data root path for mask.
            json_folder (str): Data root path for json.
            json_list (list): List of json files.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(BIRDataset_TwoDegradation, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512
        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        # Read the list of json files to get each item
        self.items = []
        for file in opt['json_list']:
            if file.lower().startswith('lsdir'):
                json_folder = opt['json_folder']['LSDIR']
            elif file.lower().startswith('entityseg'):
                json_folder = opt['json_folder']['Entity_Seg']
            elif file.lower().startswith('ebb'):
                json_folder = opt['json_folder']['EBB']

            cur_list = json.load(open(os.path.join(json_folder, file)))
            self.items = self.items + cur_list
        

        # limit number of pictures for test
        if 'num_pic' in opt:
            if 'val' or 'test' in opt:
                random.shuffle(self.items)
                self.items = self.items[:opt['num_pic']]
            else:
                self.items = self.items[:opt['num_pic']]

        if 'mul_num' in opt:
            self.items = self.items * opt['mul_num']


        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        cur_item = self.items[index]
        #print('cur_item', cur_item)
        cur_source = cur_item['img_source']
        gt_path = os.path.join(self.opt['source_image_folder'][cur_source], cur_item['content_image_name'])
        #print('gt_path', gt_path)
        mask_path = os.path.join(self.opt['source_mask_folder'][cur_source], cur_item['mask_image_name'])
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                print('gt_path', gt_path)
                img_bytes = self.file_client.get(gt_path, 'gt')
                mask_bytes = self.file_client.get(mask_path, 'gt')
            except (IOError, OSError) as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                cur_item = self.items[index]
                cur_source = cur_item['img_source']
                if cur_source == 'LSDIR':
                    content_image_name = cur_item['content_image_name']
                    img_number = int(os.path.splitext(content_image_name)[0])
                    sub_folder = str((img_number //1000+1)*1000).zfill(7)
                    gt_path = os.path.join(self.opt['source_image_folder'][cur_source], sub_folder, content_image_name)
                    
                else:
                    gt_path = os.path.join(self.opt['source_image_folder'][cur_source], cur_item['content_image_name'])
                mask_path = os.path.join(self.opt['source_mask_folder'][cur_source], cur_item['mask_image_name'])
                img_bytes = self.file_client.get(gt_path, 'gt')
                mask_bytes = self.file_client.get(mask_path, 'gt')
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)
        mask = imfrombytes(mask_bytes, float32=True) 


        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt, mask = augment([img_gt, mask], self.opt['use_hflip'], self.opt['use_rot'])

        img_gt, mask = resize_image_and_mask(img_gt, mask) # if the max size of the image is larger than 6000, resize it

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.crop_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        # crop under the condition of forcing the crop center inside mask
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            img_gt, mask = self.dual_forcemask_crop(img_gt, mask, h, w, crop_pad_size)
        
        kernel1_list = []
        kernel2_list = []
        sinc_kernel_list = []

        for i in range(2):
            # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.opt['sinc_prob']:
                # this sinc filter setting is for kernels ranging from [7, 21]
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)
            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

            # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.opt['sinc_prob2']:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel2 = random_mixed_kernels(
                    self.kernel_list2,
                    self.kernel_prob2,
                    kernel_size,
                    self.blur_sigma2,
                    self.blur_sigma2, [-math.pi, math.pi],
                    self.betag_range2,
                    self.betap_range2,
                    noise_range=None)

            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

            # ------------------------------------- the final sinc kernel ------------------------------------- #
            if np.random.uniform() < self.opt['final_sinc_prob']:
                kernel_size = random.choice(self.kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = self.pulse_tensor

            kernel = torch.FloatTensor(kernel)
            kernel2 = torch.FloatTensor(kernel2)

            kernel1_list.append(kernel)
            kernel2_list.append(kernel2)
            sinc_kernel_list.append(sinc_kernel)
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, mask = img2tensor([img_gt, mask], bgr2rgb=True, float32=True)

        return_d = {'gt': img_gt, 'mask': mask, 'kernel1_0': kernel1_list[0], 'kernel1_1': kernel1_list[1], 'kernel2_0': kernel2_list[0], 'kernel2_1': kernel2_list[1], 'sinc_kernel_0': sinc_kernel_list[0], 'sinc_kernel_1':sinc_kernel_list[1], 'json_item': cur_item}
        return return_d

    def __len__(self):
        return len(self.items)


    def dual_forcemask_crop(self, img_gt, mask, h, w, patch_size):

        y_s = random.randint(0, h - patch_size)
        x_s = random.randint(0, w - patch_size)

        mask_pixs = np.stack(np.where(mask[patch_size // 2:-patch_size // 2, patch_size // 2:-patch_size // 2, 0]), 1)
        if len(mask_pixs) > 0:
            y_s, x_s = random.choice(mask_pixs)

        img_gt = img_gt[y_s:y_s + patch_size, x_s:x_s + patch_size, ...]
        mask = mask[y_s:y_s + patch_size, x_s:x_s + patch_size, ...]

        return img_gt, mask


class BIRDataset_TwoDegradation_nocrop(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """
    
    def __init__(self, opt):
        super(BIRDataset_TwoDegradation_nocrop, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512
        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        # Read the gt path
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
        self.items = []
        for folder in opt['gt_folders']:
            for ext in image_extensions:
                self.items.extend(glob.glob(os.path.join(folder, ext)))
        
        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.items[index]
        image_name = os.path.basename(gt_path)
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        
        kernel1_list = []
        kernel2_list = []
        sinc_kernel_list = []

        for i in range(2):
            # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.opt['sinc_prob']:
                # this sinc filter setting is for kernels ranging from [7, 21]
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)
            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

            # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.opt['sinc_prob2']:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel2 = random_mixed_kernels(
                    self.kernel_list2,
                    self.kernel_prob2,
                    kernel_size,
                    self.blur_sigma2,
                    self.blur_sigma2, [-math.pi, math.pi],
                    self.betag_range2,
                    self.betap_range2,
                    noise_range=None)

            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

            # ------------------------------------- the final sinc kernel ------------------------------------- #
            if np.random.uniform() < self.opt['final_sinc_prob']:
                kernel_size = random.choice(self.kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = self.pulse_tensor

            kernel = torch.FloatTensor(kernel)
            kernel2 = torch.FloatTensor(kernel2)

            kernel1_list.append(kernel)
            kernel2_list.append(kernel2)
            sinc_kernel_list.append(sinc_kernel)
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]

        return_d = {'gt': img_gt, 'kernel1_0': kernel1_list[0], 'kernel1_1': kernel1_list[1], 'kernel2_0': kernel2_list[0], 'kernel2_1': kernel2_list[1], 'sinc_kernel_0': sinc_kernel_list[0], 'sinc_kernel_1':sinc_kernel_list[1], 'image_name':image_name}
        return return_d

    def __len__(self):
        return len(self.items)


def custom_collate_fn(batch):
    batch_dict = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            #print('tensor key', key)
            batch_dict[key] = torch.stack([d[key] for d in batch])
        else:
            #print('nontensor key', key)
            #print('type of value', type(batch[0][key]))
            batch_dict[key] = [d[key] for d in batch]
    return batch_dict



if __name__ == '__main__':
    
    
    args_training_dataset = {}

    # Please set your gt path here. If you have multi dirs, you can set it as ['PATH1', 'PATH2', 'PATH3', ...]
    args_training_dataset['json_list'] = os.listdir('/home/notebook/data/personal/S9048593/reloblur_data')
    args_training_dataset['source_image_folder']= {'LSDIR':'/home/notebook/data/group/LSDIR/HR'}
    args_training_dataset['source_mask_folder']= {'LSDIR':'/home/notebook/data/personal/S9048593/LSDIR_Mask_Level2_merge_same_subject'}
    args_training_dataset['json_folder'] = '/home/notebook/data/personal/S9048593/reloblur_data'

    #################### REALESRGAN SETTING ###########################
    args_training_dataset['queue_size'] = 160
    args_training_dataset['crop_size'] =  512
    args_training_dataset['io_backend'] = {}
    args_training_dataset['io_backend']['type'] = 'disk'

    args_training_dataset['blur_kernel_size'] = 21
    args_training_dataset['kernel_list'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    args_training_dataset['kernel_prob'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    args_training_dataset['sinc_prob'] = 0.1
    args_training_dataset['blur_sigma'] = [0.2, 3]
    args_training_dataset['betag_range'] = [0.5, 4]
    args_training_dataset['betap_range'] = [1, 2]

    args_training_dataset['blur_kernel_size2'] = 11
    args_training_dataset['kernel_list2'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    args_training_dataset['kernel_prob2'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    args_training_dataset['sinc_prob2'] = 0.1
    args_training_dataset['blur_sigma2'] = [0.2, 1.5]
    args_training_dataset['betag_range2'] = [0.5, 4.0]
    args_training_dataset['betap_range2'] = [1, 2]

    args_training_dataset['final_sinc_prob'] = 0.8

    args_training_dataset['use_hflip'] = True
    args_training_dataset['use_rot'] = False

    train_dataset = BIRDataset_TwoDegradation(args_training_dataset)
    data1 = train_dataset[0]
    print('num of dataset', len(train_dataset))
    print('gt shape', data1['gt'].shape)
    print('mask shape', data1['mask'].shape)
    print('json item', data1['json_item'])
    print('kernel1 shape', data1['kernel1_0'].shape)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,collate_fn=custom_collate_fn)
        # 获取 DataLoader 的迭代器
    data_iter = iter(train_dataloader)
    
    # 获取第一个批次的数据
    data = next(data_iter)
    print('data type', type(data))
    print('data keys', data.keys())
    print('gt shape', data['gt'].shape)
    print('mask shape', data['mask'].shape)
    print('json item', data['json_item'])
    print('kernel1 shape', data['kernel1_0'].shape)
    print('kernel2 shape', data['kernel2_0'].shape)
    print('sinc_kernel shape', data['sinc_kernel_0'].shape)
    print('kernel1_1 shape', data['kernel1_1'].shape)
    print('kernel2_1 shape', data['kernel2_1'].shape)
    print('sinc_kernel_1 shape', data['sinc_kernel_1'].shape)



