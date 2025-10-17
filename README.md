
<div align="center">
<h2>InstructRestore: Region-Customized Image Restoration with Human Instructions</h2>

<a href='http://arxiv.org/abs/2503.24357'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>


Shuaizheng Liu<sup>1,2</sup>
| Jianqi Ma<sup>1</sup> | 
Lingchen Sun<sup>1,2</sup> | 
Xiangtao Kong<sup>1,2</sup> | 
Lei Zhang<sup>1,2</sup>

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute
</div>

## ⏰ News
- **2025.9.19** : The paper has been accepted in NeurIPS 2025 ! 

##  💡  Overview

![InstructRestore](figs/teasers1.png)

Our proposed **InstructionRestore** framework enables region-customized restoration following human instruction. 

(a) current methods tend to incorrectly restore the bokeh blur, while our method allows for adjustable control over the degree of blur based on user instructions. 

(b) existing methods fail to achieve region-specific enhancement intensities, while our approach can simultaneously suppress the over-enhancement in areas of building and improve the visual quality in areas of leaves.


##  🎨 Application
### Demo on Real-world Localized Enhancement
<img src="figs/localized_enhancement.png" alt="InstructRestore" width="600">

By following the instruction, the details in flowers are enhanced gradually while the other regions keeping almost unchanged.
### Demo on Controllable Bokeh Effects 
<img src="figs/controllable_bokeh.png" alt="InstructRestore" width="600">

By following the instruction, 

(a) Restoration with controlled bokeh effect while restoring foreground. 

(b) Restoration with varying foreground enhancement levels while preserving background bokeh.


### Comparisons with Other DM-Based global restoration Methods
(a) For the localized enhancement
![InstructRestore](figs/local_compare.png)

(b) For the preservation of bokeh effects
![InstructRestore](figs/bokeh_compare.png)

##  🍭 Achitecture
![InstructRestore](figs/architecture.png)

## 🌱  Dataset Construction Pipeline
![InstructRestore](figs/Dataset_construction.png)

## ⚙ Dependencies and Installation
```shell
## git clone this repository
git clone https://github.com/shuaizhengliu/InstructRestore.git
cd InstructRestore


# create an environment
conda create -n InstructRestore python=3.10
conda activate InstructRestore
pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 Demo & Quick inferences 
1. First, download the [InstructRestore model](https://drive.google.com/drive/folders/1oAGr-k5Rvg7hRh6sDa1gU0kFl5jbSbim?usp=sharing) and prepare the [Stable Diffusion v2.1 base model](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).

2. In `demo.sh`, update the following content by setting the correct paths for `--controlnet_path` and `--base_model_path`, as well as your test image path (`--image_path`), desired instruction (`--instruction`), and the folder path where results will be saved (`--save_folder`).
   Here is the content of `demo.sh`:

```bash
python demo.py \
    --cfg_scale 3.6 \
    --mask_init_step 0 \
    --align_method 'adain' \
    --start_point 'lr' \
    --step 20 \
    --save_folder 'path/to/save_folder' \
    --controlnet_path 'path/to/InstructRestore_sd21.pth'\
    --base_model_path 'path/to/sd2.1_base_model'\
    --image_path 'path/to/image.jpg'\
    --instruction 'make [target] clear with [mask_inner_scale value] and keep other parts clear with [mask_outer_scale value]' \
    # Instruction template example:
    # normal situation:
    # 'make [target] clear with [mask_inner_scale value] and keep other parts clear with  [mask_outer_scale value]'
    # Example: make the person clear with 0.8 and keep other parts clear with 1.0
    # bokeh situation:
    # 'make [target] clear with [mask_inner_scale value] and keep other parts bokeh blur with [mask_outer_scale value]'
    # : make the person clear with 1 and keep other parts bokeh blur with 0.7
```

3. After editing the above paths and parameters, simply run:

```bash
bash demo.sh
```

to quickly experience region-customized restoration with instructions!



## 🗂️ Tri-IR Dataset

High-quality triplet datasets including GT, semantic mask annotations, and region-level captions are extremely scarce, yet crucial for controllable local image restoration. To fill this gap, we introduce Tri-IR, a large-scale, high-quality dataset consisting of 560k GT-mask-local caption triplets. We provide public access via Google Drive: [Tri-IR Dataset (Google Drive)](https://drive.google.com/drive/folders/1ilrbsjvTTg-c7L6gj4eqXbHCanFOTtaz?usp=sharing). The entire dataset is approximately 161GB in size.

## 🏋️ Training
#### Step1: Prepare training data
 Please refer to the instructions in [Data_process/README.md](https://github.com/shuaizhengliu/InstructRestore/blob/main/Data_process/README.md) to download and process the Tri-IR dataset. Use the processed paired data for training.

#### Step2: Train model
1. Download pretrained [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) to provide generative capabilities.

    ```shell
    wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate
    ```

2. Training based on data of normal GT.

    Please remember to update the variable paths in the `sh` script (such as the paths to your training data, save directory, and pretrained weights) to your own directories before running the training.

    ```shell
    bash train_localbir_maskdecoder.sh
    ```

3. Training by adding data of Bokeh GT.
   
   After the training on normal data has converged and the restoration performance stabilizes, you can further improve the model's ability to handle bokeh effects by including the Bokeh GT data for joint training. Please remember to update the paths in the corresponding `.sh` script below to your own directories before running the training.
    ```shell
    bash train_localbir_maskdecoder_bokeh_dataratio.sh
    ```

## 🧪 Evaluation


## Citation
If you find our paper helpful, please consider citing our papers and staring us! Thanks!
```
@article{liu2025instructrestore,
  title={InstructRestore: Region-Customized Image Restoration with Human Instructions},
  author={Liu, Shuaizheng and Ma, Jianqi and Sun, Lingchen and Kong, Xiangtao and Zhang, Lei},
  journal={arXiv preprint arXiv:2503.24357},
  year={2025}
}
```




