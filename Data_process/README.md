# InstructRestore Data Processing Guide

This folder contains scripts and tools for processing InstructRestore training data. The data processing consists of two main steps: original data download and paired training data generation.

## Step 1: Original Data Download

### Data Source
Download the original data from the following link:

 [Tri-IR Dataset (Google Drive)](https://drive.google.com/drive/folders/1ilrbsjvTTg-c7L6gj4eqXbHCanFOTtaz?usp=sharing)

unzip the data to your path 
```
'Original_data_folder path'
```


### Original Data Structure
The downloaded original data should contain the following structure:

```
Original_data_folder/
├── Normal/                          # Normal image data
│   ├── LSDIR/                           # LSDIR dataset
│   │   ├── GT/                              # High-resolution GT images
│   │   ├── masks/                           # Mask images
│   │   └── annotated_jsons/                 # Annotation JSON files
│   └── EntitySeg_Sem/                   # Filtered dataset from Entity_Seg
│       ├── GT/                              # High-resolution GT images
│       ├── masks/                           # Mask images
│       └── annotated_jsons/                 # Annotation JSON files
└── Bokeh/                           # Bokeh blur image data
    └── EBB!/                            # EBB! training dataset
        ├── GT/                              # High-resolution GT images
        ├── masks/                           # Mask images
        └── annotated_jsons/                 # Annotation JSON files
```

### Data Content Description
- **GT folder**: Contains high-quality high-resolution images
- **masks folder**: Contains corresponding mask images to identify regions
- **annotated_jsons folder**: Contains JSON format annotation files that record text descriptions for each mask region

## Step 2: Generate Paired Training Data

### Overview
Use Real-ESRGAN degradation methods to generate paired training data with degraded inputs. This step generates multiple degraded versions of low-quality images for each original image, used for training image restoration models.

### Execution
Processing the normal (Normal) GT images *first*, followed by the Bokeh GT images.

### 1. Normal Data Processing  
Modify the 'make_paired_data_normal.sh' as follows:

```bash
pip install basicsr
python make_paired_data.py \
--instance_id 0 \
--num_instances 1 \
--original_data_folder 'Original_data_folder path' \
--gt_type 'Normal' \
--save_dir 'Training_data/normal path' \
--num_instances 1 \
--instance_id 0
```

- Replace `'Original_data_folder path'` with the path where you put your original data (e.g., `/path/to/your/original/data`).
- Replace `'Training_data/normal path'` with the directory where the generated training data should be saved (e.g., `/path/to/save/training_data/normal`).

After modifying these paths, you can run the processing with:
```bash
bash make_paired_data_normal.sh
```

> **Note**: Running the above script will process all data in sequence and may be slow. You can split the workload to achieve batch processing as follows.

#### Batch Processing & Multi-Machine Usage

- You can split the dataset into multiple batches by adjusting the parameters `--num_instances` (total number of splits) and `--instance_id` (current split index, starting from 0).
- For example, to divide the data into 4 batches and process them in parallel (e.g., on 4 different machines), create separate `.sh` scripts for each batch and only change the `instance_id`:

```bash
# make_paired_data_normal_0.sh
python make_paired_data.py --instance_id 0 --num_instances 4 --original_data_folder 'Original_data_folder path' --gt_type 'Normal' --save_dir 'Training_data/normal path'

# make_paired_data_normal_1.sh
python make_paired_data.py --instance_id 1 --num_instances 4 --original_data_folder 'Original_data_folder path' --gt_type 'Normal' --save_dir 'Training_data/normal path'

# ...and so on
```
- You can launch each script on a different node, allowing the batches to be processed in parallel.

---

### 2. Bokeh Data Processing  
After completing Normal data processing, process the Bokeh data using a similar approach. The script is almost identical; you only need to adjust the relevant parameters (change `--gt_type` and `--save_dir`):

```bash
pip install basicsr
python make_paired_data.py \
--instance_id 0 \
--num_instances 1 \
--original_data_folder 'Original_data_folder path' \
--gt_type 'Bokeh' \
--save_dir 'Training_data/bokeh path' \
--num_instances 1 \
--instance_id 0
```
- As before, set the correct paths for `original_data_folder` and `save_dir`.

For batch/multi-machine processing, simply change the `instance_id` and `num_instances` as above, prepare additional scripts, and run them on different machines if desired.

---

### Output Data Structure

After processing is complete, the training data will be saved in the specified directory with the following structure:

```
Training_data/
├── Normal/
│   ├── gt/                          # Original high-resolution images
│   ├── mask/                        # Mask images
│   ├── unideg1/                     # First degradation type
│   │   ├── lr/                      # Low-resolution images
│   │   ├── sr_bicubic/              # Bicubic upsampled images
│   │   └── items_*.json             # Data index files
│   └── unideg2/                     # Second degradation type
│       ├── lr/                      # Low-resolution images
│       ├── sr_bicubic/              # Bicubic upsampled images
│       └── items_*.json             # Data index files
└── Bokeh/
    ├── gt/                          # Original high-resolution images
    ├── mask/                        # Mask images
    ├── unideg1/                     # First degradation type
    │   ├── lr/                      # Low-resolution images
    │   ├── sr_bicubic/              # Bicubic upsampled images
    │   └── items_*.json             # Data index files
    └── unideg2/                     # Second degradation type
        ├── lr/                      # Low-resolution images
        ├── sr_bicubic/              # Bicubic upsampled images
        └── items_*.json             # Data index files
```

### Main Parameter Description

- `--original_data_folder`: Original data folder path
- `--gt_type`: Data type ('Normal' or 'Bokeh')
- `--save_dir`: Training data save directory
- `--batch_size`: Batch size (recommended to use 1 for richer degradation effects)
- `--epoch`: Number of generation epochs (default 3 epochs)
- `--num_instances`: Number of batches
- `--instance_id`: Current batch ID


