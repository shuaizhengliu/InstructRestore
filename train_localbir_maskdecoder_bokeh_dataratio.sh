
accelerate config default

export MODEL_DIR="path of stable-diffusion-2-1-base"
export OUTPUT_DIR="./Experiment/normal_data"

mkdir -p "$OUTPUT_DIR"
LOG_SH_FILE="$OUTPUT_DIR/sh.txt"
cat "$0" > "$LOG_SH_FILE"
echo "SH file content saved to $LOG_SH_FILE"

accelerate launch train_controlnetsd21_localbir_maskdecoder_bokeh_dataratio.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --controlnet_model_name_or_path='./Experiment/normal_data/checkpoint-120000/controlnet'\
 --train_dataset_module='LocalBIRDataset'\
 --train_dataset_type='BIR_UniDeg2Clean_mainprompt_controlprompt_mask_addbokeh_ratioargs'\
 --test_dataset_module='LocalBIRDataset'\
 --test_dataset_type='Val_image'\
 --train_datafolder='/home/notebook/data/personal/S9048593/LocalRestoration_Data'\
 --train_bokeh_datafolder='Training_data/bokeh path'\
 --deg_type_list='unideg'\
 --test_datafolder='/home/notebook/data/personal/S9048593/LocalRestoration_Val_Data'\
 --val_json='/home/notebook/data/personal/S9048593/LocalRestoration_Val_Data/val_globalclear_prompt2.json'\
 --resolution=512 \
 --learning_rate=5e-5\ 
 --train_batch_size=16\
 --proportion_main_empty_prompts=0.05\
 --proportion_control_empty_prompts=0.05\
 --gradient_accumulation_steps=1\
 --num_train_epochs=2000\
 --validation_steps=500\
 --checkpointing_steps=500\
 --dataloader_num_workers=8\
 --mask_decoder_type='MaskDecoder'\
 --bir_loss_weight=1\
 --mask_loss_weight=0.01\
 --bokeh_ratio=0.75\