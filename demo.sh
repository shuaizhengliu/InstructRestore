python demo.py \
    --cfg_scale 3.6 \
    --mask_init_step 0 \
    --align_method 'adain' \
    --start_point 'lr' \
    --step 20 \
    --save_folder 'path/to/save_folder' \
    --controlnet_path 'path/to/instruct_restore_sd21base.pth'\
    --base_model_path 'path/to/sd2.1_base_model'\
    --image_path 'path/to/image.jpg'\
    --instruction 'make [target] clear with [mask_inner_scale value] and keep other parts with [mask_outer_scale value]' \
    # Instruction template example:
    # normal situation:
    # 'make [target] clear with [mask_inner_scale value] and keep other parts with [mask_outer_scale value]'
    # Example: make the person clear with 0.8 and keep other parts with 1.0
    # bokeh situation:
    # 'make [target] clear with [mask_inner_scale value] and keep other parts bokeh blur with [mask_outer_scale value]'
    # : make the person clear with 1 and keep other parts bokeh blur with 0.7










