### Work in Progress

The training scripts for pix2pix and cycle gan are working using CUDA

# Example usage:

## Pix2Pix
accelerate launch src/train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --is_sdxl=True \
    --output_dir="output/pix2pix_turbo/fill50k" \
    --dataset_folder="data/my_fill50k" \
    --resolution=512 \
    --train_batch_size=2 \
    --enable_xformers_memory_efficient_attention --viz_freq 50 \
    --track_val_fid \
    --report_to "wandb" --tracker_project_name "pix2pix_turbo_fill50k"


## Pix2Pix
accelerate launch src/train_cyclegan_turbo.py \
    --output_dir="output/cyclegan/horsezebra" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"\
    --is_sdxl=True \
    --dataset_folder="data/my_horse2zebra" \
    --train_img_prep="resize_512" \
    --val_img_prep="resize_512" \
    --train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --viz_freq 50 \
    --max_train_steps=10000 \
    --report_to "wandb" --tracker_project_name "cycleGan_turbo_horse2zebra"
