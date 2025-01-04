# Pix2Pix Turbo: Diffusion-Based Image-to-Image and CycleGAN-Style Training

Forked from GaParmar/img2img-turbo. The structure has been refactored so that the pix2pix as well as the cyclegan training and inference scripts are relying on the same pix2pix-turbo model. 
Features have been added to explore different model configurations and to support newer models including Flow-Matching ones and it is possible to run on apple MPS. 

This repository provides code to train diffusion-based image-to-image models using Stable Diffusion, Stable Diffusion XL (SDXL), or Stable Diffusion 3 (SD3) in a GAN setup. It includes:
- A **pix2pix**-style approach for paired datasets (`train_pix2pix_turbo.py`).
- A **cycleGAN**-style approach for unpaired datasets (`train_cyclegan_turbo.py`).

Both rely on the core model logic provided in `pix2pix_turbo.py` and several helper utilities in `training_utils.py`.  
Below is an overview of each file, their functions, and the command-line parameters that configure the training process.

---

## Project status
There are still left-overs from the original code that are not needed anymore. Some refactor still needs to be done. It is possible to train on MPS, but the GAN works different, which is why the cyclegan does not converge. The problem is the discriminator. If the discriminator runs on CPU and the rest on MPS the training metrics are very similar to cuda. I'm currently testing if I can make the discriminator run on MPS, otherwise I will publish a version with the option to put the discriminator on CPU. 

---

## Repository Structure

```
├── pix2pix_turbo.py           # Core model class, including prompt encoding logic for SD, SDXL, SD3
├── train_pix2pix_turbo.py     # Training script for pix2pix-style (paired) diffusion fine-tuning
├── train_cyclegan_turbo.py    # Training script for cycleGAN-style (unpaired) diffusion fine-tuning
├── my_utils/training_utils.py # Dataset classes, argument parsers, and other helper utilities
├── model.py                   # Contains model related functions like make_1step_sched
├── my_utils/dino_struct.py    # extracts multi-scale transformer features from a DINO ViT mode
├── inference_paired.py        # inference script for a pretrained model trained with paired data
├── inference_unpaired.py      # inference script for a pretrained model trained with unpaired data
└── ...
```

---

## 1. `pix2pix_turbo.py`

This file defines the main `Pix2Pix_Turbo` class, which wraps a diffusion model (Stable Diffusion, SDXL, or SD3) for image-to-image or cycleGAN training. Key points:

- **Supports**: 
  - Regular stable diffusion (v1.x or v2.x), 
  - Stable Diffusion XL (SDXL),
  - Stable Diffusion 3 (SD3).
- **LoRA** (Low-Rank Adaptation) integration for training certain layers in UNet and VAE.
- **Skip connections** injection into the VAE for flexible architecture (useful in certain advanced setups).
- **Prompt encoding** logic for SD, SDXL, and SD3.

### Main Components

1. **`Pix2Pix_Turbo` class**  
   The central class that:
   - Loads base model (UNet, VAE, tokenizers, text encoders).
   - Optionally adds LoRA adapters to the UNet and VAE.
   - Provides forward passes for a single step of inference/training.
   - Supports cyclegan-like forward pass (`forward_cycle`) for unpaired data.
   - Has helper methods for text prompt encoding in SD (`encode_prompt_sd`), SDXL (`encode_prompt_sdxl`), and SD3 (`encode_prompt_sd3`).

2. **`add_channels(unet)`**  
   A utility function that modifies the UNet to handle additional input channels (e.g., from 4 to 8 channels) for different tasks.

3. **`VAE_encode`, `VAE_decode`**  
   Simple wrapper modules for encoding/decoding with a VAE.  
   In cycleGAN mode, these can also switch to an "inverse" VAE for the reverse direction.

4. **`encode_prompts` and `compute_embeddings_for_prompts`**  
   Helper functions for SDXL prompt-encoding logic.  

5. **`save_model(self, outf)`**  
   Saves the state dictionary (including LoRA config) for UNet and VAE to a single file.

---

## 2. `train_pix2pix_turbo.py`

A script to train a diffusion-based image-to-image translation model (similar to pix2pix) on **paired datasets**. For example:
- Input domain A: `train_A`
- Output domain B: `train_B`
- The code will attempt to learn a direct mapping from domain A to domain B.

### Usage Example

```bash
python train_pix2pix_turbo.py \
    --dataset_folder /path/to/dataset \
    --train_image_prep resized_crop_512 \
    --test_image_prep resized_crop_512 \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --model_type sdxl \
    --num_inputs 2 \
    --output_dir /path/to/output \
    --max_train_steps 10000 \
    --train_batch_size 4 \
    --viz_freq 500 \
    --track_val_fid \
    --eval_freq 500 \
    --checkpointing_steps 2000 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --enable_xformers_memory_efficient_attention
```

### Command-Line Parameters

Below is a summary of **paired**-training arguments from `parse_args_paired_training()`:

| Argument                                 | Type    | Default | Description                                                                                                                                                                                                                                                                                                                                       |
|------------------------------------------|---------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **--dataset_folder**                     | str     | None    | **Required**. Path to your dataset folder (it should contain `train_A`, `train_B`, possibly `test_A`, `test_B`, and prompt JSONs).                                                                                                                                                                                                               |
| **--train_image_prep**                   | str     | `resized_crop_512` | Preprocessing pipeline for training images (see `build_transform`). e.g., `resize_256`, `resized_crop_512`, etc.                                                                                                                                                                                                                                |
| **--test_image_prep**                    | str     | `resized_crop_512` | Preprocessing pipeline for testing images.                                                                                                                                                                                                                                                                                                         |
| **--num_inputs**                         | int     | 2       | Number of latent inputs to the UNet (4 channels by default). If 3 inputs → 8 channels total.                                                                                                                                                                                                                                                      |
| **--gan_disc_type**                      | str     | `vagan_clip` | Type of discriminator in the GAN pipeline. Currently `vagan_clip` is implemented.                                                                                                                                                                                                                                                                 |
| **--cv_type**                            | str     | `clip`  | Type of computer vision backbone in the discriminator (e.g., CLIP).                                                                                                                                                                                                                                                                                |
| **--gan_loss_type**                      | str     | `multilevel_sigmoid_s` | Loss function for the GAN portion.                                                                                                                                                                                                                                                                                                                |
| **--lambda_gan**                         | float   | 0.5     | Weight for the GAN loss.                                                                                                                                                                                                                                                                                                                          |
| **--lambda_lpips**                       | float   | 5.0     | Weight for the LPIPS loss.                                                                                                                                                                                                                                                                                                                        |
| **--lambda_l2**                          | float   | 1.0     | Weight for the L2 (MSE) reconstruction loss.                                                                                                                                                                                                                                                                                                      |
| **--lambda_clipsim**                     | float   | 5.0     | Weight for the CLIP similarity loss.                                                                                                                                                                                                                                                                                                              |
| **--track_val_fid**                      | bool    | False   | If set, evaluates FID on the test set at intervals.                                                                                                                                                                                                                                                                                                |
| **--num_samples_eval**                   | int     | 100     | Number of samples from the test set for evaluation.                                                                                                                                                                                                                                                                                                |
| **--viz_freq**                           | int     | 100     | Frequency (in steps) to log images to W&B or other loggers.                                                                                                                                                                                                                                                                                       |
| **--eval_freq**                          | int     | 100     | Frequency (in steps) to run validation (computing metrics like FID, L2, LPIPS, etc.).                                                                                                                                                                                                                                                             |
| **--checkpointing_steps**               | int     | 500     | Frequency (in steps) to save model checkpoints.                                                                                                                                                                                                                                                                                                   |
| **--seed**                               | int     | None    | Random seed for reproducibility.                                                                                                                                                                                                                                                                                                                  |
| **--resolution**                         | int     | 512     | Resolution (used for certain transformations).                                                                                                                                                                                                                                                                                                    |
| **--pretrained_model_name_or_path**      | str     | `stabilityai/stable-diffusion-xl-base-1.0` | The base model to fine-tune. Accepts Hugging Face model hub ID or local path.                                                                                                                                                                                                                               |
| **--pretrained_path**                    | str     | None    | Path to a local `.pkl` or `.ckpt` with LoRA or custom model weights to initialize from.                                                                                                                                                                                                                                                           |
| **--out_model_name**                     | str     | `test_model` | Name for your final model. Not strictly used by the training script.                                                                                                                                                                                                                                                                              |
| **--model_type**                         | str     | `sd`    | Which diffusion variant: `sd` (Stable Diffusion), `sdxl`, or `sd3`.                                                                                                                                                                                                                                                                                |
| **--revision**                           | str     | None    | Hugging Face revision.                                                                                                                                                                                                                                                                                                                             |
| **--variant**                            | str     | None    | Hugging Face variant.                                                                                                                                                                                                                                                                                                                              |
| **--tokenizer_name**                     | str     | None    | If not provided, uses default from `pretrained_model_name_or_path`.                                                                                                                                                                                                                                                                                |
| **--lora_rank_unet**                     | int     | 8       | Rank for LoRA in the UNet.                                                                                                                                                                                                                                                                                                                        |
| **--lora_rank_vae**                      | int     | 4       | Rank for LoRA in the VAE.                                                                                                                                                                                                                                                                                                                         |
| **--no_skips**                           | bool    | False   | If set, disables skip connections in the VAE.                                                                                                                                                                                                                                                                                                     |
| **--freeze_vae**                         | bool    | False   | Whether to freeze the VAE weights (only UNet LoRA is trained).                                                                                                                                                                                                                                                                                    |
| **--freeze_unet**                        | bool    | False   | Whether to freeze the UNet weights (only VAE LoRA is trained).                                                                                                                                                                                                                                                                                    |
| **--original_unet**                      | bool    | False   | If set, do not add LoRA or custom changes to the UNet.                                                                                                                                                                                                                                                                                            |
| **--original_vae**                       | bool    | False   | If set, do not add LoRA or custom changes to the VAE.                                                                                                                                                                                                                                                                                             |
| **--output_dir**                         | str     | None    | **Required**. Where to save outputs and checkpoints.                                                                                                                                                                                                                                                                                              |
| **--cache_dir**                          | str     | None    | (Optional) Hugging Face cache directory.                                                                                                                                                                                                                                                                                                          |
| **--train_batch_size**                   | int     | 4       | Batch size for training.                                                                                                                                                                                                                                                                                                                           |
| **--num_training_epochs**                | int     | 10      | Number of training epochs (fallback if not using max_train_steps).                                                                                                                                                                                                                                                                                 |
| **--max_train_steps**                    | int     | 10000   | Number of total training steps.                                                                                                                                                                                                                                                                                                                   |
| **--gradient_accumulation_steps**        | int     | 1       | Number of steps to accumulate grads before update.                                                                                                                                                                                                                                                                                                |
| **--gradient_checkpointing**             | bool    | False   | Use gradient checkpointing to save memory.                                                                                                                                                                                                                                                                                                        |
| **--learning_rate**                      | float   | 5e-6    | Learning rate.                                                                                                                                                                                                                                                                                                                                    |
| **--lr_scheduler**                       | str     | `constant` | Learning rate scheduler type. Options: `[linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup]`.                                                                                                                                                                                                                      |
| **--lr_warmup_steps**                    | int     | 500     | Warmup steps for scheduler.                                                                                                                                                                                                                                                                                                                       |
| **--lr_num_cycles**                      | int     | 1       | Number of cycles for some schedulers.                                                                                                                                                                                                                                                                                                             |
| **--lr_power**                           | float   | 1.0     | Power factor for polynomial scheduler.                                                                                                                                                                                                                                                                                                            |
| **--dataloader_num_workers**             | int     | 0       | Number of subprocesses for data loading.                                                                                                                                                                                                                                                                                                          |
| **--adam_beta1**                         | float   | 0.9     | β1 for AdamW.                                                                                                                                                                                                                                                                                                                                     |
| **--adam_beta2**                         | float   | 0.999   | β2 for AdamW.                                                                                                                                                                                                                                                                                                                                     |
| **--adam_weight_decay**                  | float   | 1e-2    | Weight decay.                                                                                                                                                                                                                                                                                                                                     |
| **--adam_epsilon**                       | float   | 1e-8    | Epsilon for AdamW.                                                                                                                                                                                                                                                                                                                                |
| **--max_grad_norm**                      | float   | 1.0     | Gradient clipping threshold.                                                                                                                                                                                                                                                                                                                      |
| **--allow_tf32**                         | bool    | False   | Allow TF32 on Ampere GPUs for speed.                                                                                                                                                                                                                                                                                                             |
| **--report_to**                          | str     | `wandb` | Logging destination. Options: `tensorboard`, `wandb`, `comet_ml`, `all`.                                                                                                                                                                                                                                                                           |
| **--enable_xformers_memory_efficient_attention** | bool | False   | Use xFormers memory-efficient attention.                                                                                                                                                                                                                                                                                                          |
| **--set_grads_to_none**                  | bool    | False   | Sets grads to `None` instead of zero for possible performance gains.                                                                                                                                                                                                                                                                              |

---

## 3. `train_cyclegan_turbo.py`

A script to train a diffusion-based cycleGAN for **unpaired** datasets. For example:
- Domain A images in `train_A`
- Domain B images in `train_B`
- The model learns both forward (`A->B`) and backward (`B->A`) diffusion transformations.

### Usage Example

```bash
python train_cyclegan_turbo.py \
    --dataset_folder /path/to/dataset \
    --train_img_prep resize_256 \
    --val_img_prep resize_256 \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --output_dir /path/to/output \
    --model_type sdxl \
    --max_train_steps 20000 \
    --validation_steps 1000 \
    --viz_freq 200 \
    --train_batch_size 2 \
    --lambda_gan 0.5 \
    --lambda_cycle 1.0 \
    --lambda_idt 1.0
```

### Command-Line Parameters

Below is a summary of **unpaired**-training arguments from `parse_args_unpaired_training()` (in `training_utils.py`):

| Argument                                 | Type    | Default | Description                                                                                                                                                                                                                                                          |
|------------------------------------------|---------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **--dataset_folder**                     | str     | None    | **Required**. Path to your dataset folder containing `train_A`, `train_B`, possibly `test_A`, `test_B`, plus `fixed_prompt_a.txt` and `fixed_prompt_b.txt`.                                                                                                           |
| **--train_img_prep**                     | str     | None    | Image preprocessing pipeline for training (see `build_transform`).                                                                                                                                                                                                    |
| **--val_img_prep**                       | str     | None    | Image preprocessing pipeline for validation.                                                                                                                                                                                                                         |
| **--dataloader_num_workers**             | int     | 0       | Number of subprocesses for data loading.                                                                                                                                                                                                                             |
| **--train_batch_size**                   | int     | 4       | Batch size for training.                                                                                                                                                                                                                                                                                    |
| **--max_train_epochs**                   | int     | 100     | Number of epochs if not using max_train_steps.                                                                                                                                                                                                                                                             |
| **--max_train_steps**                    | int     | None    | Number of total training steps. If None, uses epochs.                                                                                                                                                                                                                                                      |
| **--seed**                               | int     | 42      | Random seed for reproducibility.                                                                                                                                                                                                                                                                            |
| **--pretrained_model_name_or_path**      | str     | `stabilityai/stable-diffusion-xl-base-1.0` | Base model to fine-tune.                                                                                                                                                                                                                                                                                    |
| **--pretrained_path**                    | str     | None    | Path to local weights (if any).                                                                                                                                                                                                                                                                             |
| **--out_model_name**                     | str     | `test_model` | Name for your final model. Not strictly used by the training script.                                                                                                                                                                                                                                        |
| **--model_type**                         | str     | `sd`    | Which variant: `sd`, `sdxl`, or `sd3`.                                                                                                                                                                                                                                                                       |
| **--revision**                           | str     | None    | Hugging Face revision.                                                                                                                                                                                                                                                                                      |
| **--variant**                            | str     | None    | Hugging Face variant.                                                                                                                                                                                                                                                                                       |
| **--tokenizer_name**                     | str     | None    | If not provided, use default from `pretrained_model_name_or_path`.                                                                                                                                                                                                                                          |
| **--lora_rank_unet**                     | int     | 8       | Rank for LoRA in the UNet.                                                                                                                                                                                                                                                                                  |
| **--lora_rank_vae**                      | int     | 4       | Rank for LoRA in the VAE.                                                                                                                                                                                                                                                                                   |
| **--no_skips**                           | bool    | False   | If set, do not use skip connections in the VAE.                                                                                                                                                                                                                                                             |
| **--freeze_vae**                         | bool    | False   | Freeze VAE weights (train only UNet LoRA).                                                                                                                                                                                                                                                                 |
| **--freeze_unet**                        | bool    | False   | Freeze UNet weights (train only VAE LoRA).                                                                                                                                                                                                                                                                 |
| **--original_unet**                      | bool    | False   | Disable LoRA or custom changes in UNet.                                                                                                                                                                                                                                                                     |
| **--original_vae**                       | bool    | False   | Disable LoRA or custom changes in VAE.                                                                                                                                                                                                                                                                      |
| **--viz_freq**                           | int     | 20      | Frequency (in steps) to log images to W&B or other loggers.                                                                                                                                                                                                                                                 |
| **--output_dir**                         | str     | None    | **Required**. Directory for logs, checkpoints, etc.                                                                                                                                                                                                                                                         |
| **--report_to**                          | str     | `wandb` | Logging destination.                                                                                                                                                                                                                                                                                         |
| **--tracker_project_name**               | str     | None    | W&B project name for logging.                                                                                                                                                                                                                                                                               |
| **--validation_steps**                   | int     | 500     | Frequency (in steps) of validation loops.                                                                                                                                                                                                                                                                   |
| **--validation_num_images**             | int     | -1      | Number of images to use for validation. -1 means use all.                                                                                                                                                                                                                                                   |
| **--checkpointing_steps**               | int     | 500     | Steps between checkpoints.                                                                                                                                                                                                                                                                                  |
| **--learning_rate**                      | float   | 5e-6    | Learning rate.                                                                                                                                                                                                                                                                                              |
| **--adam_beta1**                         | float   | 0.9     | β1 for AdamW.                                                                                                                                                                                                                                                                                               |
| **--adam_beta2**                         | float   | 0.999   | β2 for AdamW.                                                                                                                                                                                                                                                                                               |
| **--adam_weight_decay**                  | float   | 1e-2    | Weight decay for AdamW.                                                                                                                                                                                                                                                                                    |
| **--adam_epsilon**                       | float   | 1e-08   | Epsilon for AdamW.                                                                                                                                                                                                                                                                                          |
| **--max_grad_norm**                      | float   | 10.0    | Gradient clipping threshold.                                                                                                                                                                                                                                                                                |
| **--lr_scheduler**                       | str     | `constant` | Scheduler type.                                                                                                                                                                                                                                                                                             |
| **--lr_warmup_steps**                    | int     | 500     | Warmup steps.                                                                                                                                                                                                                                                                                               |
| **--lr_num_cycles**                      | int     | 1       | Cycles in e.g. cosine_with_restarts.                                                                                                                                                                                                                                                                        |
| **--lr_power**                           | float   | 1.0     | Power factor for polynomial.                                                                                                                                                                                                                                                                                |
| **--gradient_accumulation_steps**        | int     | 1       | Steps to accumulate grads.                                                                                                                                                                                                                                                                                  |
| **--allow_tf32**                         | bool    | False   | TF32 on Ampere GPUs.                                                                                                                                                                                                                                                                                        |
| **--gradient_checkpointing**             | bool    | False   | Use gradient checkpointing.                                                                                                                                                                                                                                                                                 |
| **--enable_xformers_memory_efficient_attention** | bool | False | Use xFormers.                                                                                                                                                                                                                                                                                               |
| **--gan_disc_type**                      | str     | `vagan_clip` | Discriminator type.                                                                                                                                                                                                                                                                                         |
| **--gan_loss_type**                      | str     | `multilevel_sigmoid` | Loss function for the GAN portion.                                                                                                                                                                                                                                                                          |
| **--lambda_gan**                         | float   | 0.5     | Weight for the GAN loss.                                                                                                                                                                                                                                                                                    |
| **--lambda_idt**                         | float   | 1.0     | Weight for identity loss (cycleGAN style).                                                                                                                                                                                                                                                                 |
| **--lambda_cycle**                       | float   | 1.0     | Weight for cycle-consistency loss.                                                                                                                                                                                                                                                                          |
| **--lambda_cycle_lpips**                 | float   | 10.0    | Weight for cycle-consistency LPIPS loss.                                                                                                                                                                                                                                                                    |
| **--lambda_idt_lpips**                   | float   | 1.0     | Weight for identity LPIPS loss.                                                                                                                                                                                                                                                                             |

---

## 4. `training_utils.py`

A helper file containing:
- **Argument parsing** functions:
  - `parse_args_paired_training()`: CLI for `train_pix2pix_turbo.py`.
  - `parse_args_unpaired_training()`: CLI for `train_cyclegan_turbo.py`.
- **Dataset classes**:
  - `PairedDataset`: For paired image-to-image training data (e.g., `train_A` + `train_B`).
  - `PairedDatasetSDXL`: A specialized version for SDXL with embed-at-load workflows.
  - `UnpairedDataset`: For cycleGAN-like unpaired data. It picks random images from domain A and domain B.
  - `UnpairedDatasetSDXL`: Similar unpaired logic for SDXL, with prompt embeddings stored ahead of time.
- **Transforms**:  
  - `build_transform(image_prep)`: Returns a `torchvision.transforms.Compose` for resizing/cropping, etc.  
- **Utility**:
  - `encode_prompt(prompt, tokenizers, text_encoders)`: Encodes text for SDXL.  
  - Additional smaller utility functions (e.g., `str2bool`, logging, etc.).

---

## Installation & Requirements

1. **Python 3.8+** recommended.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # Afterwards:
   pip install torch torchvision torchaudio --upgrade
   pip install xformers --upgrade
   accelerate config default
   ```
3. Make sure to export your wandb token to your environment if using `--report_to wandb`.
4. Make sure to export your huggingface token to your environment when using a gated model like sd3
 
---

## Training Flow

1. **Prepare data**:
   - For **paired** approach:
     - Folder `dataset/train_A`: Input images (domain A).
     - Folder `dataset/train_B`: Target images (domain B).
     - JSON `dataset/train_prompts.json`: Keyed by filename with textual prompts.
     - Similarly for `dataset/test_A`, `dataset/test_B`, `dataset/test_prompts.json`.
   - For **unpaired** approach:
     - Folder `dataset/train_A`, `dataset/train_B` with unaligned images from each domain.
     - `dataset/fixed_prompt_a.txt`: A default domain A prompt.
     - `dataset/fixed_prompt_b.txt`: A default domain B prompt.
     - Similarly for `dataset/test_A`, `dataset/test_B`.

2. **Run script**:
   - **Paired**: `python train_pix2pix_turbo.py [arguments...]`
   - **Unpaired**: `python train_cyclegan_turbo.py [arguments...]`

3. **Check outputs**:
   - `output_dir/checkpoints/...` for saved model files.
   - `output_dir/eval/...` for images/metrics if using `--track_val_fid`.
   - W&B or other logger if specified in `--report_to`.

---


## License

This repository is provided under the MIT License and forked from GaParmar/img2img-turbo. See [LICENSE](LICENSE) for details. 

Happy training!
