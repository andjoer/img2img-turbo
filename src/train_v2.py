import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, ProfileKwargs
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import parse_args_paired_training, PairedDataset
from transformers import AutoTokenizer, PretrainedConfig
import vision_aided_loss

from diffusers.utils.torch_utils import is_compiled_module

import patch_loss

import psutil
import os

import time

def show_ram_usage():
    import psutil, os
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size: physical memory
    mem_mb = mem_bytes / (1024 ** 2)
    print(f"Current memory usage: {mem_mb:.2f} MB")

def human_readable_size(num_bytes: int) -> str:
    """Convert a byte value into a human-readable string (MB)."""
    return f"{num_bytes / (1024**2):.2f} MB"

def measure_model_sizes(model: Pix2Pix_Turbo) -> None:
    """
    Prints parameter sizes (data + gradient) for:
      - UNet
      - VAE
      - text_encoder (if available)
      - text_encoder_2 (if available)
      - (optionally) VAE_inv if unpaired

    * "PARAM total" = sum of all param.data sizes
    * "trainable only" = sum of sizes where param.requires_grad=True
    * same breakdown for gradients if param.grad is not None.

    Make sure you've run a forward+backward pass if you want to see non-zero gradient sizes.
    """

    def measure_submodule(submodule):
        data_total = 0
        data_trainable = 0
        grad_total = 0
        grad_trainable = 0
        for name, param in submodule.named_parameters():
            # parameter data
            data_sz = param.numel() * param.element_size()
            data_total += data_sz
            if param.requires_grad:
                data_trainable += data_sz

            # gradient (if allocated)
            grad_sz = 0
            if param.grad is not None:
                grad_sz = param.grad.numel() * param.grad.element_size()

            grad_total += grad_sz
            if param.requires_grad:
                grad_trainable += grad_sz

        return data_total, data_trainable, grad_total, grad_trainable

    # -----------------------------
    # 1) Measure UNet
    # -----------------------------
    unet_data_total, unet_data_trainable, unet_grad_total, unet_grad_trainable = measure_submodule(model.unet)

    # -----------------------------
    # 2) Measure VAE
    # -----------------------------
    vae_data_total, vae_data_trainable, vae_grad_total, vae_grad_trainable = measure_submodule(model.vae)

    # -----------------------------
    # 3) If unpaired => measure VAE_inv
    # -----------------------------
    vae_inv_data_total = 0
    vae_inv_data_trainable = 0
    vae_inv_grad_total = 0
    vae_inv_grad_trainable = 0
    if getattr(model, "unpaired", False) and hasattr(model, "vae_inv"):
        (vae_inv_data_total,
         vae_inv_data_trainable,
         vae_inv_grad_total,
         vae_inv_grad_trainable) = measure_submodule(model.vae_inv)

    # -----------------------------
    # 4) text_encoder (if it exists)
    # -----------------------------
    textenc_data_total = 0
    textenc_data_trainable = 0
    textenc_grad_total = 0
    textenc_grad_trainable = 0
    if hasattr(model, "text_encoder") and model.text_encoder is not None:
        (textenc_data_total,
         textenc_data_trainable,
         textenc_grad_total,
         textenc_grad_trainable) = measure_submodule(model.text_encoder)

    # -----------------------------
    # 5) text_encoder_2 (if it exists)
    # -----------------------------
    textenc2_data_total = 0
    textenc2_data_trainable = 0
    textenc2_grad_total = 0
    textenc2_grad_trainable = 0
    if hasattr(model, "text_encoder_2") and model.text_encoder_2 is not None:
        (textenc2_data_total,
         textenc2_data_trainable,
         textenc2_grad_total,
         textenc2_grad_trainable) = measure_submodule(model.text_encoder_2)

    # -----------------------------
    # Print results
    # -----------------------------
    print("==== Parameter + Gradient Size Report ====")
    # UNet
    print(f"UNet PARAM total     : {human_readable_size(unet_data_total)}")
    print(f"  └─ trainable only  : {human_readable_size(unet_data_trainable)}")
    print(f"UNet GRAD total      : {human_readable_size(unet_grad_total)}")
    print(f"  └─ trainable only  : {human_readable_size(unet_grad_trainable)}")

    # VAE
    print(f"\nVAE PARAM total      : {human_readable_size(vae_data_total)}")
    print(f"  └─ trainable only  : {human_readable_size(vae_data_trainable)}")
    print(f"VAE GRAD total       : {human_readable_size(vae_grad_total)}")
    print(f"  └─ trainable only  : {human_readable_size(vae_grad_trainable)}")

    # VAE_inv (only if unpaired)
    if getattr(model, "unpaired", False):
        print(f"\nVAE_inv PARAM total  : {human_readable_size(vae_inv_data_total)}")
        print(f"  └─ trainable only  : {human_readable_size(vae_inv_data_trainable)}")
        print(f"VAE_inv GRAD total   : {human_readable_size(vae_inv_grad_total)}")
        print(f"  └─ trainable only  : {human_readable_size(vae_inv_grad_trainable)}")

    # text_encoder
    if hasattr(model, "text_encoder") and model.text_encoder is not None:
        print(f"\nText Encoder PARAM total: {human_readable_size(textenc_data_total)}")
        print(f"  └─ trainable only     : {human_readable_size(textenc_data_trainable)}")
        print(f"Text Encoder GRAD total : {human_readable_size(textenc_grad_total)}")
        print(f"  └─ trainable only     : {human_readable_size(textenc_grad_trainable)}")

    # text_encoder_2
    if hasattr(model, "text_encoder_2") and model.text_encoder_2 is not None:
        print(f"\nText Encoder 2 PARAM total: {human_readable_size(textenc2_data_total)}")
        print(f"  └─ trainable only       : {human_readable_size(textenc2_data_trainable)}")
        print(f"Text Encoder 2 GRAD total : {human_readable_size(textenc2_grad_total)}")
        print(f"  └─ trainable only       : {human_readable_size(textenc2_grad_trainable)}")

    print("===========================================")




def report_duplicates(optimizer, opt_name=""):
    seen = set()
    for group_idx, param_group in enumerate(optimizer.param_groups):
        for p in param_group["params"]:
            pid = id(p)
            if pid in seen:
                print(f"[{opt_name}] Duplicate param in param_group {group_idx} -> shape={p.shape}")
            else:
                seen.add(pid)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def unwrap_model(model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def main(args):

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    profile_kwargs = ProfileKwargs(
    activities=["cuda"] if torch.cuda.is_available() else ["cpu"],
    profile_memory=True,
    record_shapes=True
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers=[profile_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # Updated to use `args.validation_image_prep` instead of `args.test_image_prep`
    dataset_val = PairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.validation_image_prep,
        split="test"
    )
    dataset_train = PairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.train_image_prep,
        split="train"
    )

    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers
    )
    dl_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    net_pix2pix = Pix2Pix_Turbo(
        model_name=args.pretrained_model_name_or_path,
        pretrained_path=args.pretrained_path,
        model_type=args.model_type,
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        device=accelerator.device,
        freeze_unet=args.freeze_unet,
        freeze_vae=args.freeze_vae,
        no_skips=args.no_skips,
        num_inputs=args.num_inputs,
        original_unet=args.original_unet,
        original_vae=args.original_vae,
        accelerator=accelerator
    )

    show_ram_usage()
    measure_model_sizes(net_pix2pix)


    if args.enable_xformers_memory_efficient_attention and "mps" not in str(accelerator.device):
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.lambda_gan > 0:
        if args.gan_disc_type == "vagan_clip":
            import vision_aided_loss
            net_disc = vision_aided_loss.Discriminator(
                cv_type='clip',
                loss_type=args.gan_loss_type,
                device=accelerator.device
            )
        else:
            raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

        net_disc = net_disc.to(accelerator.device)
        net_disc.requires_grad_(True)
        net_disc.cv_ensemble.requires_grad_(False)
        net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').to(accelerator.device)
    net_clip, _ = clip.load("ViT-B/32", device=accelerator.device)
    net_clip.requires_grad_(False)
    net_clip.eval()
    net_lpips.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    if not args.original_unet:
        for n, _p in net_pix2pix.unet.named_parameters():
            if "lora" in n:
                assert _p.requires_grad
                layers_to_opt.append(_p)
        if not args.model_type in ["sd3","flux"]:
            layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())

    if not args.freeze_vae:
        for n, _p in net_pix2pix.vae.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert _p.requires_grad
                layers_to_opt.append(_p)
        if not args.no_skips and not args.original_vae:
            layers_to_opt = (
                layers_to_opt
                + list(net_pix2pix.vae.decoder.skip_conv_1.parameters())
                + list(net_pix2pix.vae.decoder.skip_conv_2.parameters())
                + list(net_pix2pix.vae.decoder.skip_conv_3.parameters())
                + list(net_pix2pix.vae.decoder.skip_conv_4.parameters())
            )

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if args.lambda_gan > 0:
        optimizer_disc = torch.optim.AdamW(
            net_disc.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

   
        lr_scheduler_disc = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power
        )

    report_duplicates(optimizer, "G_optim")

    if "mps" in str(accelerator.device):  # size needs to be divisible by 224
        resize_to_opt = [224, 448, 672, 896, 1120]
        image_size = dataset_train[0]["conditioning_pixel_values"].size()[-1]
        resize_to = max([num for num in resize_to_opt if num < image_size])

   
    # Prepare everything with our `accelerator`.
    (
        net_pix2pix,
        optimizer,
        dl_train,
        lr_scheduler,
    ) = accelerator.prepare(
        net_pix2pix, optimizer, dl_train, lr_scheduler
    )
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)

    if args.lambda_gan > 0:
        net_disc, optimizer_disc, lr_scheduler_disc = accelerator.prepare(
            net_disc, optimizer_disc, lr_scheduler_disc
        )
    # renorm with image net statistics
    t_clip_renorm = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move all networks to device and cast to weight_dtype
    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    
    ####################################
    ############# prepare discriminator#
    ####################################
    if args.lambda_gan > 0:
        net_disc.to(accelerator.device, dtype=weight_dtype)
        # Turn off memory-efficient attention for the discriminator
        for name, module in net_disc.named_modules():
            if "attn" in name:
                module.fused_attn = False

        if "mps" in str(accelerator.device):
            from prepare_disc import prepare_cv_ensamble_mps
            net_disc = prepare_cv_ensamble_mps(args.cv_type,net_disc)

    ###################################

    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    #net_pix2pix.vae.to("cpu")

    # compute the reference stats for FID tracking
    if accelerator.is_main_process and args.track_val_fid:
        feat_model = build_feature_extractor("clean", accelerator.device, use_dataparallel=False)

        def fn_transform(x):
            x_pil = Image.fromarray(x)
            out_pil = transforms.Resize(
                args.resolution,
                interpolation=transforms.InterpolationMode.LANCZOS
            )(x_pil)
            return np.array(out_pil)

        ref_stats = get_folder_features(
            os.path.join(args.dataset_folder, "test_B"),
            model=feat_model,
            num_workers=0,
            num=None,
            shuffle=False,
            seed=0,
            batch_size=8,
            device=torch.device(accelerator.device),
            mode="clean",
            custom_image_tranform=fn_transform,
            description="",
            verbose=True
        )

    global_step = 0

    for epoch in range(0, args.max_train_epochs):
        for step, batch in enumerate(dl_train):
            if args.lambda_gan > 0:
                l_acc = [net_pix2pix, net_disc]
            else:
                l_acc = [net_pix2pix]

            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                if args.num_inputs == 3:                                
                    x_src_2 = batch["conditioning_pixel_values_2"]
                else:
                    x_src_2 = None
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape

                # forward pass
                x_tgt_pred = net_pix2pix(x_src, c_t_2=x_src_2, prompt=batch["caption"], deterministic=True)
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                # Use the patched LPIPS function in place of a direct call.
                loss_lpips = compute_patched_lpips_loss(net_lpips, x_tgt_pred.float(), x_tgt.float(), args) * args.lambda_lpips
                loss = loss_l2 + loss_lpips

                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(
                        x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False
                    )
                    caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = (1 - clipsim.mean() / 100)
                    loss += loss_clipsim * args.lambda_clipsim

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                if args.lambda_gan > 0:
                    # Generator GAN loss: use the patched discriminator loss.
                    lossG = compute_patched_disc_loss(net_disc, x_tgt_pred.float(), {"for_G": True}, args) * args.lambda_gan
                    accelerator.backward(lossG)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    # Discriminator losses:
                    lossD_real = compute_patched_disc_loss(net_disc, x_tgt.float().detach(), {"for_real": True}, args) * args.lambda_gan
                    accelerator.backward(lossD_real)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                    lossD_fake = compute_patched_disc_loss(net_disc, x_tgt_pred.float().detach(), {"for_real": False}, args) * args.lambda_gan
                    accelerator.backward(lossD_fake)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                    lossD = lossD_real + lossD_fake
                else:
                    lossD = torch.tensor(0)
                    lossG = torch.tensor(0)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    logs = {}
                    loss_l2_val = loss_l2.detach().item()
                    logs["lossG"] = lossG.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2_val
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    progress_bar.set_postfix(**logs)
                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(x_src[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]
                    if global_step % args.checkpointing_steps == 1 and global_step > 10:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)
                    if global_step % args.validation_steps == 1 and global_step > 3:
                        print('start validation')
                        l_l2, l_lpips_vals, l_clipsim = [], [], []
                        if args.track_val_fid:
                            os.makedirs(
                                os.path.join(args.output_dir, "eval", f"fid_{global_step}"),
                                exist_ok=True
                            )
                        for step_val, batch_val in enumerate(dl_val):
                            if args.validation_num_images != -1 and step_val >= args.validation_num_images:
                                break
                            x_src_val = batch_val["conditioning_pixel_values"].to(accelerator.device)
                            x_tgt_val = batch_val["output_pixel_values"].to(accelerator.device)
                            if args.num_inputs == 3:
                                x_src_2_val = batch_val["conditioning_pixel_values_2"]
                            else:
                                x_src_2_val = None
                            B_val, C_val, H_val, W_val = x_src_val.shape
                            assert B_val == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                x_tgt_pred_val = net_pix2pix(
                                    x_src_val, c_t_2=x_src_2_val,
                                    prompt=batch_val["caption"],
                                    deterministic=True
                                )
                                loss_l2_ = F.mse_loss(x_tgt_pred_val.float(), x_tgt_val.float(), reduction="mean")
                                loss_lpips_ = net_lpips(x_tgt_pred_val.float(), x_tgt_val.float()).mean()
                                x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred_val * 0.5 + 0.5)
                                x_tgt_pred_renorm = F.interpolate(
                                    x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False
                                )
                                caption_tokens = clip.tokenize(batch_val["caption"], truncate=True).to(x_tgt_pred_val.device)
                                clipsim_, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                                clipsim_ = clipsim_.mean()
                                l_l2.append(loss_l2_.item())
                                l_lpips_vals.append(loss_lpips_.item())
                                l_clipsim.append(clipsim_.item())
                            if args.track_val_fid:
                                output_pil = transforms.ToPILImage()(
                                    x_tgt_pred_val[0].cpu() * 0.5 + 0.5
                                )
                                outf = os.path.join(
                                    args.output_dir, "eval", f"fid_{global_step}", f"val_{step_val}.png"
                                )
                                output_pil.save(outf)
                        if args.track_val_fid:
                            curr_stats = get_folder_features(
                                os.path.join(args.output_dir, "eval", f"fid_{global_step}"),
                                model=feat_model,
                                num_workers=0,
                                num=None,
                                shuffle=False,
                                seed=0,
                                batch_size=8,
                                device=torch.device(accelerator.device),
                                mode="clean",
                                custom_image_tranform=fn_transform,
                                description="",
                                verbose=True
                            )
                            fid_score = fid_from_feats(ref_stats, curr_stats)
                            logs["val/clean_fid"] = fid_score
                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips_vals)
                        logs["val/clipsim"] = np.mean(l_clipsim)
                        gc.collect()
                        if "mps" in str(accelerator.device):
                            torch.mps.empty_cache()
                        elif "cuda" in str(accelerator.device):
                            torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
