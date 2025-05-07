import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
from typing import Any, Callable, Dict, List, Optional, Union
import traceback
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    PretrainedConfig,
    T5EncoderModel,
    CLIPTokenizer,
    T5TokenizerFast,
)
from diffusers import AutoencoderKL, SD3Transformer2DModel
from diffusers import UNet2DConditionModel
from diffusers.loaders.lora_pipeline import SD3LoraLoaderMixin
from diffusers.loaders.single_file import FromSingleFileMixin
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from peft import LoraConfig
import torch.nn as nn
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd

from transformers.pytorch_utils import Conv1D

def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []
    
    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            # model name parsing 

            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
    
    return layer_names


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


def add_channels(unet):
    """
    Expands the UNet from 4 to 8 channels if necessary.
    """
    try:
        print(f"unet current in channels: {unet.conv_in.in_channels}")
        current_in = unet.conv_in.in_channels
    except:
        print(f"unet current in channels: {unet.conv_in.base_layer.in_channels}")
        current_in = unet.conv_in.base_layer.in_channels

    if current_in == 4:
        print("adding additional channels")
        in_channels = 8
        try:
            out_channels = unet.conv_in.out_channels
        except:
            out_channels = unet.conv_in.base_layer.out_channels
        unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            print(f"unet input {unet.conv_in}")
            try:
                new_conv_in = nn.Conv2d(
                    in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
                )
            except:
                new_conv_in = nn.Conv2d(
                    in_channels,
                    out_channels,
                    unet.conv_in.base_layer.kernel_size,
                    unet.conv_in.base_layer.stride,
                    unet.conv_in.base_layer.padding,
                )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in
            print(f"unet output {unet.conv_in}")
        try:
            print(f"unet updated in channels: {unet.conv_in.in_channels}")
        except:
            print(f"unet updated in channels: {unet.conv_in.base_layer.in_channels}")

        return unet


class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction, has_skips=True):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        if has_skips:
            _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


class TwinConv(torch.nn.Module):
    def __init__(self, convin_pretrained, convin_curr):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None

    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.r) + x2 * (self.r)


def encode_prompt(prompt, tokenizers, text_encoders):
    prompt_embeds_list = []

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)


    return prompt_embeds, pooled_prompt_embeds

def encode_prompts(prompts, tokenizers, text_encoders):
    """
    Used for SDXL text-encoder logic. 
    Returns: (prompt_embeds_all, pooled_prompt_embeds_all)
    """
    prompt_embeds_all = []
    pooled_prompt_embeds_all = []

    for prompt in prompts:
        prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt, tokenizers, text_encoders)
        prompt_embeds_all.append(prompt_embeds[0, :])
        pooled_prompt_embeds_all.append(pooled_prompt_embeds[0, :])

    return torch.stack(prompt_embeds_all), torch.stack(pooled_prompt_embeds_all)


def compute_embeddings_for_prompts(prompts, tokenizers, text_encoders, device="cuda"):
    """
    For SDXL use-case: here, we compute text-encoder and pooled text-encoder embeddings.
    """
    with torch.no_grad():
        prompt_embeds_all, pooled_prompt_embeds_all = encode_prompts(prompts, tokenizers, text_encoders)
        add_text_embeds_all = pooled_prompt_embeds_all

        prompt_embeds_all = prompt_embeds_all.to(device)
        add_text_embeds_all = add_text_embeds_all.to(device)
    return prompt_embeds_all, add_text_embeds_all


class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(
        self,
        model_name="stabilityai/sd-turbo",
        pretrained_name=None,
        pretrained_path=None,
        ckpt_folder="checkpoints",
        lora_rank_unet=8,
        lora_rank_vae=4,
        device="cuda",
        freeze_vae=False,
        freeze_unet=False,
        no_skips=False,
        unpaired=False,
        num_inputs=2,
        original_vae=False,
        original_unet=False,
        model_type=None,
        accelerator=None,
    ):
        super().__init__()
        self.unpaired = unpaired
        self.model_name = model_name
        self.has_skips = False
        self.device = device
        self.freeze_unet = freeze_unet
        self.freeze_vae = freeze_vae
        self.no_skips = no_skips
        self.num_inputs = num_inputs
        self.model_type = model_type

        # ------------------------------------------------------------------
        # SETUP: tokenizers, text_encoders, and self.prompt_encoder 
        #        depending on model_type
        # ------------------------------------------------------------------
        if model_type == "sd":
            # Basic stable diffusion
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer", use_fast=False)
            self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
            self.tokenizers = [self.tokenizer]
            self.text_encoders = [self.text_encoder]
            self.sched = make_1step_sched(model_name, device)
            self.device = device
            self.text_encoder.requires_grad_(False)
            self.prompt_encoder = self.encode_prompt_sd  # <--- function for SD

        elif model_type == "sdxl":
            # Stable Diffusion XL
            print("using stable diffusion xl")
            self.tokenizer_1 = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer", use_fast=False)
            self.tokenizer_2 = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer_2", use_fast=False)

            text_encoder_cls_1 = import_model_class_from_model_name_or_path(model_name, None)
            text_encoder_cls_2 = import_model_class_from_model_name_or_path(model_name, None, subfolder="text_encoder_2")

            self.text_encoder_1 = text_encoder_cls_1.from_pretrained(
                model_name, subfolder="text_encoder", revision=None, variant=None
            ).to(device)
            self.text_encoder_2 = text_encoder_cls_2.from_pretrained(
                model_name, subfolder="text_encoder_2", revision=None, variant=None
            ).to(device)
            self.text_encoder_1.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)

            self.sched = make_1step_sched(model_name, device)
            self.device = device
            self.tokenizers = [self.tokenizer_1, self.tokenizer_2]
            self.text_encoders = [self.text_encoder_1, self.text_encoder_2]
            self.prompt_encoder = self.encode_prompt_sdxl  # <--- function for SDXL

        elif model_type == "sd3":
            # stable diffusion 3
            self.tokenizer_1 = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer", use_fast=False)
            self.tokenizer_2 = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer_2", use_fast=False)
            self.tokenizer_3 = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer_3", use_fast=False)

            text_encoder_cls_1 = import_model_class_from_model_name_or_path(model_name, None)
            text_encoder_cls_2 = import_model_class_from_model_name_or_path(
                model_name, None, subfolder="text_encoder_2"
            )
            # text_encoder_cls_3 = import_model_class_from_model_name_or_path(model_name, None, subfolder="text_encoder_3")

            self.text_encoder_1 = text_encoder_cls_1.from_pretrained(
                model_name, subfolder="text_encoder", revision=None, variant=None
            )
            self.text_encoder_2 = text_encoder_cls_2.from_pretrained(
                model_name, subfolder="text_encoder_2", revision=None, variant=None
            )
            # self.text_encoder_3 = text_encoder_cls_3.from_pretrained(
            #     model_name, subfolder="text_encoder_3", revision=None, variant=None
            # )
            self.text_encoder_3 = None  # if you want to load it, do it here

            self.text_encoder_1.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
            # self.text_encoder_3.requires_grad_(False)

            self.device = device
            self.tokenizers = [self.tokenizer_1, self.tokenizer_2, self.tokenizer_3]
            self.text_encoders = [self.text_encoder_1, self.text_encoder_2, self.text_encoder_3]
            self.tokenizer_max_length = 77
            self.prompt_encoder = self.encode_prompt_sd3  # <--- function for SD3

        elif model_type == "flux":
            from diffusers import FluxTransformer2DModel

            # 1) Load tokenizers:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
            self.tokenizer_2   = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer_2")

            # 2) Load text encoders:
            self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
            self.text_encoder_2   = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder_2").to(device)
            self.text_encoder.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)

            self.tokenizer_max_length = (
                    self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
                )
            self.device = device
            self.timesteps = torch.tensor([1000], device=self.device).float()

            self.prompt_encoder = self.encode_prompt_flux # <--- function for FLUX

        # ------------------------------------------------------------------
        # END SETUP
        # ------------------------------------------------------------------

        for i in range(2):
            vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
            if model_type not in  ["sd3","flux"]:
                unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
            elif model_type == "sd3":
                unet = SD3Transformer2DModel.from_pretrained(model_name, subfolder="transformer")
            elif model_type == "flux":

                unet = FluxTransformer2DModel.from_pretrained(model_name, subfolder="transformer")
                self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8

                self.vae_shift_factor = getattr(vae.config, "shift_factor", 0.0)
                self.vae_scaling_factor = getattr(vae.config, "scaling_factor", 1.0)

            unet.requires_grad_(False)
            vae.requires_grad_(False)

            if i == 1 and num_inputs == 3:
                unet = add_channels(unet)

            try:
                target_modules_vae = [
                    "conv1",
                    "conv2",
                    "conv_in",
                    "conv_shortcut",
                    "conv",
                    "conv_out",
                    "skip_conv_1",
                    "skip_conv_2",
                    "skip_conv_3",
                    "skip_conv_4",
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                ]

                if model_type == "flux":
                    target_modules_unet = [
                        "attn.to_k",
                        "attn.to_q",
                        "attn.to_v",
                        "attn.to_out.0",
                        "attn.add_k_proj",
                        "attn.add_q_proj",
                        "attn.add_v_proj",
                        "attn.to_add_out",
                        "ff.net.0.proj",
                        "ff.net.2",
                        "ff_context.net.0.proj",
                        "ff_context.net.2",
                    ]
                    #target_modules_unet = list(set(get_specific_layer_names(unet)))
                    #print(f"lora target modules: {target_modules_unet}")

                elif model_type == "sd3":
                    target_modules_unet = [
                        "attn.add_k_proj",
                        "attn.add_q_proj",
                        "attn.add_v_proj",
                        "attn.to_add_out",
                        "attn.to_k",
                        "attn.to_out.0",
                        "attn.to_q",
                        "attn.to_v",
                    ]
                else:
                    target_modules_unet = [
                        "to_k",
                        "to_q",
                        "to_v",
                        "to_out.0",
                        "conv",
                        "conv1",
                        "conv2",
                        "conv_shortcut",
                        "conv_out",
                        "proj_in",
                        "proj_out",
                        "ff.net.2",
                        "ff.net.0.proj",
                    ]

                if self.unpaired and not model_type in ["sd3","flux"]:
                    target_modules_unet.append("conv_in")

                self.unet_has_adapter = False
                self.vae_has_adapter = False

                if pretrained_path is not None:
                    print("###pretrained")
                    sd = torch.load(pretrained_path, map_location="cpu")

                    if not original_vae:
                        for parameter in sd["state_dict_vae"].keys():
                            if "decoder.skip_conv" in parameter:
                                self.has_skips = True
                                vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
                                vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
                                vae.decoder.skip_conv_1 = torch.nn.Conv2d(
                                    512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                                ).to(device)
                                vae.decoder.skip_conv_2 = torch.nn.Conv2d(
                                    256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                                ).to(device)
                                vae.decoder.skip_conv_3 = torch.nn.Conv2d(
                                    128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                                ).to(device)
                                vae.decoder.skip_conv_4 = torch.nn.Conv2d(
                                    128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                                ).to(device)
                                vae.decoder.ignore_skip = False
                                self.has_skips = True
                                break
                            
                    if sd["unet_lora_target_modules"] and not original_unet:
                        unet_lora_config = LoraConfig(
                            r=sd["rank_unet"],
                            init_lora_weights="gaussian",
                            target_modules=sd["unet_lora_target_modules"],
                        )
                        unet.add_adapter(unet_lora_config)
                        self.unet_has_adapter = True

                    if sd["vae_lora_target_modules"] and not original_vae:
                        vae_lora_config = LoraConfig(
                            r=sd["rank_vae"],
                            init_lora_weights="gaussian",
                            target_modules=sd["vae_lora_target_modules"],
                        )
                        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
                        self.vae_has_adapter = True

                    # load weights
                    if not original_vae:
                        _sd_vae = vae.state_dict()
                        for k in sd["state_dict_vae"]:
                            _sd_vae[k] = sd["state_dict_vae"][k]
                        vae.load_state_dict(_sd_vae)

                        # cyclegan (b2a)
                        if "state_dict_vae_inv" in sd.keys():
                            vae_inv = copy.deepcopy(vae)
                            _sd_vae_inv = vae_inv.state_dict()
                            for k in sd["state_dict_vae_inv"]:
                                _sd_vae_inv[k] = sd["state_dict_vae_inv"][k]

                            self.unpaired = True

                    if not original_unet:
                        _sd_unet = unet.state_dict()
                        for k in sd["state_dict_unet"]:
                            _sd_unet[k] = sd["state_dict_unet"][k]
                        unet.load_state_dict(_sd_unet)

                        if num_inputs == 3 and i == 0:
                            unet = add_channels(unet)  # pretrained model (lora) has only 4 channels
                break

            except Exception as e:
                # Print the full traceback
                print("An error occurred:")
                traceback.print_exc()
                print("pretrained model has additional channels, repeat loading")

        # If we haven't found skip connections from the loaded model and 
        # we specifically want them, add them.
        if not self.has_skips and not no_skips and not original_vae:
            print("adding skips")
            vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
            vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

            vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(
                device
            )
            vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(
                device
            )
            vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(
                device
            )
            vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).to(
                device
            )
            vae.decoder.ignore_skip = False
            self.has_skips = True

            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)

        # Setup new LoRA layers if requested
        if lora_rank_vae > 0 and not self.freeze_vae and not self.vae_has_adapter:
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        elif not self.vae_has_adapter:
            target_modules_vae = []


        if lora_rank_unet > 0 and not self.freeze_unet and not self.unet_has_adapter:
            unet_lora_config = LoraConfig(r=lora_rank_unet, lora_alpha = lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
            unet.add_adapter(unet_lora_config)
        elif not self.unet_has_adapter:
            target_modules_unet = []


        unet.to(self.device)
        vae.to(self.device)
        if self.unpaired:
            vae_inv = copy.deepcopy(vae)

        if self.unpaired:
            vae_inv.to(self.device)
            self.vae_inv = vae_inv
            self.vae_inv.decoder.gamma = 1

        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1

        if self.unpaired:
            self.vae_enc = VAE_encode(vae, vae_b2a=vae_inv)
            self.vae_dec = VAE_decode(vae, vae_b2a=vae_inv)

        if self.model_type not in ["sd3","flux"]:
            self.timesteps = torch.tensor([999], device=self.device).long()
        else:
            self.timesteps = torch.tensor([1000], device=self.device).float()

        if original_vae:
            self.no_skips = True

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        print("finished loading")

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        if self.unpaired:
            self.vae_inv.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def get_traininable_params(self):
        layers_to_opt = []
        for n, _p in self.unet.named_parameters():
            if "lora" in n or self.lora_rank_unet == 0:
                assert _p.requires_grad
                layers_to_opt.append(_p)

        # For SD (not sd3) also train conv_in
        if not self.model_type in ["sd3","flux"]:
            layers_to_opt += list(self.unet.conv_in.parameters())

        for n, _p in self.vae.named_parameters():
            if ("lora" in n and "vae_skip" in n) or self.lora_rank_vae == 0:
                assert _p.requires_grad
                layers_to_opt.append(_p)

        # If we have skip connections, train them
        if not self.no_skips:
            layers_to_opt = (
                layers_to_opt
                + list(self.vae.decoder.skip_conv_1.parameters())
                + list(self.vae.decoder.skip_conv_2.parameters())
                + list(self.vae.decoder.skip_conv_3.parameters())
                + list(self.vae.decoder.skip_conv_4.parameters())
            )

        if self.unpaired:
            for n, _p in self.vae_inv.named_parameters():
                if ("lora" in n and "vae_skip" in n) or self.lora_rank_vae == 0:
                    assert _p.requires_grad
                    layers_to_opt.append(_p)
            if not self.no_skips:
                layers_to_opt = (
                    layers_to_opt
                    + list(self.vae_inv.decoder.skip_conv_1.parameters())
                    + list(self.vae_inv.decoder.skip_conv_2.parameters())
                    + list(self.vae_inv.decoder.skip_conv_3.parameters())
                    + list(self.vae_inv.decoder.skip_conv_4.parameters())
                )

        return layers_to_opt

    def set_train(self):
        if not self.freeze_unet:
            self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n or self.lora_rank_unet == 0:
                _p.requires_grad = True

        if not self.model_type in ["sd3","flux"]:
            self.unet.conv_in.requires_grad_(True)

        if not self.freeze_vae:
            self.vae.train()
            for n, _p in self.vae.named_parameters():
                if "lora" in n or self.lora_rank_vae == 0:
                    _p.requires_grad = True
            if self.unpaired:
                for n, _p in self.vae_inv.named_parameters():
                    if "lora" in n or self.lora_rank_vae == 0:
                        _p.requires_grad = True

        if not self.no_skips:
            self.vae.decoder.skip_conv_1.requires_grad_(True)
            self.vae.decoder.skip_conv_2.requires_grad_(True)
            self.vae.decoder.skip_conv_3.requires_grad_(True)
            self.vae.decoder.skip_conv_4.requires_grad_(True)

            if self.unpaired:
                self.vae_inv.decoder.skip_conv_1.requires_grad_(True)
                self.vae_inv.decoder.skip_conv_2.requires_grad_(True)
                self.vae_inv.decoder.skip_conv_3.requires_grad_(True)
                self.vae_inv.decoder.skip_conv_4.requires_grad_(True)

    # --------------------------------------------------------------------------------
    # HELPER for SDXL: compute the add_time_ids needed by unet
    # --------------------------------------------------------------------------------
    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(tuple(original_size) + tuple(crops_coords_top_left) + tuple(target_size))
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # --------------------------------------------------------------------------------
    # SD3 "prompt encoding" routines, mostly from Hugging Face SD3 pipeline
    # --------------------------------------------------------------------------------
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.device
        dtype = dtype or (self.text_encoder_3.dtype if self.text_encoder_3 else torch.float32)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.unet.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            print(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f"{max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device
        clip_tokenizers = [self.tokenizer_1, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder_1, self.text_encoder_2]
        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index].to(device)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer_max_length, truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt_sd3(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]] = None,
        prompt_3: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
        guidance_scale = None,
    ):
        """
        SD3-style prompt encoding.
        Returns a dict: 
          {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
          }
        """
        device = self.text_encoder_1.device

        # set lora scale so that monkey-patched LoRA function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale
            if self.text_encoder_1 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_1, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_3 = prompt_3 or prompt

            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt, clip_skip=clip_skip, clip_model_index=0
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2, device=device, num_images_per_prompt=num_images_per_prompt, clip_skip=clip_skip, clip_model_index=1
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3, num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length, device=device
            )

            # pad clip prompt to match t5 dims
            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt_1
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt, device=device, num_images_per_prompt=num_images_per_prompt, clip_skip=None, clip_model_index=0
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2, device=device, num_images_per_prompt=num_images_per_prompt, clip_skip=None, clip_model_index=1
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3, num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length, device=device
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1)

        # revert lora scale
        if self.text_encoder_1 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND and lora_scale is not None:
                unscale_lora_layers(self.text_encoder_1, lora_scale)
        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND and lora_scale is not None:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }

    # --------------------------------------------------------------------------------
    # NEW: SD prompt encoding
    # Returns dictionary so that forward(...) can handle them similarly to the SD3 logic
    # --------------------------------------------------------------------------------
    def encode_prompt_sd(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 0.0,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Basic stable diffusion prompt encoding with optional negative prompt support.
        Returns a dict with 
          {
            "prompt_embeds": <tensor>,
            "negative_prompt_embeds": <tensor or None>,
            "pooled_prompt_embeds": None,
            "negative_pooled_prompt_embeds": None
          }
        """
        device = self.device
        # If string, turn into list:
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_list)

        # Encode text
        text_input = self.tokenizer(
            prompt_list,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        prompt_embeds = self.text_encoder(text_input)[0]

        negative_prompt_embeds = None
        if do_classifier_free_guidance and negative_prompt is not None:
            # if negative_prompt is a single string
            negative_prompt_list = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            if len(negative_prompt_list) != batch_size:
                # If mismatch, repeat negative prompt
                if len(negative_prompt_list) == 1:
                    negative_prompt_list = negative_prompt_list * batch_size
                else:
                    raise ValueError("Mismatched lengths for prompt and negative_prompt in SD encode.")

            negative_input = self.tokenizer(
                negative_prompt_list,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            negative_prompt_embeds = self.text_encoder(negative_input)


        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": None,
            "negative_pooled_prompt_embeds": None,
        }

    # --------------------------------------------------------------------------------
    # SDXL prompt encoding
    # Returns dictionary so that forward(...) can handle them similarly to the SD3 logic
    # --------------------------------------------------------------------------------
    def encode_prompt_sdxl(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 0.0,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        SDXL prompt encoding. Internally uses `compute_embeddings_for_prompts(...)`.
        If negative_prompt is provided, we do a 2x duplication of embeddings 
        for classifier-free guidance (uncond + cond).
        Returns a dict with 
          {
            "prompt_embeds": final_prompt_embeds,
            "negative_prompt_embeds": None or separate if we want to store it,
            "pooled_prompt_embeds": final_pooled_embeds,
            "negative_pooled_prompt_embeds": None,
          }
        """
        device = self.device
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_list)

        # Main prompt embeddings
        prompt_embeds, pooled_prompt_embeds = compute_embeddings_for_prompts(
            prompt_list, self.tokenizers, self.text_encoders, device=device
        )

        if do_classifier_free_guidance and negative_prompt is not None:
            # E.g. fallback: 'unsharp' or real negative prompt
            negative_prompt_list = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            if len(negative_prompt_list) != batch_size:
                if len(negative_prompt_list) == 1:
                    negative_prompt_list = negative_prompt_list * batch_size
                else:
                    raise ValueError("Mismatched lengths for prompt and negative_prompt in SDXL encode.")

            negative_prompt_embeds, negative_pooled_prompt_embeds = compute_embeddings_for_prompts(
                negative_prompt_list, self.tokenizers, self.text_encoders, device=device
            )
            # Cat them for uncond + cond
            # i.e. first half is uncond, second half is cond
            final_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            final_pooled_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        else:
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            final_prompt_embeds = prompt_embeds
            final_pooled_embeds = pooled_prompt_embeds

        return {
            "prompt_embeds": final_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": final_pooled_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }

    # ------------------------------------------------------------------------
    # FLUX prompt encoding
    # ------------------------------------------------------------------------

    def _get_t5_prompt_embeds_flux(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)


        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            print(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds_flux(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds
    
    def encode_prompt_flux(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        negative_prompt=None, # compatibility
        guidance_scale = None, # compatibility
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = self.device 

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds_flux(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds_flux(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.unet.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)


        return {
            "prompt_embeds": prompt_embeds,  
            "negative_prompt_embeds": None,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": None,
            "text_ids": text_ids
        }

    @staticmethod
    def _pack_latents_flux(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def _unpack_latents_flux(self,latents, height, width):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents
    
    @staticmethod
    def _prepare_latent_image_ids_flux(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)


    def prepare_latents_flux(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        #height = 2 * (int(height) // (self.vae_scale_factor * 2))
        #width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        latents = self._pack_latents_flux(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids_flux(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids
    
    # --------------------------------------------------------------------------------
    # FORWARD CYCLE 
    # --------------------------------------------------------------------------------
    def forward_cycle(self, x, direction, timesteps, text_emb, x_2=None):
        """
        One forward pass for cyclegan-like approach. 
        """
        B = x.shape[0]
        assert direction in ["a2b", "b2a"]
        x_enc = self.vae_enc(x, direction=direction).to(x.dtype)

        if self.num_inputs == 3:
            x_enc_2 = self.vae_enc(x, direction=direction).to(x.dtype)
            x_conc = torch.cat([x_enc, x_enc_2], dim=1)
        else:
            x_conc = x_enc

        if self.model_type == "sd":

            prompt_embeds = text_emb["prompt_embeds"]
            model_pred = self.unet(x_conc, timesteps, encoder_hidden_states=prompt_embeds).sample
            x_out = torch.stack(
                [self.sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)]
            )
        elif self.model_type == "sdxl":

            prompt_embeds = text_emb["prompt_embeds"]
            pooled_embeds = text_emb["pooled_prompt_embeds"]

            add_time_ids = self._get_add_time_ids(x.size()[-2:], [0, 0], x.size()[-2:], x_enc.dtype)
            add_time_ids = add_time_ids.to(pooled_embeds.device).repeat(x_enc.size()[0], 1)
            added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}

            model_pred = self.unet(x_conc, timesteps, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample
            x_out = torch.stack(
                [self.sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)]
            )
        elif self.model_type == "sd3":
            prompt_embeds = text_emb["prompt_embeds"]
            pooled_prompt_embeds = text_emb["pooled_prompt_embeds"]

            model_pred = self.unet(
                hidden_states=x_conc,
                timestep=self.timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
            x_out = x_conc - model_pred

        elif self.model_type == "flux":

            guidance_scale =3.5

            prompt_embeds = text_emb["prompt_embeds"]
            pooled_prompt_embeds = text_emb["pooled_prompt_embeds"]

            num_images_per_prompt=1
            batch_size, num_channels_latents, height, width = x_conc.shape

            latents = (x_conc * self.vae.config.scaling_factor - self.vae.config.shift_factor)

            latents, latent_image_ids = self.prepare_latents_flux(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    x_conc.device,
                    latents,
                )
            
            t_tensor = self.timesteps.expand(latents.size(0)).to(latents.dtype)/1000 #multiplied in transformer by 1000 

            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)

            #if self.unet.config.guidance_embeds: <- attribute not available in distributed training
            guidance = torch.full([1], guidance_scale, device=x_conc.device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
            #else:
                #guidance = None

            out = self.unet(
                    hidden_states=latents.to(self.unet.device),
                    timestep=t_tensor.to(self.unet.device),
                    pooled_projections=pooled_prompt_embeds.to(self.unet.device),
                    encoder_hidden_states=prompt_embeds.to(self.unet.device),
                    return_dict=False,
                    img_ids=latent_image_ids.to(self.unet.device),
                    txt_ids=text_ids.to(self.unet.device),
                    guidance=guidance.to(self.unet.device),
                )
            model_pred = out[0]

            latents = latents - model_pred  

            # "unpack"
            latents = self._unpack_latents_flux(latents, height, width)

            # scale+shift => decode
            x_out = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            if self.has_skips:
                self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            #output_image = self.vae.decode(latents, return_dict=False)[0]
            #output_image = output_image.clamp(-1, 1)


        x_out_decoded = self.vae_dec(x_out, direction=direction, has_skips=self.has_skips)
        return x_out_decoded

    # --------------------------------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------------------------------
    def forward(
        self,
        c_t,
        c_t_2=None,
        prompt=None,
        prompt_tokens=None,
        caption_enc=None,
        caption_enc_pooled=None,
        deterministic=True,
        r=1.0,
        noise_map=None,
        tokenizers=None,
        text_encoders=None,
        negative_prompt=None,
        guidance_scale=0,
        direction=None,
    ):
        
        """
        Forward pass. Uses self.prompt_encoder to encode text (SD, SDXL, or SD3).
        Then feeds to unet, does one schedule step, and decodes result with VAE.
        """

        if getattr(self, "model_type", None) == "flux" and guidance_scale == 0:
            guidance_scale = 3.5

        if self.unpaired:
            raise NotImplementedError("It is not implemented to call the forward function for unpaired tasks yet. Use forward_cycle instead")
        B = c_t.shape[0]

        # 1) ENCODE TEXT
        # -----------------------------------------------------
        # For the given model type, call the relevant encoder function:
        text_emb_dict = self.prompt_encoder(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
        )
        prompt_embeds = text_emb_dict["prompt_embeds"]
        negative_prompt_embeds = text_emb_dict["negative_prompt_embeds"]  
        pooled_prompt_embeds = text_emb_dict["pooled_prompt_embeds"]
        negative_pooled_prompt_embeds = text_emb_dict["negative_pooled_prompt_embeds"]


        # 2) ENCODE CONTROL (VAE) 
        # -----------------------------------------------------
        if self.unpaired:
            encoded_control = self.vae_enc(c_t.to(self.device), direction=direction).to(c_t.dtype)
            if self.num_inputs == 3:
                encoded_control_2 = self.vae_enc(c_t_2.to(self.device), direction=direction).to(c_t.dtype)
                encoded_control_net = torch.cat([encoded_control, encoded_control_2], dim=1)
            else:
                encoded_control_net = encoded_control
        else:
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            if self.num_inputs == 3:
                encoded_control_2 = self.vae.encode(c_t_2).latent_dist.sample() * self.vae.config.scaling_factor
                encoded_control_net = torch.cat([encoded_control, encoded_control_2], dim=1)
            else:
                encoded_control_net = encoded_control

        # For classifier-free guidance in SDXL we typically cat the unconditional 
        # embeddings. We'll handle that inside the call below.
        # -----------------------------------------------------
        if self.model_type == "sd":
            # same as your existing code for "sd"
            model_pred = self.unet(encoded_control_net, self.timesteps, encoder_hidden_states=prompt_embeds).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control_net, return_dict=True).prev_sample
            # decode
            if self.has_skips:
                self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        elif self.model_type == "sdxl":
            # same as your existing code for "sdxl"
            add_time_ids = self._get_add_time_ids(c_t.size()[-2:], [0, 0], c_t.size()[-2:], c_t.dtype)
            add_time_ids = add_time_ids.to(self.device).repeat(c_t.size()[0], 1)
            # If negative_prompt is given, we do uncond + cond cat
            if negative_prompt is not None:
                encoded_control_net = torch.cat([encoded_control_net] * 2)
                add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
                prompt_embeds_combined = prompt_embeds
            else:
                prompt_embeds_combined = prompt_embeds
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(encoded_control_net, self.timesteps, encoder_hidden_states=prompt_embeds_combined, added_cond_kwargs=added_cond_kwargs).sample
            if negative_prompt is not None:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                model_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                model_pred = noise_pred
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            if self.has_skips:
                self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        elif self.model_type == "sd3":
            # same as your existing code for "sd3"
            model_pred = self.unet(
                hidden_states=encoded_control_net,
                timestep=self.timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
            x_denoised = encoded_control_net - model_pred
            if self.has_skips:
                self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        elif self.model_type == "flux":

            num_images_per_prompt=1
            batch_size, num_channels_latents, height, width = encoded_control_net.shape

            latents = (encoded_control_net * self.vae.config.scaling_factor - self.vae.config.shift_factor)

            latents, latent_image_ids = self.prepare_latents_flux(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    c_t.device,
                    latents,
                )
            

            t_tensor = self.timesteps.expand(latents.size(0)).to(latents.dtype)/1000 #multiplied in transformer by 1000 


            text_ids = text_emb_dict["text_ids"]

            #if self.unet.config.guidance_embeds:  <- attribute not available in distributed training
            guidance = torch.full([1], guidance_scale, device=c_t.device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
            #else:
                #guidance = None

            out = self.unet(
                    hidden_states=latents.to(self.unet.device),
                    timestep=t_tensor.to(self.unet.device),
                    pooled_projections=pooled_prompt_embeds.to(self.unet.device),
                    encoder_hidden_states=prompt_embeds.to(self.unet.device),
                    return_dict=False,
                    img_ids=latent_image_ids.to(self.unet.device),
                    txt_ids=text_ids.to(self.unet.device),
                    guidance=guidance.to(self.unet.device),
                )
            model_pred = out[0]

            latents = latents - model_pred  

            # "unpack"
            latents = self._unpack_latents_flux(latents, height, width)

            # scale+shift => decode
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            if self.has_skips:
                self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = self.vae.decode(latents, return_dict=False)[0]
            output_image = output_image.clamp(-1, 1)

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items()}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items()}
        torch.save(sd, outf)
