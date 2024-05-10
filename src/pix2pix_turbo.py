import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd

from my_utils.training_utils import encode_prompt
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

from diffusers.utils.torch_utils import is_compiled_module
import torch.nn as nn


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

    def forward(self, x, direction,has_skips=True):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        #assert _vae.encoder.current_down_blocks is not None
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

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompts(prompts, tokenizers, text_encoders):
    prompt_embeds_all = []
    pooled_prompt_embeds_all = []

    for prompt in prompts:
        prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt,tokenizers,text_encoders)
        prompt_embeds_all.append(prompt_embeds[0,:])
        pooled_prompt_embeds_all.append(pooled_prompt_embeds[0,:])

    return torch.stack(prompt_embeds_all), torch.stack(pooled_prompt_embeds_all)
# Adapted from examples.dreambooth.train_dreambooth_lora_sdxl
# Here, we compute not just the text embeddings but also the additional embeddings
# needed for the SD XL UNet to operate.
def compute_embeddings_for_prompts(prompts, tokenizers, text_encoders,device="cuda"):
    with torch.no_grad():
        prompt_embeds_all, pooled_prompt_embeds_all = encode_prompts(prompts, tokenizers,text_encoders)
        add_text_embeds_all = pooled_prompt_embeds_all

        prompt_embeds_all = prompt_embeds_all.to(device)
        add_text_embeds_all = add_text_embeds_all.to(device)
    return prompt_embeds_all, add_text_embeds_all




class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, model_name= "stabilityai/sd-turbo",is_sdxl=False,pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4,device="cuda",
    tokenizers=None, text_encoders=None,freeze_vae=False,freeze_unet=False, no_skips=False,unpaired=False):
        super().__init__()
        self.unpaired = unpaired
        self.model_name = model_name
        self.has_skips = False
        self.device = device
        self.freeze_unet = freeze_unet
        self.freeze_vae = freeze_vae
        self.no_skips = no_skips
        self.is_sdxl = is_sdxl



        if not is_sdxl:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer",use_fast=False)
            self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
            self.sched = make_1step_sched(model_name, device)
            self.device = device
            self.text_encoder.requires_grad_(False)
            vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        else:
            print('using stable diffusion xl')
            self.tokenizer_1 = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer",use_fast=False)
           
            self.tokenizer_2 = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer_2",use_fast=False)

            text_encoder_cls_1 = import_model_class_from_model_name_or_path(model_name, None)
            text_encoder_cls_2 = import_model_class_from_model_name_or_path(model_name, None, subfolder="text_encoder_2"
            )

            self.text_encoder_1 = text_encoder_cls_1.from_pretrained(
                model_name, subfolder="text_encoder", revision=None, variant=None
            )
            self.text_encoder_2 = text_encoder_cls_2.from_pretrained(
                model_name, subfolder="text_encoder_2", revision=None, variant=None
            )
            self.text_encoder_1.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
            self.sched = make_1step_sched(model_name, device)
            self.device = device
            self.tokenizers=[self.tokenizer_1,self.tokenizer_2]
            self.text_encoders=[self.text_encoder_1,self.text_encoder_2]
            vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")

        if self.unpaired:
            vae_inv = copy.deepcopy(vae)

        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

        target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]

         #l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"] -> conv_in missing in pix2pix
        target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ] 

        if self.unpaired: 
            target_modules_unet.append("conv_in")
              
        self.unet_has_adapter = False
        self.vae_has_adapter = False

        if pretrained_path is not None:
            print('###pretrained')
            sd = torch.load(pretrained_path, map_location="cpu")

            for parameter in sd["state_dict_vae"].keys():
                if "decoder.skip_conv" in parameter:
                    self.has_skips = True
                    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
                    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
                    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
                    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
                    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
                    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
                    vae.decoder.ignore_skip = False
                    self.has_skips = True
                    break 
            if sd["unet_lora_target_modules"]:
                unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
                unet.add_adapter(unet_lora_config)
                self.unet_has_adapter = True
            if sd["vae_lora_target_modules"]:
                vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
                vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
                self.vae_has_adapter = True

            # pix2pix or vae of cyclegan
            
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)

            # cyclegan
            if "state_dict_vae_inv" in sd.keys():
                vae_inv = copy.deepcopy(vae)

                _sd_vae_inv = vae_inv.state_dict()
                for k in sd["state_dict_vae_inv"]:
                    _sd_vae_inv[k] = sd["state_dict_vae_inv"][k]

                self.unpaired = True

            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        if not self.has_skips and not no_skips:
            print('adding skips')
            # forward functions for vae with skip

            vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
            vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

            vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
            vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
            vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
            vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
            vae.decoder.ignore_skip = False
            self.has_skips = True

            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)

            if self.unpaired:          # temporary solution, needs refacturing

                vae_inv.encoder.forward = my_vae_encoder_fwd.__get__(vae_inv.encoder, vae_inv.encoder.__class__)
                vae_inv.decoder.forward = my_vae_decoder_fwd.__get__(vae_inv.decoder, vae_inv.decoder.__class__)

                vae_inv.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
                vae_inv.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
                vae_inv.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
                vae_inv.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
                vae_inv.decoder.ignore_skip = False

                print("Initializing model with random weights")
                torch.nn.init.constant_(vae_inv.decoder.skip_conv_1.weight, 1e-5)
                torch.nn.init.constant_(vae_inv.decoder.skip_conv_2.weight, 1e-5)
                torch.nn.init.constant_(vae_inv.decoder.skip_conv_3.weight, 1e-5)
                torch.nn.init.constant_(vae_inv.decoder.skip_conv_4.weight, 1e-5)

        if lora_rank_vae > 0 and not self.freeze_vae and not self.vae_has_adapter:

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            
            if not self.vae_has_adapter:
                vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        elif not self.vae_has_adapter:
            target_modules_vae = []

        if lora_rank_unet > 0 and not self.freeze_unet and not self.unet_has_adapter:
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)
        elif not self.unet_has_adapter:
            target_modules_unet = []

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        #unet.enable_xformers_memory_efficient_attention()
        unet.to(self.device)
        vae.to(self.device)

        if self.unpaired:
            vae_inv.to(self.device)
            self.vae_inv = vae_inv
            self.vae_inv.decoder.gamma = 1
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device=self.device).long()

        if self.is_sdxl:
            self.text_encoder_1.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
        else:
            self.text_encoder.requires_grad_(False)

        if self.unpaired: 
            self.vae_enc=VAE_encode(vae,vae_b2a=vae_inv)
            self.vae_dec=VAE_decode(vae,vae_b2a=vae_inv)

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
        layers_to_opt += list(self.unet.conv_in.parameters())
        for n, _p in self.vae.named_parameters():
            if ("lora" in n and "vae_skip" in n) or self.lora_rank_vae == 0:
                assert _p.requires_grad
                layers_to_opt.append(_p)

        if not self.no_skips:
            layers_to_opt = layers_to_opt + list(self.vae.decoder.skip_conv_1.parameters()) + \
                list(self.vae.decoder.skip_conv_2.parameters()) + \
                list(self.vae.decoder.skip_conv_3.parameters()) + \
                list(self.vae.decoder.skip_conv_4.parameters())

        if self.unpaired:
            for n, _p in self.vae_inv.named_parameters():
                if ("lora" in n and "vae_skip" in n) or self.lora_rank_vae == 0:
                    assert _p.requires_grad
                    layers_to_opt.append(_p)

            if not self.no_skips:
                layers_to_opt = layers_to_opt + list(self.vae_inv.decoder.skip_conv_1.parameters()) + \
                    list(self.vae_inv.decoder.skip_conv_2.parameters()) + \
                    list(self.vae_inv.decoder.skip_conv_3.parameters()) + \
                    list(self.vae_inv.decoder.skip_conv_4.parameters())
            
        return layers_to_opt

    def set_train(self):
        if not self.freeze_unet:
            self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n or self.lora_rank_unet == 0:
                _p.requires_grad = True
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

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(tuple(original_size) + tuple(crops_coords_top_left) + tuple(target_size))

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids


    def forward(self, c_t, prompt=None, prompt_tokens=None, caption_enc=None, caption_enc_pooled=None, deterministic=True, r=1.0, noise_map=None,tokenizers=None,text_encoders=None,negative_prompt=None, guidance_scale=0,direction=None):
        # either the prompt or the prompt_tokens should be provided
        #assert ((prompt is None) != (prompt_tokens is None)), "Either prompt or prompt_tokens should be provided"

        #negative_prompt = "unsharp"

        B = c_t.shape[0]   # needs refactor 
        if not self.is_sdxl:
            if prompt is not None:
                # encode the text prompt
                caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                                padding="max_length", truncation=True, return_tensors="pt").input_ids.to(self.device)
                caption_enc = self.text_encoder(caption_tokens)[0]

                #caption_enc_pooled = caption_enc[0]
            else:

                caption_enc = self.text_encoder(prompt_tokens)[0]

        else: 

            caption_enc, caption_enc_pooled = compute_embeddings_for_prompts(prompt, self.tokenizers, self.text_encoders,device=self.device)

            add_time_ids = self._get_add_time_ids(c_t.size()[-2:],[0,0],c_t.size()[-2:],c_t.dtype) 

            add_time_ids = add_time_ids.to(caption_enc_pooled.device).repeat(c_t.size()[0], 1)

            if negative_prompt is not None:
                negative_prompt_embeds, negative_pooled_prompt_embeds = compute_embeddings_for_prompts(['unsharp']*len(prompt), self.tokenizers, self.text_encoders,device=self.device)
                caption_enc = torch.cat([negative_prompt_embeds, caption_enc], dim=0)
                caption_enc_pooled = torch.cat([negative_pooled_prompt_embeds, caption_enc_pooled ], dim=0)
                add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

            added_cond_kwargs = {"text_embeds": caption_enc_pooled, "time_ids": add_time_ids}

        if self.unpaired:
            encoded_control = self.vae_enc(c_t.to(self.device), direction=direction).to(c_t.dtype)
        else:
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        if negative_prompt is not None and self.is_sdxl: 
            encoded_control_conc = torch.cat([encoded_control] * 2)

        else:
            encoded_control_conc = encoded_control

        if deterministic:

            if not self.is_sdxl: 
                model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc).sample
            else:
                noise_pred = self.unet(encoded_control_conc, self.timesteps, encoder_hidden_states=caption_enc,added_cond_kwargs=added_cond_kwargs).sample
                if negative_prompt is not None: 
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    model_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    model_pred = noise_pred
            
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample

            if self.has_skips and not self.unpaired:
                self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks

            if self.unpaired:
                x_out = torch.stack([self.sched.step(model_pred[i], self.timesteps, encoded_control[i], return_dict=True).prev_sample for i in range(B)])
                output_image = self.vae_dec(x_out, direction=direction,has_skips=self.has_skips)

            if not self.unpaired:
                output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            # scale the lora weights based on the r value
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            # combine the input and noise
            unet_input = encoded_control * r + noise_map * (1 - r)
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
            if not self.is_sdxl: 
                model_pred = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
            else:
                noise_pred = self.unet(encoded_control_conc, self.timesteps, encoder_hidden_states=caption_enc,added_cond_kwargs=added_cond_kwargs).sample
                if negative_prompt is not None: 
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    model_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    model_pred = noise_pred
            self.unet.conv_in.r = None
            
            x_denoised = self.sched.step(model_pred, self.timesteps, unet_input, return_dict=True).prev_sample

            if self.has_skips:
                self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            if self.unpaired:
                x_out = torch.stack([sched.step(model_pred[i], timesteps, encoded_control[i], return_dict=True).prev_sample for i in range(B)])
                output_image = vae_dec(x_out, direction=direction,has_skips=self.has_skips)

            if not self.unpaired:
                output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def save_model(self, outf,push_to_hub=False):
        save_all = True # temporary - to be able to chain trainings
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        if self.lora_rank_unet == 0 or save_all:
            sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items()}
        else:
            sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        if not self.unpaired:
    
            if self.lora_rank_vae == 0 or save_all:
                sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items()}
            else:
                sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        else:
           
            if self.lora_rank_vae == 0 or save_all:
                sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items()}
                sd["state_dict_vae_inv"] = {k: v for k, v in self.vae_inv.state_dict().items()}
            else:
                sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
                sd["state_dict_vae_inv"] = {k: v for k, v in self.vae_inv.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)
