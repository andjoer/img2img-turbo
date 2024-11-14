import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd

from my_utils.training_utils import encode_prompt
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from transformers import AutoTokenizer, PretrainedConfig

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
    def __init__(self, model_name= "stabilityai/sd-turbo",pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", is_sdxl = False,lora_rank_unet=8, lora_rank_vae=4,device="cuda"):
        super().__init__()
        self.is_sdxl = is_sdxl
        self.model_name = model_name
        self.device = device
        if not is_sdxl:
            self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").to(device)
            self.sched = make_1step_sched(model_name, device)
            self.device = device
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        else:
            print('###using stable diffusion xl')
            tokenizer_1 = AutoTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer",
            use_fast=False,
            )
            tokenizer_2 = AutoTokenizer.from_pretrained(
                model_name,
                subfolder="tokenizer_2",
                use_fast=False,
            )
            text_encoder_cls_1 = import_model_class_from_model_name_or_path(model_name,revision=None)
            text_encoder_cls_2 = import_model_class_from_model_name_or_path(
                model_name,  subfolder="text_encoder_2",revision=None
            )

            text_encoder_1 = text_encoder_cls_1.from_pretrained(
                model_name, subfolder="text_encoder"
            )
            text_encoder_2 = text_encoder_cls_2.from_pretrained(
                model_name, subfolder="text_encoder_2"
            )

            # We ALWAYS pre-compute the additional condition embeddings needed for SDXL
            # UNet as the model is already big and it uses two text encoders.
            text_encoder_1.to(device)
            text_encoder_2.to(device)
            self.tokenizers = [tokenizer_1, tokenizer_2]
            self.text_encoders = [text_encoder_1, text_encoder_2]

            # Freeze vae and text_encoders
            text_encoder_1.requires_grad_(False)
            text_encoder_2.requires_grad_(False)
            self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
            self.sched = make_1step_sched(model_name, device)
            self.device = device
            vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")

        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        #add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        vae.decoder.ignore_skip = False

        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

        if pretrained_name == "edge_to_image":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "edge_to_image_loras.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name == "sketch_to_image_stochastic":
            # download from url
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "sketch_to_image_stochastic_lora.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            convin_pretrained = copy.deepcopy(unet.conv_in)
            unet.conv_in = TwinConv(convin_pretrained, unet.conv_in)
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            self.lora_rank_unet = sd["rank_unet"]
            self.lora_rank_vae = sd["rank_vae"]
            self.target_modules_vae = sd["vae_lora_target_modules"]
            self.target_modules_unet = sd["unet_lora_target_modules"]
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet


        #unet.enable_xformers_memory_efficient_attention()
        unet.to(self.device)
        vae.to(self.device)
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device=self.device).long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)


    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(tuple(original_size) + tuple(crops_coords_top_left) + tuple(target_size))

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids



    def forward(self, c_t, prompt=None, prompt_tokens=None, caption_enc=None, caption_enc_pooled=None, deterministic=True, r=1.0, noise_map=None,tokenizers=None,text_encoders=None,negative_prompt=None, guidance_scale=0):
        # either the prompt or the prompt_tokens should be provided
        #assert ((prompt is None) != (prompt_tokens is None)), "Either prompt or prompt_tokens should be provided"

        #negative_prompt = "unsharp"

        if not self.is_sdxl:
            if prompt is not None:
                # encode the text prompt
                caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                                padding="max_length", truncation=True, return_tensors="pt").input_ids.to(self.device)
                caption_enc = self.text_encoder(caption_tokens)[0]

                caption_enc_pooled = caption_enc[0]
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
        
        if deterministic:

            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

            if negative_prompt is not None: 
                encoded_control_conc = torch.cat([encoded_control] * 2)

            else:
                encoded_control_conc = encoded_control
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
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
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
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items()}#{k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        #print([k for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k])
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items()}#{k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)
