from diffusers import StableDiffusionXLPipeline
import torch

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_instruct_pix2pix import (
    StableDiffusionXLInstructPix2PixPipeline,
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("mps")

prompt = "chocolate circle with bisque background"
image = pipe(prompt=prompt).images[0]



'''from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_instruct_pix2pix import (
    StableDiffusionXLInstructPix2PixPipeline,
)
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, PretrainedConfig
import numpy as np 

import torch 

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder
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


generator = torch.Generator(device="mps").manual_seed(42)


unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet"
    )
vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae" 
    )

# Load scheduler, tokenizer and models.
tokenizer_1 = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="tokenizer",
    use_fast=False,
)
tokenizer_2 = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="tokenizer_2",
    use_fast=False,
)
text_encoder_cls_1 = import_model_class_from_model_name_or_path("stabilityai/stable-diffusion-xl-base-1.0")
text_encoder_cls_2 = import_model_class_from_model_name_or_path(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2"
)

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
text_encoder_1 = text_encoder_cls_1.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder"
)
text_encoder_2 = text_encoder_cls_2.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2"
)

# We ALWAYS pre-compute the additional condition embeddings needed for SDXL
# UNet as the model is already big and it uses two text encoders.
#text_encoder_1.to("mps")
#text_encoder_2.to("mps")
tokenizers = [tokenizer_1, tokenizer_2]
text_encoders = [text_encoder_1, text_encoder_2]


pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    unet=unet,
    text_encoder=text_encoder_1,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer_1,
    tokenizer_2=tokenizer_2,
    vae=vae,
    torch_dtype = torch.float
    )

pipeline = pipeline.to("mps")
pipeline.set_progress_bar_config(disable=True)


prompt = "test prompt"

image = np.zeros([512,512,3])
prediction = pipeline(
                prompt,
                image=image,
                num_inference_steps=20,
                image_guidance_scale=1.5,
                guidance_scale=7,
                generator=generator,
            ).images[0]'''