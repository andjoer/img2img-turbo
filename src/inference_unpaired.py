import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import build_transform

def get_device():
    """Prioritize CUDA, then MPS (Apple Silicon), and fallback to CPU."""
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
        return 'cuda'
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return 'mps'
    else:
        print("Using CPU")
        return 'cpu'
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=False, help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_name', type=str, default=None, help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--model_type', type=str, default="sd", help='type of the used model')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method')
    parser.add_argument('--direction', type=str, default=None, help='the direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name is None != args.model_path is None:
        raise ValueError('Either model_name or model_path should be provided')

    if args.model_path is not None and args.prompt is None:
        raise ValueError('prompt is required when loading a custom model_path.')


    device = get_device()
    # initialize the model
    model = Pix2Pix_Turbo(model_name=args.model_name, pretrained_path=args.model_path, model_type=args.model_type, device=device,unpaired=True)
    model.set_eval()

    if "cuda" in device:
        model.unet.enable_xformers_memory_efficient_attention()

    T_val = build_transform(args.image_prep)

    input_image = Image.open(args.input_image).convert('RGB')
    caption = model.prompt_encoder(args.prompt)
    # translate the image
    with torch.no_grad():
        input_img = T_val(input_image)
        x_t = transforms.ToTensor()(input_img)
        x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).to(device)
        output = model.forward_cycle(x_t, direction=args.direction, timesteps=model.timesteps, text_emb=caption)

    output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
    output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

    # save the output image
    bname = os.path.basename(args.input_image)
    os.makedirs(args.output_dir, exist_ok=True)
    output_pil.save(os.path.join(args.output_dir, bname))
