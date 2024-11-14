import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo import Pix2Pix_Turbo
from image_prep import canny_from_pil
from my_utils.training_utils import str2bool

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
    parser.add_argument('--prompt', type=str, required=True, help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument("--is_sdxl", type=str2bool, default=False)
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold')
    parser.add_argument('--gamma', type=float, default=0.4, help='The sketch interpolation guidance amount')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    args = parser.parse_args()

    # Ensure that either model_name or model_path is provided, but not both
    if args.model_name == '' == args.model_path:
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine the best available device
    device = get_device()

    # Initialize the model
    model = Pix2Pix_Turbo(model_name=args.model_name, pretrained_path=args.model_path, is_sdxl=args.is_sdxl, device=device)
    model.set_eval()

    # Load and preprocess the input image
    input_image = Image.open(args.input_image).convert('RGB')
    input_image = input_image.resize((512, 512), Image.LANCZOS)
    bname = os.path.basename(args.input_image)

    # Translate the image
    with torch.no_grad():
        c_t = F.to_tensor(input_image).unsqueeze(0).to(device)
        output_image = model(c_t, [args.prompt])
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

    # Save the output image
    output_pil.save(os.path.join(args.output_dir, bname))
