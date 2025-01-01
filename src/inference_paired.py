import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo import Pix2Pix_Turbo
from image_prep import canny_from_pil
from my_utils.training_utils import str2bool
import time

Image.MAX_IMAGE_PIXELS = None

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

def process_batch(images, model, device, prompt):
    """Process a batch of images."""
    c_t = torch.stack([F.to_tensor(img).to(device) for img in images])
    with torch.no_grad():
        output_images = model(c_t, prompt = [prompt] * len(images))
        output_pils = [
            transforms.ToPILImage()(output.cpu() * 0.5 + 0.5) for output in output_images
        ]
    return output_pils

def process_single_image(input_path, model, device, prompt, output_dir, resize_dim):
    start_time = time.time()
    print(f"prompt: {prompt}")
    """Process a single image."""
    image = Image.open(input_path).convert('RGB').resize((resize_dim, resize_dim), Image.LANCZOS)
    with torch.no_grad():
        c_t = F.to_tensor(image).unsqueeze(0).to(device)
        output_image = model(c_t, prompt = [prompt])
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
    bname = os.path.basename(input_path)
    output_pil.save(os.path.join(output_dir, bname))
    end_time = time.time()

    # Calculate and print execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.5f} seconds")
    print(f"Processed single image: {bname}")

def process_folder(input_folder, model, device, prompt, output_dir, batch_size, resize_dim):
    """Process all images in a folder in batches."""
    image_paths = [
        os.path.join(input_folder, fname)
        for fname in os.listdir(input_folder)
        if fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))
    ]

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [
            Image.open(p).convert('RGB').resize((resize_dim, resize_dim), Image.LANCZOS) for p in batch_paths
        ]
        output_images = process_batch(batch_images, model, device, prompt)

        # Save the output images
        for img, path in zip(output_images, batch_paths):
            bname = os.path.basename(path)
            img.save(os.path.join(output_dir, bname))
            print(f"Processed image: {bname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_image', type=str, help='path to the input image')
    group.add_argument('--input_folder', type=str, help='path to the folder containing input images')
    parser.add_argument('--prompt', type=str, required=True, help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument("--model_type", type=str, default="sd")
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for processing images in folders')
    parser.add_argument('--resize_dim', type=int, default=1024, help='resize dimension for input images (default: 1024)')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold')
    parser.add_argument('--gamma', type=float, default=0.4, help='The sketch interpolation guidance amount')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--original_unet', action='store_true', default=False, help='use the original untrained UNet of the base model')
    parser.add_argument('--original_vae', action='store_true', default=False, help='use the original untrained VAE of the base model')

    args = parser.parse_args()

    # Ensure that either model_name or model_path is provided, but not both
    if args.model_name == '' == args.model_path:
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine the best available device
    device = get_device()

    # Initialize the model
    model = Pix2Pix_Turbo(model_name=args.model_name, pretrained_path=args.model_path, model_type=args.model_type, device=device,original_unet=args.original_unet, original_vae=args.original_vae)
    model.set_eval()

    if args.input_image:
        # Process a single image
        process_single_image(args.input_image, model, device, args.prompt, args.output_dir, args.resize_dim)
    elif args.input_folder:
        # Process images in the folder
        process_folder(args.input_folder, model, device, args.prompt, args.output_dir, args.batch_size, args.resize_dim)

    print("Processing complete. Outputs saved to:", args.output_dir)
