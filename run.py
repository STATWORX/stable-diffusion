import torch
import argparse
from diffusers import StableDiffusionPipeline
from PIL import Image

TOKEN_PATH = './token'


def image_grid(imgs, rows, cols):
    """ Grid of images"""
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def dummy(images, **kwargs):
    """Dummy function that just returns the input (avoids NSFW filter)"""
    return images, False


def run_diffusion(pipe: StableDiffusionPipeline,
                  prompt: str,
                  n_images: int = 1,
                  steps: int = 100,
                  height: int = 512,
                  width: int = 512):
    """
    Wrapper that executes a Diffusion pipeline
    :param pipe: Pipeline object
    :param prompt: A text prompt used in the pipeline
    :param n_images: Number of images to generate
    :param steps: Number of diffusion steps
    :param height: Image Height
    :param width: Image Width
    """
    return pipe([prompt] * n_images, num_inference_steps=steps, height=height, width=width)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Stable Diffusion parameters')
    parser.add_argument('--steps', type=int, nargs=1, default=100)
    parser.add_argument('--n_images', type=int, nargs=1, default=1)
    parser.add_argument('--dims', type=int, nargs=1)

    args = parser.parse_args()

    # Define device (either GPU, M1/2, or CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Define image dimensions (squared)
    if args.dims is None:
        dims_ls = (512, 512)
    else:
        dims_ls = (args.dims, args.dims)

    # Read token from file
    with open(TOKEN_PATH, 'r') as f:
        token = f.read()

    # Model ID on huggingface hub
    model_id = 'CompVis/stable-diffusion-v1-4'

    # Load the model and transfer it to the correct device
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=token)
    pipe.safety_checker = dummy
    pipe = pipe.to(device)

    while True:

        prompt = input("Please enter prompt: ")
        print(f'Starting the diffusion process for: {prompt} @{args.steps} and resolution {dims_ls}')
        result = run_diffusion(pipe,
                               prompt,
                               n_images=args.n_images,
                               steps=args.steps,
                               height=dims_ls[0],
                               width=dims_ls[1])

        images_as_grid = image_grid(result.images, 1, args.n_images * 1)
        images_as_grid.show()
