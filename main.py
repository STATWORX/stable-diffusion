import torch
import time
import argparse
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image
from datetime import datetime

TOKEN_PATH = './token'
IMG_PATH = './img'
RESULTS_PATH = './results'


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
                  steps: int = 50,
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

    start_time = time.time()
    result = pipe([prompt] * n_images, num_inference_steps=steps, height=height, width=width)
    end_time = time.time()

    total_time = end_time - start_time
    time_per_step = total_time / steps

    return result, total_time, time_per_step


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulation parameters')
    parser.add_argument('--runs', type=int, nargs=1, default=5)
    parser.add_argument('--steps', type=int, nargs=1, default=50)
    parser.add_argument('--dims', type=int, nargs=1)
    parser.add_argument('--device', type=str)
    parser.add_argument('--prompt', type=str)

    args = parser.parse_args()

    # Define device (either GPU, M1/2, or CPU)
    if args.device is None or args.device not in ['cuda', 'cpu', 'mps']:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f'No device specified, using default device {device}')
    else:
        device = args.device

    # Define image dimensions (squared)
    if args.dims is None:
        dims_ls = [(512, 512), (768, 768)]
    else:
        dims_ls = [(args.dims, args.dims)]

    # Define prompt
    if args.prompt is None:
        prompt = "A photo of an astronaut riding a horse in the style of H.P. Lovecraft trending on artstation"
    else:
        prompt = args.prompt

    # Read token from file
    with open(TOKEN_PATH, 'r') as f:
        token = f.read()

    # Model ID on huggingface hub
    model_id = 'CompVis/stable-diffusion-v1-4'

    # Load the model and transfer it to the correct device
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=token)
    pipe.safety_checker = dummy
    pipe = pipe.to(device)

    results_df = pd.merge(
        pd.DataFrame({'steps': args.steps}, index=(0, )),
        pd.DataFrame({'dims': dims_ls}),
        how='cross'
    )
    results_df['avg_total_time'] = np.nan
    results_df['avg_time_step'] = np.nan
    results_df['device'] = device

    for idx, params in results_df.iterrows():

        print(f'Running parameters {idx + 1}/{results_df.shape[0]}: step={params["steps"]} with {params["dims"]}')

        image_ls = []
        total_time_ls = []
        time_per_step_ls = []

        for run in np.arange(args.runs):

            result, time_total, time_step = run_diffusion(pipe, prompt,
                                                          steps=params['steps'],
                                                          height=params['dims'][0],
                                                          width=params['dims'][1])

            image_ls.extend(result.images)
            total_time_ls.append(time_total)
            time_per_step_ls.append(time_step)

        images_as_grid = image_grid(image_ls, 1, args.runs * 1)
        path = f'{IMG_PATH}/{prompt.replace(" ", "_")}_{params["steps"]}_{params["dims"][0]}x{params["dims"][1]}_.png'
        images_as_grid.save(path)

        images_as_grid.show()

        results_df.loc[idx, 'avg_total_time'] = np.mean(total_time_ls)
        results_df.loc[idx, 'avg_time_step'] = np.mean(time_per_step_ls)

    results_df.to_pickle(f'{RESULTS_PATH}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl')
