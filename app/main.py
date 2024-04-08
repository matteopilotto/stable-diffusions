from diffusers import StableDiffusionPipeline
import torch
import random
from PIL import Image

import os
import argparse
import math
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    default="Japanese cherry blossoms and Mount Fuji, the style of a maximalist illustration",
    type=str
)

parser.add_argument(
    "--inference_steps",
    default=20,
    type=int
)

parser.add_argument(
    "--guidance_scale",
    default=7.5,
    type=float
)

parser.add_argument(
    "--seed",
    default=None,
    type=int
)

args = parser.parse_args()

PROMPT = args.prompt
INFERENCE_STEPS = args.inference_steps
GUIDANCE_SCALE=args.guidance_scale
SEED = args.seed if args.seed is not None else random.randint(0, 1e6)


def init_pipeline():
    # model_id = "runwayml/stable-diffusion-v1-5"
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    return pipe


def generate_image(prompt, seed=None):
    pipe = init_pipeline()
    generator = torch.Generator(device).manual_seed(seed)
    
    images = pipe(
        prompt,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_images_per_prompt=1,
        generator=generator
    ).images

    return images


image = generate_image(PROMPT, seed=SEED)[0]

target_dir = "./saved_mages"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

creation_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
image_name = f"{target_dir}/image-{creation_time}.png"

print(f"[INFO] Prompt: {PROMPT}")
print(f"[INFO] Random seed: {SEED}")
image.save(image_name)
print(f"[INFO] Image save to {image_name}")





