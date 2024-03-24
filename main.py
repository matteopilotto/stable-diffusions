from diffusers import StableDiffusionPipeline
import torch
import random
from PIL import Image

import os
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


def init_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    return pipe


def generate_image(prompt, seed=None):
    pipe = init_pipeline()
    generator = torch.Generator(device).manual_seed(seed if seed is not None else random.randint(0, 1e6))
    
    images = pipe(
        prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        num_images_per_prompt=1,
        generator=generator
    ).images

    return images

image = generate_image("Japanese cherry blossoms and Mount Fuji, the style of a maximalist illustration")[0]

image.show()