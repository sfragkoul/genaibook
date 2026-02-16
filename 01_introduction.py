import diffusers
import huggingface_hub
import transformers

# check device-------------------------------------------------------------------------------
from genaibook.core import get_device
device = get_device()
print(f"Using device: {device}")

# Generating Images-------------------------------------------------------------------------
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
#torch_dtype=torch.float16,
variant="fp16",
).to(device)

prompt = "a photograph of an astronaut riding a horse"
pipe(prompt).images[0]

# Generating Text---------------------------------------------------------------------------
from transformers import pipeline

classifier = pipeline("text-classification", device=device)
classifier("This movie is disgustingly good!")


from transformers import set_seed
# Setting the seed ensures we get the same results every time we run this code
set_seed(10)
generator = pipeline("text-generation", device=device)
prompt = "It was a dark and stormy"
generator(prompt)[0]["generated_text"]