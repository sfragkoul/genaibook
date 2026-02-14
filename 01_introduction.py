import diffusers
import huggingface_hub
import transformers

from genaibook.core import get_device
device = get_device()
print(f"Using device: {device}")


diffusers.logging.set_verbosity_error()
huggingface_hub.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()


