import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from chatrex.tools.visualize import visualize_chatrex_output
from chatrex.upn import UPNWrapper

processor = AutoProcessor.from_pretrained(
        "checkpoints/chatrex7b",
        trust_remote_code=True,
        device_map="cuda",
    )

print(f"loading chatrex model...")
# load chatrex model
model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/chatrex7b",
    trust_remote_code=True,
    use_safetensors=True,
).to("cuda")

# load upn model
print(f"loading upn model...")
ckpt_path = "checkpoints/upn_checkpoints/upn_large.pth"
model_upn = UPNWrapper(ckpt_path)

if __name__ == '__main__':
    