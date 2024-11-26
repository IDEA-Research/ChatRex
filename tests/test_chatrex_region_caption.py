import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from chatrex.tools.visualize import visualize_chatrex_output
from chatrex.upn import UPNWrapper

if __name__ == "__main__":
    # load the processor
    processor = AutoProcessor.from_pretrained(
        "IDEA-Research/ChatRex-7B",
        trust_remote_code=True,
        device_map="cuda",
    )

    print(f"loading chatrex model...")
    # load chatrex model
    model = AutoModelForCausalLM.from_pretrained(
        "IDEA-Research/ChatRex-7B",
        trust_remote_code=True,
        use_safetensors=True,
    ).to("cuda")

    test_image_path = "tests/images/test_chatrex_install.jpg"

    inputs = processor.process(
        image=Image.open(test_image_path),
        question="Can you provide a one sentence description of <obj0> in the image? Answer the question with a one sentence description.",
        bbox=[[73.88417, 56.62228, 227.69223, 216.34338]],
    )

    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # perform inference
    gen_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=(
            processor.tokenizer.pad_token_id
            if processor.tokenizer.pad_token_id is not None
            else processor.tokenizer.eos_token_id
        ),
    )
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        prediction = model.generate(
            inputs, gen_config=gen_config, tokenizer=processor.tokenizer
        )
    print(f"prediction:", prediction)

    # visualize the prediction
    vis_image = visualize_chatrex_output(
        Image.open(test_image_path),
        [[73.88417, 56.62228, 227.69223, 216.34338]],
        prediction,
        font_size=15,
        draw_width=5,
    )
    vis_image.save("tests/test_chatrex_region_caption.jpeg")
    print(f"prediction is saved at tests/test_chatrex_region_caption.jpeg")
