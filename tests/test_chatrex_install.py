from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

if __name__ == "__main__":
    # load the processor
    processor = AutoProcessor.from_pretrained(
        "IDEA-Research/ChatRex-7B",
        trust_remote_code=True,
        device_map="cuda",
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        "IDEA-Research/ChatRex-7B",
        trust_remote_code=True,
        use_safetensors=True,
    ).to("cuda")

    inputs = processor.process(
        image=Image.open("tests/images/test_chatrex_install.jpg"),
        question="Can you provide me with a brief description of <obj0>?",
        bbox=[[73.88417, 56.62228, 227.69223, 216.34338]],  # box in xyxy format
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
    prediction = model.generate(
        inputs, gen_config=gen_config, tokenizer=processor.tokenizer
    )
    print(f"prediction:", prediction)
