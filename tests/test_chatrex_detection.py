import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from chatrex.tools.visualize import visualize_chatrex_output
from chatrex.upn import UPNWrapper

if __name__ == "__main__":
    # load the processor
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
    test_image_path = "tests/images/test_chatrex_detection.jpg"

    # get upn predictions
    fine_grained_proposals = model_upn.inference(
        test_image_path, prompt_type="fine_grained_prompt"
    )
    fine_grained_filtered_proposals = model_upn.filter(
        fine_grained_proposals, min_score=0.3, nms_value=0.8
    )

    inputs = processor.process(
        image=Image.open(test_image_path),
        question="Please detect person; pigeon in this image. Answer the question with object indexes.",
        bbox=fine_grained_filtered_proposals["original_xyxy_boxes"][
            0
        ],  # box in xyxy format
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
        fine_grained_filtered_proposals["original_xyxy_boxes"][0],
        prediction,
        font_size=15,
        draw_width=5,
    )
    vis_image.save("tests/test_chatrex_detection.jpeg")
    print(f"prediction is saved at tests/test_chatrex_detection.jpeg")
