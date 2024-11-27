import gradio as gr
import numpy as np
import torch
from gradio_image_prompter import ImagePrompter
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

import chatrex.tools.prompt_templates as prompt_templates
from chatrex.tools.visualize import visualize_chatrex_output
from chatrex.upn import UPNWrapper

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

gen_config = GenerationConfig(
    max_new_tokens=4096,
    do_sample=False,
    eos_token_id=processor.tokenizer.eos_token_id,
    pad_token_id=(
        processor.tokenizer.pad_token_id
        if processor.tokenizer.pad_token_id is not None
        else processor.tokenizer.eos_token_id
    ),
)

# load upn model
print(f"loading upn model...")
ckpt_path = "checkpoints/upn_checkpoints/upn_large.pth"
model_upn = UPNWrapper(ckpt_path)


class ColorGenerator:

    def __init__(self, color_type) -> None:
        self.color_type = color_type

        if color_type == "same":
            self.color = tuple((np.random.randint(0, 127, size=3) + 128).tolist())
        elif color_type == "text":
            np.random.seed(3396)
            self.num_colors = 300
            self.colors = np.random.randint(0, 127, size=(self.num_colors, 3)) + 128
        else:
            raise ValueError

    def get_color(self, text):
        if self.color_type == "same":
            return self.color

        if self.color_type == "text":
            text_hash = hash(text)
            index = text_hash % self.num_colors
            color = tuple(self.colors[index])
            return color

        raise ValueError


TASK2PROMPT_MAPPER = {
    "RegionCap_(Category)": prompt_templates.REGION_CAPTION_SINGLE_REGION_FREE_FORMAT_CATEGORY_NAME_STAGE2[
        0
    ],
    "RegionCap_(ShortPhrase)": prompt_templates.REGION_CAPTION_SINGLE_REGION_SHORT_PHRASE_STAGE2[
        0
    ],
    "RegionCap_(Breif)": prompt_templates.REGION_CAPTION_SINGLE_REGION_BREIFLY_STAGE2[
        0
    ],
    "RegionCap_(OneSentence)": prompt_templates.REGION_CAPTION_ONE_SENTENCE_STAGE2[0],
    "Grounding": prompt_templates.GROUNDING_SINGLE_REGION_STAGE2[0],
    "Grounded_ShortImageCap": prompt_templates.BREIF_CAPTION_WITH_GROUDING_STAGE2[0],
    "Grounded_LongImageCap": prompt_templates.DETAILED_CAPTION_WITH_GROUDING_STAGE2[0],
}


def question_creater(
    object_ids_or_cates: str, task: str, caption_all_region, boxes_length
):
    prompts = [prompt.strip() for prompt in object_ids_or_cates.split(",")]
    if "RegionCap" in task and caption_all_region:
        prompts = list(range(boxes_length))
        prompts = "; ".join([f"<obj{prompt}>" for prompt in prompts])
        question = TASK2PROMPT_MAPPER[task].replace("[OBJ]", prompts)
    else:
        prompts = "; ".join(prompts)
        question = TASK2PROMPT_MAPPER[task].replace("[OBJ]", prompts)
    return question


def visualize(
    image_pil: Image,
    boxes,
    scores,
    labels=None,
    filter_score=-1,
    topN=900,
    font_size=15,
    draw_width: int = 6,
    draw_index: bool = True,
    random_color: bool = False,
) -> Image:
    """Plot bounding boxes and labels on an image.

    Args:
        image_pil (PIL.Image): The input image as a PIL Image object.
        model_targetoutput (Dict[str, Union[torch.Tensor, List[torch.Tensor]]]): The target dictionary containing
            the bounding boxes and labels. The keys are:
                - boxes (List[int]): A list of bounding boxes in shape (N, 4), [x1, y1, x2, y2] format.
                - scores (List[float]): A list of scores for each bounding box. shape (N)
                - labels (List[str]): A list of string labels for each bounding box. shape (N)
        return_point (bool): Draw center point instead of bounding box. Defaults to False.
        draw_width (float): The width of the drawn bounding box or point. Defaults to 1.0.
        random_color (bool): Use random color for each category. Defaults to True.
        overwrite_color (Dict): Overwrite color for each category. Defaults to None.
        agnostic_random_color (bool): If True, we will use random color for all boxes.
        draw_score (bool): Draw score on the image. Defaults to False.

    Returns:
        Union[PIL.Image, PIL.Image]: A tuple containing the input image and ploted image.
    """
    # Get the bounding boxes and labels from the target dictionary
    font_path = "chatrex/tools/Tahoma.ttf"
    font = ImageFont.truetype(font_path, font_size)
    # Create a PIL ImageDraw object to draw on the input image
    draw = ImageDraw.Draw(image_pil)
    boxes = boxes[:topN]
    scores = scores[:topN]
    # Draw boxes and masks for each box and label in the target dictionary
    box_idx = 0
    color_generaor = ColorGenerator("text")
    if labels is None:
        labels = [""] * len(boxes)
    for box, score, label in zip(boxes, scores, labels):
        if score < filter_score:
            continue
        # Extract the box coordinates
        x0, y0, x1, y1 = box
        # rescale the box coordinates to the input image size
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        if draw_index:
            text = str(box_idx) + f" {label}"
        else:
            text = label
        max_words_per_line = 10
        words = text.split()
        lines = []
        line = ""
        for word in words:
            if len(line.split()) < max_words_per_line:
                line += word + " "
            else:
                lines.append(line)
                line = word + " "
        lines.append(line)
        text = "\n".join(lines)

        if random_color:
            color = tuple(np.random.randint(0, 255, size=3).tolist())
        else:
            color = color_generaor.get_color(text)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=draw_width)

        bbox = draw.textbbox((x0, y0), text, font)
        box_h = bbox[3] - bbox[1]
        box_w = bbox[2] - bbox[0]

        y0_text = y0 - box_h - (draw_width * 2)
        y1_text = y0 + draw_width
        box_idx += 1
        if y0_text < 0:
            y0_text = 0
            y1_text = y0 + 2 * draw_width + box_h
        draw.rectangle(
            [x0, y0_text, bbox[2] + draw_width * 2, y1_text],
            fill=color,
        )
        draw.text(
            (x0 + draw_width, y0_text),
            str(text),
            fill="black",
            font=font,
        )
    return image_pil


def parse_visual_prompt(points):
    boxes = []
    pos_points = []
    neg_points = []
    for point in points:
        if point[2] == 2 and point[-1] == 3:
            x1, y1, _, x2, y2, _ = point
            boxes.append([x1, y1, x2, y2])
        elif point[2] == 1 and point[-1] == 4:
            x, y, _, _, _, _ = point
            pos_points.append([x, y])
        elif point[2] == 0 and point[-1] == 4:
            x, y, _, _, _, _ = point
            neg_points.append([x, y])
    return boxes, pos_points, neg_points


def display_vp(target_image, interactive_input, font_size, draw_width):
    target_image = Image.fromarray(target_image)
    ref_visual_prompt = interactive_input["points"]
    boxes, _, _ = parse_visual_prompt(ref_visual_prompt)
    fake_scores = np.ones(len(boxes))
    vis_result = visualize(
        target_image.copy(),
        boxes,
        fake_scores,
        font_size=font_size,
        draw_width=draw_width,
        draw_index=True,
        random_color=True,
    )
    return vis_result


def object_proposal(
    input_image,
    visual_threshold,
    nms_value,
    use_fine_grained,
    use_coarse_grained,
    font_size,
    draw_width,
):
    input_image = Image.fromarray(input_image)
    if use_fine_grained and use_coarse_grained:
        raise gr.Error("Please select only one prompt type")

    if not use_coarse_grained and not use_fine_grained:
        raise gr.Error("Please select a prompt type")

    if use_fine_grained:
        prompt_type = "fine_grained_prompt"
    if use_coarse_grained:
        prompt_type = "coarse_grained_prompt"

    proposals = model_upn.inference(input_image, prompt_type=prompt_type)
    filtered_proposals = model_upn.filter(
        proposals, min_score=float(visual_threshold), nms_value=float(nms_value)
    )
    # visualize
    vis_result = visualize(
        input_image,
        filtered_proposals["original_xyxy_boxes"][0],
        filtered_proposals["scores"][0],
        font_size=font_size,
        draw_width=draw_width,
        draw_index=True,
        random_color=True,
    )
    return vis_result


def infer_chatrex(
    target_image,
    interactive_input,
    input_raw_text,
    template_input_object_ids,
    template_task_select,
    visual_threshold,
    nms_value,
    use_fine_grained,
    use_coarse_grained,
    caption_all_region,
    font_size,
    draw_width,
):
    target_image = Image.fromarray(target_image)

    if input_raw_text != "":
        question = input_raw_text
    else:
        question = None

    if use_fine_grained:
        prompt_type = "fine_grained_prompt"
    elif use_coarse_grained:
        prompt_type = "coarse_grained_prompt"
    elif interactive_input is not None:
        prompt_type = "interactive"
    else:
        raise gr.Error(
            "Please select a prompt type, or provide a visual prompt, for box input"
        )

    object_ids_or_cates = template_input_object_ids
    task = template_task_select

    if "grained" in prompt_type:
        proposals = model_upn.inference(target_image, prompt_type=prompt_type)
        filtered_proposals = model_upn.filter(
            proposals, min_score=float(visual_threshold), nms_value=float(nms_value)
        )

        question = (
            question_creater(
                object_ids_or_cates,
                task,
                caption_all_region,
                len(filtered_proposals["original_xyxy_boxes"][0]),
            )
            if question is None
            else question
        )

        inputs = processor.process(
            image=target_image,
            question=question,
            bbox=filtered_proposals["original_xyxy_boxes"][0],  # box in xyxy format
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            answer = model.generate(
                inputs, gen_config=gen_config, tokenizer=processor.tokenizer
            )

        vis_result = visualize_chatrex_output(
            target_image,
            filtered_proposals["original_xyxy_boxes"][0],
            answer,
            font_size=font_size,
            draw_width=draw_width,
        )

    else:
        # for interactive prompt
        ref_visual_prompt = interactive_input["points"]
        pred_boxes, _, _ = parse_visual_prompt(ref_visual_prompt)
        # prepare for chatrex
        question = (
            question_creater(
                object_ids_or_cates, task, caption_all_region, len(pred_boxes)
            )
            if question is None
            else question
        )
        inputs = processor.process(
            image=target_image,
            question=question,
            bbox=pred_boxes,  # box in xyxy format
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            answer = model.generate(
                inputs, gen_config=gen_config, tokenizer=processor.tokenizer
            )

        vis_result = visualize_chatrex_output(
            target_image,
            pred_boxes,
            answer,
            font_size=font_size,
            draw_width=draw_width,
        )
    return vis_result, answer, question


if __name__ == "__main__":
    interactive_1 = ImagePrompter(label="1", width=500, scale=1)
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label="Input Image", width=500)
                    interactive = gr.TabbedInterface(
                        [interactive_1], ["Interactive Visual Prompt"]
                    )
                with gr.Row():
                    middle_output_image = gr.Image(
                        label="Proposal & Visual Prompt Vis", width=500
                    )
                    output_image = gr.Image(label="Output Image", width=500)
                with gr.Row():
                    llm_raw_output = gr.Textbox(
                        label="Answer from LLM", placeholder="Answer from LLM", lines=10
                    )
            with gr.Column():
                with gr.Row():
                    use_fine_grained_proposal = gr.Checkbox(
                        label="Fine Grained Proposal", value=True
                    )
                    use_coarse_grained_proposal = gr.Checkbox(
                        label="Coarse Grained Proposal"
                    )
                with gr.Row():
                    visual_threshold = gr.Slider(
                        label="UPN Threshold",
                        value=0.3,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                    )
                    nms_value = gr.Textbox(label="NMS", value=-1)
                with gr.Row():
                    font_size = gr.Slider(
                        label="font_size",
                        value=15,
                        minimum=1,
                        maximum=100,
                        step=1,
                    )
                    draw_width = gr.Slider(
                        label="draw_width",
                        value=6,
                        minimum=1,
                        maximum=100,
                        step=1,
                    )
                with gr.Row():
                    display_visual_prompt = gr.Button("Display Visual Prompt")
                    run_object_proposal = gr.Button("Display UPN Proposal")
                with gr.Row():
                    input_raw_text = gr.Textbox(
                        label="Raw Question Input",
                        placeholder="Please ask me any question",
                        lines=4,
                    )
                with gr.Row():
                    caption_all_region = gr.Checkbox(label="Caption all regions")
                    template_input_object_ids = gr.Textbox(
                        label="object ids & cates",
                        placeholder="<obj1>,<obj2> or dog,cat",
                        lines=2,
                    )
                with gr.Row():
                    template_task_select = gr.Dropdown(
                        label="Pre-defined Question Templates",
                        choices=[
                            "RegionCap_(Category)",
                            "RegionCap_(ShortPhrase)",
                            "RegionCap_(Breif)",
                            "RegionCap_(OneSentence)",
                            "Grounding",
                            "Grounded_ShortImageCap",
                            "Grounded_LongImageCap",
                        ],
                    )
                with gr.Row():
                    template_output = gr.Textbox(
                        label="Tempalte Output",
                        placeholder="",
                        lines=6,
                    )
                with gr.Row():
                    run_chatrex = gr.Button("Run ChatRex")

        run_object_proposal.click(
            fn=object_proposal,
            inputs=[
                input_image,
                visual_threshold,
                nms_value,
                use_fine_grained_proposal,
                use_coarse_grained_proposal,
                font_size,
                draw_width,
            ],
            outputs=[middle_output_image],
        )

        display_visual_prompt.click(
            fn=display_vp,
            inputs=[input_image, interactive_1, font_size, draw_width],
            outputs=[middle_output_image],
        )

        run_chatrex.click(
            fn=infer_chatrex,
            inputs=[
                input_image,
                interactive_1,
                input_raw_text,
                template_input_object_ids,
                template_task_select,
                visual_threshold,
                nms_value,
                use_fine_grained_proposal,
                use_coarse_grained_proposal,
                caption_all_region,
                font_size,
                draw_width,
            ],
            outputs=[output_image, llm_raw_output, template_output],
        )

    demo.launch(debug=True)
