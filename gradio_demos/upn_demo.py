import argparse

import gradio as gr
from PIL import Image

from chatrex.tools.visualize import plot_boxes_to_image
from chatrex.upn import UPNWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Test Llava-region Model")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/upn_checkpoints/upn_large.pth",
        help="the checkpoint model path",
    )
    args = parser.parse_args()
    return args


def get_proposal(
    image,
    use_fine_grained,
    use_coarse_grained,
    score_threshold,
    nms_value,
    draw_width,
    draw_points,
    return_score,
):
    image = Image.fromarray(image)
    # if use_fine_grained and use_coarse_grained:
    #     raise gr.Error("Please select only one prompt type")

    # if not use_coarse_grained and not use_fine_grained:
    #     raise gr.Error("Please select a prompt type")

    if use_fine_grained:
        prompt_type = "fine_grained_prompt"
    else:
        prompt_type = "coarse_grained_prompt"
    proposals = model.inference(image, prompt_type=prompt_type)
    filtered_proposals = model.filter(
        proposals, min_score=float(score_threshold), nms_value=float(nms_value)
    )
    visualized_image, _ = plot_boxes_to_image(
        image.copy(),
        filtered_proposals["original_xyxy_boxes"],
        filtered_proposals["scores"],
        point_width=draw_width,
        return_point=draw_points,
        return_score=return_score,
    )
    return visualized_image


args = parse_args()

model = UPNWrapper(args.ckpt_path)

if __name__ == "__main__":
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(label="Input Image", width=400)
                output_image = gr.Image(label="Output Image", width=400)
        with gr.Column():
            with gr.Row():
                use_fine_grained = gr.Checkbox(label="Fine Grained Prompt")
                use_coarse_grained = gr.Checkbox(label="Coarse Grained Prompt")
                score_threshold = gr.Slider(
                    label="Score Threshold",
                    value=0.3,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                )
                nms_value = gr.Textbox(label="NMS", value=-1)
                draw_width = gr.Slider(
                    label="draw_width",
                    value=10,
                    minimum=1,
                    maximum=100,
                    step=1,
                )
                return_score = gr.Checkbox(label="Return Score")
                draw_points = gr.Checkbox(label="Draw Points")
            with gr.Row():
                run_button = gr.Button("Run UPN")
        run_button.click(
            fn=get_proposal,
            inputs=[
                input_image,
                use_fine_grained,
                use_coarse_grained,
                score_threshold,
                nms_value,
                draw_width,
                draw_points,
                return_score,
            ],
            outputs=output_image,
        )
    demo.launch(debug=True)
