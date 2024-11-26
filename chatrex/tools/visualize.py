import re
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


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


def plot_boxes_to_image(
    image_pil: Image,
    boxes: List[List[float]],
    scores: List[float],
    return_point: bool = False,
    point_width: float = 10.0,
    return_score=True,
) -> Image:
    """Plot bounding boxes and labels on an image.

    Args:
        image_pil (PIL.Image): The input image as a PIL Image object.
        boxes: A list of bounding boxes in shape (N, 4), in (x1, y1, x2, y2) format.
        scores: A list of scores for each bounding box.
        return_point (bool): Draw center point instead of bounding box. Defaults to False.

    Returns:
        Union[PIL.Image, PIL.Image]: A tuple containing the input image and ploted image.
    """
    # Create a PIL ImageDraw object to draw on the input image
    draw = ImageDraw.Draw(image_pil)
    # Create a new binary mask image with the same size as the input image
    mask = Image.new("L", image_pil.size, 0)
    # Create a PIL ImageDraw object to draw on the mask image
    mask_draw = ImageDraw.Draw(mask)

    # Draw boxes and masks for each box and label in the target dictionary
    for box, score in zip(boxes, scores):
        # Convert the box coordinates from 0..1 to 0..W, 0..H
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()

        if isinstance(score, torch.Tensor):
            score = score.item()
        # Generate a random color for the box outline
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # Extract the box coordinates
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        if return_point:
            ceter_x = int((x0 + x1) / 2)
            ceter_y = int((y0 + y1) / 2)
            # Draw the center point on the input image
            draw.ellipse(
                (
                    ceter_x - point_width,
                    ceter_y - point_width,
                    ceter_x + point_width,
                    ceter_y + point_width,
                ),
                fill=color,
                width=point_width,
            )
        else:
            # Draw the box outline on the input image
            draw.rectangle([x0, y0, x1, y1], outline=color, width=int(point_width))

        # Draw the label text on the input image
        if return_score:
            text = f"{score:.2f}"
        else:
            text = f""
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), text, font)
        else:
            w, h = draw.textsize(text, font)
            bbox = (x0, y0, w + x0, y0 + h)
        if not return_point:
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), text, fill="white")

        # Draw the box on the mask image
        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)
    return image_pil, mask


def convert_all_cate_prediction_to_ans(prediction: str) -> Dict[str, List[str]]:
    # Define the pattern to extract ground truth labels and object tags
    pattern = r"<ground>(.*?)<\/ground><objects>(.*?)<\/objects>"

    # Find all matches in the prediction string
    matches = re.findall(pattern, prediction)

    # Initialize the dictionary to store the result
    ans = {}
    existed_obj_index = []
    for ground, objects in matches:
        ground = ground.strip()
        # Extract the object tags, e.g., <obj0>, <obj1>, <obj2>
        object_tags = re.findall(r"<obj\d+>", objects)
        filtered_object_tags = []
        for object_tag in object_tags:
            if object_tag in existed_obj_index:
                continue
            filtered_object_tags.append(object_tag)
            existed_obj_index.append(object_tag)

        if len(filtered_object_tags) == 0:
            continue
        # Add the ground truth label and associated object tags to the dictionary
        if ground not in ans:
            ans[ground] = object_tags

    return ans


def visualize(image, boxes, labels, font_size: int = 12, draw_width: int = 6):
    image = image.copy()
    font_path = "chatrex/tools/Tahoma.ttf"
    font = ImageFont.truetype(font_path, font_size)
    color_generaor = ColorGenerator("text")
    draw = ImageDraw.Draw(image)

    for box, label in zip(boxes, labels):
        x0, y0, x1, y1 = box
        if isinstance(label, list):
            label = label[0]
        color = color_generaor.get_color(label)
        text = label
        try:
            draw.rectangle(
                [x0, y0, x1, y1],
                outline=color,
                width=draw_width,
            )
        except Exception as e:
            print(f"error: {e}")
            continue
        bbox = draw.textbbox((x0, y0), text, font)
        box_h = bbox[3] - bbox[1]
        box_w = bbox[2] - bbox[0]

        y0_text = y0 - box_h - (draw_width * 2)
        y1_text = y0 + draw_width
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

    return image


def visualize_chatrex_output(
    image_pil: Image,
    input_boxes: List[List[int]],
    prediction_text: str,
    font_size=15,
    draw_width: int = 6,
) -> Image:
    """Plot bounding boxes and labels on an image.

    Args:
        image_pil (PIL.Image): The input image as a PIL Image object.
        input_boxes: A list of bounding boxes in shape (N, 4), in (x1, y1, x2, y2) format.
        prediction_text: The prediction text from the model.
        font_size: The font size for the text. Defaults to 15.
        draw_width: The width of the bounding box outline. Defaults to 6.

    Returns:
        PIL.Image: A tuple containing the input image and ploted image.
    """
    prediction_dict = convert_all_cate_prediction_to_ans(prediction_text)
    pred_boxes = []
    pred_labels = []
    for k, v in prediction_dict.items():
        for box in v:
            obj_idx = int(box[4:-1])
            if obj_idx < len(input_boxes):
                pred_labels.append(k)
                pred_boxes.append(input_boxes[obj_idx])
    image_pred = visualize(image_pil, pred_boxes, pred_labels, font_size, draw_width)
    return image_pred
