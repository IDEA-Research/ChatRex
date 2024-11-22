from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
