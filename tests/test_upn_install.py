from PIL import Image

from chatrex.tools.visualize import plot_boxes_to_image
from chatrex.upn import UPNWrapper

if __name__ == "__main__":
    ckpt_path = "checkpoints/upn_checkpoints/upn_large.pth"
    test_image_path = "tests/images/test_upn.jpeg"

    model = UPNWrapper(ckpt_path)
    # fine-grained prompt
    fine_grained_proposals = model.inference(
        test_image_path, prompt_type="fine_grained_prompt"
    )
    fine_grained_filtered_proposals = model.filter(
        fine_grained_proposals, min_score=0.3, nms_value=0.8
    )

    # coarse-grained prompt
    coarse_grained_proposals = model.inference(
        test_image_path, prompt_type="coarse_grained_prompt"
    )
    coarse_grained_filtered_proposals = model.filter(
        coarse_grained_proposals, min_score=0.3, nms_value=0.8
    )

    # visualize
    image = Image.open(test_image_path)
    fine_grained_vis_image, _ = plot_boxes_to_image(
        image.copy(),
        fine_grained_filtered_proposals["original_xyxy_boxes"][0],
        fine_grained_filtered_proposals["scores"][0],
    )
    fine_grained_vis_image.save("tests/test_image_fine_grained.jpeg")
    print(f"fine-grained proposal is saved at tests/test_image_fine_grained.jpeg")

    coarse_grained_vis_image, _ = plot_boxes_to_image(
        image.copy(),
        coarse_grained_filtered_proposals["original_xyxy_boxes"][0],
        coarse_grained_filtered_proposals["scores"][0],
    )
    coarse_grained_vis_image.save("tests/test_image_coarse_grained.jpeg")
    print(f"coarse-grained proposal is saved at tests/test_image_coarse_grained.jpeg")
