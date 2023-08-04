# coding: utf-8

from gradio_app import get_grounding_output, load_model, config_file, ckpt_filenmae, box_threshold, text_threshold, \
    transform_image
from PIL import Image
import torch


def main():
    image = Image.open("img_data/WechatIMG2728.jpeg")
    image_pil = image.convert("RGB")
    size = image.size  # w, h

    groundingdino_model = load_model(config_file, ckpt_filenmae, device="cpu")
    transformed_image = transform_image(image_pil)
    text_prompt = "person, cell phone"

    boxes_filt, scores, pred_phrases = get_grounding_output(groundingdino_model, transformed_image, text_prompt,
                                                            box_threshold, text_threshold)
    # process boxes
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
