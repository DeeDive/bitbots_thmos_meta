from __future__ import division

import numpy as np


import torch
import torchvision.transforms as transforms

from .utils111 import rescale_boxes, non_max_suppression
from .transforms111 import Resize, DEFAULT_TRANSFORMS


def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5):
    """
    :type model: models.Darknet
    :type image: np.array

    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: np.array
    """
    model.eval()  # Set model to evaluation mode

    # Configure input
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)
    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return detections.numpy()

