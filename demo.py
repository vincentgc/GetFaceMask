'''
Author: Fuxi
Description: 
Date: 2020-11-17 12:18:48
LastEditors: fuxi
LastEditTime: 2020-11-17 14:47:08
'''
import os

import cv2
import numpy as np

from FaceMask import FaceMask

# getting face mask demo

test_img_path = "./test_face.jpg"
assert(os.path.exists(test_img_path))
output_dir = "./results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
# read image
np_image = cv2.imread(test_img_path)
faceMask = FaceMask()

# Get a mask from image (np.ndarray)
mask = faceMask.get_mask(np_image)
cv2.imwrite(os.path.join(output_dir, "mask.png"), mask)

# Get mask of face part
# include: {"background":0, "skin":1, "right_eyebrow":2, "left_eyebrow":3, "right_eye":4, \
#           "left_eye":5, "nose":6, "up_lip":7, "tooth":8, "low_lip":9, "hair":10, \
#           "right_ear":11, "left_ear":12, "neck":13}
part_mask = faceMask.get_part_mask_from_image(np_image, "hair")
cv2.imwrite(os.path.join(output_dir, "part_mask.png"), (part_mask*255).astype(np.uint8))

# Get multi parts face mask
multi_parts_mask = faceMask.get_part_mask_from_image(np_image, ["left_eye","right_eye","skin","hair"])
cv2.imwrite(os.path.join(output_dir, "multi_parts_mask.png"), (multi_parts_mask*255).astype(np.uint8))

# Visualize the mask by color
mask = faceMask.get_mask(np_image)
mask_with_color = faceMask.visual_mask_by_color(mask)
cv2.imwrite(os.path.join(output_dir, "mask_color.png"), mask_with_color)
