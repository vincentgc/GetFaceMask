'''
Author: Fuxi
Description: 从人脸中提取出脸部分割mask
Date: 2020-11-16 18:29:56
LastEditors: fuxi
LastEditTime: 2020-11-16 20:09:10
'''
import os
import sys
import time
sys.path.append(os.path.realpath(os.path.dirname(__file__)))

import cv2
import numpy as np
import torch

from maskUtils.FaceParser import FaceParser

class FaceMask():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # check gpu
        self.face_parser = FaceParser(device=self.device)
        
        self.face_parts_idx = {"background":0, "skin":1, "right_eyebrow":2, "left_eyebrow":3, "right_eye":4, \
                        "left_eye":5, "nose":6, "up_lip":7, "tooth":8, "low_lip":9, "hair":10, \
                        "right_ear":11, "left_ear":12, "neck":13}
        self.face_parts = self.face_parts_idx.keys()
                        
    def get_mask(self, image):
        """Obtain the face mask segmentation from face image

        Args:
            image (numpy.ndarray)
        """
        start = time.time()
        mask = self.face_parser.parse(cv2.resize(image, (512, 512)))
            
        if self.device == "cuda":
            mask = mask.cpu()

        mask = mask.numpy().astype(np.uint8)
        print("Getting mask using [{}] mode cost {:3f}s".format(self.device, time.time()-start))
        return mask

    def part_name_assert(self, part):
        assert (part in self.face_parts), f"Cannot extract {part}'s mask, supported parts are {self.face_parts}"

    def get_part_mask_from_image(self, image, part):
        """Obtain the part mask from face image
            
        Args:
            image (numpy.ndarray)
            part (str or list): part name or [part1, part2, ...]
        """
        mask = self.get_mask(image)
        part_mask = self.get_part_mask_from_mask(mask, part)
        return part_mask

    def get_part_mask_from_mask(self, mask, part):
        """Obtain the part mask from face mask
            
        Args:
            mask (numpy.ndarray)
            part (str or list): part name or [part1, part2, ...]
        """
        if isinstance(part, str):
            self.part_name_assert(part)
            part_idx = self.face_parts_idx[part]
            part_mask = mask == part_idx
        elif isinstance(part, list):
            part_mask = np.zeros(mask.shape)
            for p in part:
                self.part_name_assert(p)
                part_idx = self.face_parts_idx[p]
                part_mask += mask == part_idx
        else:
            raise("Unsupported part format")
        return part_mask

    @staticmethod
    def visual_mask_by_color(mask):
        """Visual the mask using color
        Args:
            mask (numpy.ndarray)
        """
        H, W = mask.shape
        mask_with_color = np.zeros((H, W, 3), dtype=np.uint8)
        for i in range(int(mask.max())):
            color = np.random.rand(1, 1, 3) * 255
            mask_with_color += (mask == i)[:, :, None] * color.astype(np.uint8)
        return mask_with_color


if __name__ == "__main__":
    
    test_img_path = "./maskUtils/test_face.jpg"
    assert(os.path.exists(test_img_path))
    
    np_image = cv2.imread(test_img_path)
    faceMask = FaceMask()

    # # test get_mask()
    # mask = faceMask.get_mask(np_image)
    # cv2.imshow("mask", mask)
    # cv2.waitKey()

    # test get_part_mask_from_image()
    # part_mask = faceMask.get_part_mask_from_image(np_image, ["left_eye","right_eye","skin","hair"])
    # cv2.imshow("mask", (part_mask*255).astype(np.uint8))
    # cv2.waitKey()

    # test visual_mask_by_color
    mask = faceMask.get_mask(np_image)
    mask_with_color = faceMask.visual_mask_by_color(mask)
    cv2.imshow("mask", mask_with_color)
    cv2.waitKey()

