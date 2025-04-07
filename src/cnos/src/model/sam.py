from segment_anything import (
    sam_model_registry,
    SamPredictor,
    SamAutomaticMaskGenerator,
)
from segment_anything.modeling import Sam
from segment_anything.utils.amg import MaskData, generate_crop_boxes, rle_to_mask
import logging
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple
import cv2
import torch.nn.functional as F

pretrained_weight_dict = {
    "vit_l": "sam_vit_l_0b3195.pth",  # 1250MB
    "vit_b": "sam_vit_b_01ec64.pth",  # 375MB
    "vit_h": "sam_vit_h_4b8939.pth",  # 2500MB
}


def load_sam(model_type, checkpoint_dir):
    logging.info(f"Loading SAM model from {checkpoint_dir}")
    sam = sam_model_registry[model_type](
        checkpoint=osp.join(checkpoint_dir, pretrained_weight_dict[model_type])
    )
    return sam


def load_sam_predictor(model_type, checkpoint_dir, device):
    logging.info(f"Loading SAM model from {checkpoint_dir}")
    sam = sam_model_registry[model_type](
        checkpoint=osp.join(checkpoint_dir, pretrained_weight_dict[model_type])
    )
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def load_sam_mask_generator(model_type, checkpoint_dir, device):
    logging.info(f"Loading SAM model from {checkpoint_dir}")
    sam = sam_model_registry[model_type](
        checkpoint=osp.join(checkpoint_dir, pretrained_weight_dict[model_type])
    )
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, output_mode="coco_rle")
    return mask_generator


class CustomSamAutomaticMaskGenerator(SamAutomaticMaskGenerator):
    def __init__(
        self,
        sam: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        segmentor_width_size=None,
    ):
        # Pass all parameters to the parent class
        SamAutomaticMaskGenerator.__init__(
            self,
            sam,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            point_grids=point_grids,
            min_mask_region_area=min_mask_region_area,
            output_mode=output_mode,
        )
        self.segmentor_width_size = segmentor_width_size
        logging.info(f"Init CustomSamAutomaticMaskGenerator done!")

    def preprocess_resize(self, image: np.ndarray):
        orig_size = image.shape[:2]
        height_size = int(self.segmentor_width_size * orig_size[0] / orig_size[1])
        resized_image = cv2.resize(
            image.copy(), (self.segmentor_width_size, height_size)  # (width, height)
        )
        return resized_image

    def postprocess_resize(self, detections, orig_size):
        detections["masks"] = F.interpolate(
            detections["masks"].unsqueeze(1).float(),
            size=(orig_size[0], orig_size[1]),
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]
        scale = orig_size[1] / self.segmentor_width_size
        detections["boxes"] = detections["boxes"].float() * scale
        detections["boxes"][:, [0, 2]] = torch.clamp(
            detections["boxes"][:, [0, 2]], 0, orig_size[1] - 1
        )
        detections["boxes"][:, [1, 3]] = torch.clamp(
            detections["boxes"][:, [1, 3]], 0, orig_size[0] - 1
        )
        return detections

    @torch.no_grad()
    def generate_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.segmentor_width_size is not None:
            orig_size = image.shape[:2]
            image = self.preprocess_resize(image)
        # Generate masks
        mask_data = self._generate_masks(image) # Dictinary
        # Mask Data isn't a MaskData object
        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            missing_data = MaskData()
            missing_data['masks'] = mask_data['masks']
            missing_data['boxes'] = mask_data['boxes']
            missing_data['rles'] = mask_data['rles']

            processed_mask_data = self.postprocess_small_regions(
                missing_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )
            # Convert back to dictionary
            mask_data = {
                'masks': processed_mask_data['masks'],
                'boxes': processed_mask_data['boxes']
            }
        
        if self.segmentor_width_size is not None:
            mask_data = self.postprocess_resize(mask_data, orig_size)
        return mask_data

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data["masks"] = [torch.from_numpy(rle_to_mask(rle)) for rle in data["rles"]]
        data["masks"] = torch.stack(data["masks"])
        return {"masks": data["masks"].to(data["boxes"].device),
         "boxes": data["boxes"],
         "rles": data["rles"]}

    def remove_small_detections(self, mask_data: MaskData, img_size: List) -> MaskData:
        # calculate area and number of pixels in each mask
        area = box_area(mask_data["boxes"]) / (img_size[0] * img_size[1])
        idx_selected = area >= self.mask_post_processing.min_box_size
        mask_data.filter(idx_selected)
        return mask_data