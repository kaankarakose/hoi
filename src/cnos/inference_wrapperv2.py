import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from src.utils.bbox_utils import CropResizePad
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
from src.model.utils import Detections, convert_npz_to_json
from src.model.loss import Similarity
from src.utils.inout import save_json_bop23

from src.utils.object_mask_merger import MaskMergingPipeline
import cv2
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
import json

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def mask_to_rle(mask):
    """
    Encodes a single mask to an uncompressed RLE, in the format expected by
    pycoco tools.
    
    Args:
        mask: A 2D numpy array of shape (h, w) with binary values
        
    Returns:
        Dictionary with 'size' and 'counts' fields
    """
    print("mask2rle",mask.shape)
    h, w = mask.shape
    # Put in fortran order and flatten (transpose and flatten)
    mask_flat = mask.T.flatten()
    
    # Compute change indices
    diff = mask_flat[1:] != mask_flat[:-1]  # XOR operation for boolean arrays
    change_indices = np.where(diff)[0]
    
    # Encode run length
    cur_idxs = np.concatenate([
        np.array([0], dtype=change_indices.dtype),
        change_indices + 1,
        np.array([h * w])
    ])
    
    btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
    counts = [] if mask_flat[0] == 0 else [0]
    counts.extend(btw_idxs.tolist())
    
    return {
        "size": [h, w],
        "counts": counts
    }

def is_overlapping(bbox1, bbox2, threshold=5):
    """
    Check if two bounding boxes overlap or are within a specified distance threshold.
    
    Args:
        bbox1 (list): First bounding box [x_min, y_min, x_max, y_max].
        bbox2 (list): Second bounding box [x_min, y_min, x_max, y_max].
        threshold (int): Distance threshold in pixels to consider as overlapping.
        
    Returns:
        bool: True if bounding boxes overlap or are within the threshold, False otherwise.
    """
    if not bbox1 or len(bbox1) < 4:
        return False
    
    # Expand each bounding box by the threshold
    expanded_bbox1 = [bbox1[0] - threshold, bbox1[1] - threshold, bbox1[2] + threshold, bbox1[3] + threshold]
    expanded_bbox2 = [bbox2[0] - threshold, bbox2[1] - threshold, bbox2[2] + threshold, bbox2[3] + threshold]
    
    # Check if the expanded bounding boxes overlap
    if expanded_bbox1[2] < expanded_bbox2[0] or expanded_bbox1[0] > expanded_bbox2[2]:
        return False
    if expanded_bbox1[3] < expanded_bbox2[1] or expanded_bbox1[1] > expanded_bbox2[3]:
        return False
    return True

def merge_two_bboxes(bbox1, bbox2):
    """
    Merge two bounding boxes into a single bounding box.
    
    Args:
        bbox1 (list): First bounding box [x_min, y_min, x_max, y_max].
        bbox2 (list): Second bounding box [x_min, y_min, x_max, y_max].
        
    Returns:
        list: Merged bounding box as [x_min, y_min, x_max, y_max].
    """
    x_min = min(bbox1[0], bbox2[0])
    y_min = min(bbox1[1], bbox2[1])
    x_max = max(bbox1[2], bbox2[2])
    y_max = max(bbox1[3], bbox2[3])
    
    return [x_min, y_min, x_max, y_max]

def visualize(rgb, detections, save_path="./tmp/tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # img = (255*img).astype(np.uint8)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33



    masks = getattr(detections, 'masks')
    object_ids = getattr(detections, 'object_ids')
    #TODO: I need to take care of unshaped masks!! --> conf_threshold
    for mask_idx in range(len(detections)):
        # Convert mask to numpy and threshold to get boolean mask
        mask = masks[mask_idx]
        mask = mask > 0.5  # Convert to boolean mask
        
        # # Strictly validate mask dimensions - should always be 2D
        # if len(mask.shape) < 2:
        #     print(f"ERROR: 1D mask detected in visualize function, mask {mask_idx} with shape {mask.shape}")
        #     raise ValueError(f"1D mask detected in visualize function, mask {mask_idx} with shape {mask.shape}")
            
        # # Skip empty masks (all False)
        # if not mask.any():
        #     print(f"Warning: Skipping empty mask {mask_idx} (all False values)")
        #     raise ValueError("Empty Mask!")
        edge = canny(mask.astype(np.uint8) * 255)
        edge = binary_dilation(edge, np.ones((2, 2)))
        obj_id = object_ids[mask_idx].item()
        temp_id = mask_idx  # Using mask_idx as temp_id since we're visualizing detections

        r = int(255*colors[temp_id][0])
        g = int(255*colors[temp_id][1])
        b = int(255*colors[temp_id][2])
        img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
        img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
        img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
        img[edge, :] = 255
    
    img = Image.fromarray(np.uint8(img))
    return img
    # img.save(save_path)
    # prediction = Image.open(save_path)
    
    # # concat side by side in PIL
    # img = np.array(img)
    # concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    # concat.paste(rgb, (0, 0))
    # concat.paste(prediction, (img.shape[1], 0))
    # return concat


def mask_to_bbox(mask):
    # Convert mask image to grayscale numpy array
    if isinstance(mask, str):
        mask = Image.open(mask).convert('L')
    elif isinstance(mask, Image.Image):
        mask = mask.convert('L')
  
    
    mask_array = np.array(mask)
    
    # Find the indices of non-zero elements
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    
    # Get the bounding box coordinates
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Return the bounding box as a JSON
    bbox = {
        "xmin": int(xmin),
        "ymin": int(ymin),
        "xmax": int(xmax),
        "ymax": int(ymax)
    }
    return bbox





class InferenceWrapper:
    """
    A wrapper around the run_inference function that provides more control over the inference process.
    """
    
    def __init__(self, 
                 conf_threshold=0.52,
                 output_dir=None,
                 gpu_id=0):
        """
        Initialize the InferenceWrapper.
        
        Args:
            conf_threshold (float): Confidence threshold for detections.
            output_dir (str): Directory to save results. If None, will use template_dir/cnos_results.
            gpu_id (int): GPU ID to use for inference (default: 0).
        """

        self.conf_threshold = conf_threshold
        self.output_dir = output_dir
        self.gpu_id = gpu_id
        
        # Set the GPU device for this wrapper
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            logging.info(f"Using GPU: {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
        
        # Initialize model
        with initialize(version_base=None, config_path="./configs"):
            self.cfg = compose(config_name='run_inference.yaml')
        
        # Configure segmentor model
        cfg_segmentor = self.cfg.model.segmentor_model
        # if "fast_sam" in cfg_segmentor._target_:
        #     logging.info("Using FastSAM, ignore stability_score_thresh!")
        # else:
        #     self.cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
        
        # Initialize similarity metric
        self.metric = Similarity()
        
        # Initialize model
        logging.info(f"Initializing model on GPU {gpu_id}")
        self.model = instantiate(self.cfg.model)
        
        # Move model to device
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model.descriptor_model.model = self.model.descriptor_model.model.to(self.device)
        self.model.descriptor_model.model.device = self.device
        
        # Move segmentor model to device if it has a predictor
        if hasattr(self.model.segmentor_model, "predictor"):
            self.model.segmentor_model.predictor.model = (
                self.model.segmentor_model.predictor.model.to(self.device)
            )
        else:
            self.model.segmentor_model.model.setup_model(device=self.device, verbose=True)
        
        logging.info(f"Moving models to {self.device} (GPU {gpu_id}) done!")
        
        # Initialize processor for templates
        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        self.proposal_processor = CropResizePad(processing_config.image_size)
        
        # Store reference features
        self.ref_feats = None
        self.template_dir = None
        self.scores = None
    def load_templates(self, template_dir):
        """
        Load templates from the given directory.
        
        Args:
            template_path (str): Path to the directory containinh dino vectors npz.
        
        Returns:
            torch.Tensor: Reference features for the templates.
        """
        # directly load from the path!
        if template_dir.endswith('.npz'):
            data = np.load(template_dir)
            self.ref_feats = torch.from_numpy(data['ref_feats']).to(self.device)
            return self.ref_feats
            
        
    def run(self, rgb_image, custom_conf_threshold=None, custom_num_max_dets=None):
        """
        Run inference on the given RGB image.
        
        Args:
            rgb_image: Either a path to an RGB image or a PIL Image object.
            custom_conf_threshold (float, optional): Custom confidence threshold to use for this run.
            custom_num_max_dets (int, optional): Custom maximum number of detections to use for this run.
        
        Returns:
            Detections: Object containing the detection results.
        """
        if self.ref_feats is None:
            raise ValueError("Templates not loaded. Call load_templates() first.")
        
        # Use custom parameters if provided
        conf_threshold = custom_conf_threshold if custom_conf_threshold is not None else self.conf_threshold
        #num_max_dets = custom_num_max_dets if custom_num_max_dets is not None else self.num_max_dets

        frame_name = os.path.basename(rgb_image).split('.')[0] # get rid of jpg
        vis_output = os.path.join(self.output_dir, 'frames')
        os.makedirs(vis_output, exist_ok = True)

        # Load RGB image
        if isinstance(rgb_image, str):
            rgb = Image.open(rgb_image).convert("RGB")
            rgb_path = rgb_image
        else:
            rgb = rgb_image
            rgb_path = "memory_image"
        
        print(f'Input image size: {rgb.size}')
        

        mask_merging_pipeline = MaskMergingPipeline(
                                segmentor_model= self.model.segmentor_model,
                                descriptor_model= self.model.descriptor_model,
                                ref_feats= self.ref_feats,
                                similarity_metric= self.metric,
                                conf_threshold = conf_threshold
        )

        detections, scores = mask_merging_pipeline.process_image(np.array(rgb))


        # No mask detected!
        if detections is None:
            return None, None
        score_per_detection = torch.topk(scores, k=10, dim=-1)[0]
        score_per_detection = torch.mean(score_per_detection, dim=-1)
        print(f'Mean scores shape: {score_per_detection.shape}')
  
        scores, index = torch.topk(score_per_detection, k=len(score_per_detection), dim=-1)
        # Convert the index tensor to the correct format for indexing
        index = index.squeeze()  # Remove any extra dimensions

        detections.filter(index)
        

        # Convert boolean mask to indices
        keep_indices = torch.nonzero(scores > conf_threshold).squeeze()
        if len(keep_indices.shape) == 0 and keep_indices.numel() > 0:
            # Handle case where there's only one detection
            keep_indices = keep_indices.unsqueeze(0)
        if keep_indices.numel() > 0:
            detections.filter(keep_indices)
        else:
            print("Warning: No detections above confidence threshold!")

        print(f'scores={scores}')
        # Add attributes to detections
        detections.add_attribute("scores", scores)
        detections.add_attribute("object_ids", torch.zeros_like(scores))
        
        # Save results
        print('\nConverting detections to numpy and saving...')
        detections.to_numpy()
        
        # # Validate mask dimensions after numpy conversion
        # if hasattr(detections, 'masks'):
        #     masks = detections.masks
        #     if isinstance(masks, list):
        #         for i, mask in enumerate(masks):
        #             if len(mask.shape) != 2:
        #                 print(f"ERROR: 1D mask detected after numpy conversion, index {i}, shape {mask.shape}")
        #                 raise ValueError(f"1D mask detected after numpy conversion, index {i}, shape {mask.shape}")
        #     elif isinstance(masks, np.ndarray):
        #         for i in range(masks.shape[0]):
        #             mask = masks[i]
        #             if len(mask.shape) != 2:
        #                 raise ValueError(f"1D mask detected in numpy array at index {i}, shape {mask.shape}")

        # Visualize detections
        # for mask_idx, mask in enumerate(detections.masks):
        #     if len(mask.shape) < 2:
        #         print(f"ERROR: 1D mask detected in visualize function, mask {mask_idx} with invalid shape {mask.shape}")
        #         raise ValueError(f"1D mask detected in visualize function, mask {mask_idx} with shape {mask.shape}")
        vis_img = visualize(rgb, detections)
        vis_img.save(f"{vis_output}/{frame_name}.png")
       
        torch.cuda.empty_cache()
        return detections, scores
    
    def get_masks(self, detections):
        """
        Get masks from detections.
        
        Args:
            detections (Detections): Detection object.
        
        Returns:
            numpy.ndarray: Masks from detections.
        """
        return getattr(detections, 'masks')
    
    def get_scores(self, detections):
        """
        Get scores from detections.
        
        Args:
            detections (Detections): Detection object.
        
        Returns:
            numpy.ndarray: Scores from detections.
        """
        self.scores = getattr(detections, 'scores')
        return self.scores
    
    def get_object_ids(self, detections):
        """
        Get object IDs from detections.
        
        Args:
            detections (Detections): Detection object.
        
        Returns:
            numpy.ndarray: Object IDs from detections.
        """
        return getattr(detections, 'object_ids')
    
    ## This wasn't efficent for saving mask.
    # def save_masks(self, detections, frame_info=None):
    #     """
    #     Save masks from detections as individual PNG files.
        
    #     Args:
    #         detections (Detections): Detection object.
    #         output_dir (str, optional): Directory to save masks. If None, will use self.output_dir.
        
    #     Returns:
    #         list: Paths to saved mask files.
    #     """
        
    #     output_dir = os.path.join(self.output_dir, 'masks', frame_info)
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     masks = self.get_masks(detections)
    #     mask_paths = []
    #     if self.scores is None:
    #         raise ValueError('No scores!')

    #     scores = self.scores

    #     for i, mask in enumerate(masks):

    #         mask_path = f"{output_dir}/mask_{i}.png"
    #         mask_img = np.uint8(mask * 255)
    #         cv2.imwrite(mask_path, mask_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    #         mask_paths.append(mask_path)
        
    #     return True
    def save_masks(self, detections, frame_info=None):
        """
        Save masks from detections as RLE (Run-Length Encoding) format.
        
        Args:
            detections (Detections): Detection object.
            frame_info (str, optional): Frame information for directory naming.
        
        Returns:
            bool: True if successfully saved.
        """
        
        output_dir = os.path.join(self.output_dir, 'masks', frame_info)
        os.makedirs(output_dir, exist_ok=True)
        
        masks = self.get_masks(detections)
        if self.scores is None:
            raise ValueError('No scores!')

        # Dictionary to store all masks with their scores
        masks_data = {}
        
        for i, mask in enumerate(masks):
            # Convert mask to RLE format
            rle = mask_to_rle(mask)
            
            # Save individual RLE to a file
            rle_path = f"{output_dir}/mask_{i}.rle"
            with open(rle_path, 'w') as f:
                json.dump({
                        'counts': rle['counts'],
                        'size': rle['size'],
                        'score': float(self.scores[i])
                    }, f, cls=NumpyEncoder)
            
            # Add to the combined dictionary
            masks_data[f"mask_{i}"] = {
                'rle': rle,
                'score': float(self.scores[i])
            }
        
        # Save all masks in a single JSON file (optional but convenient)
        all_masks_path = f"{output_dir}/all_masks.json"
        with open(all_masks_path, 'w') as f:
            json.dump(masks_data, f, cls=NumpyEncoder)
        
        return True
    
        
     
    def save_bbox(self, detections, object_name, frame_info=None):
        """
        Save bounding boxes and object names as a JSON file.

        Args:
            detections (Detections): Detection object
            object_name (str): Name of the detected object
            frame_info (str, optional): Frame identifier for filename

        Returns:
            bool: True if save was successful
        """
        output_dir = os.path.join(self.output_dir, 'bbox')
        os.makedirs(output_dir, exist_ok=True)

        bboxes = []
        if detections is not None:
            masks = self.get_masks(detections)
            for mask in masks:
                bbox = mask_to_bbox(mask)
                bboxes.append({
                    'label': object_name,
                    'bbox_modal': bbox
                })

        json_path = os.path.join(output_dir, f'{frame_info}.json')
        with open(json_path, 'w') as f:
            json.dump(bboxes, f, indent=2)

        return True
    def save_scores(self, scores, frame_info=None):
        """
        Save scores as individual text files.
        
        Args:
            scores (torch.Tensor): Tensor containing scores.
            frame_info (str, optional): Additional frame information to include in the output directory path.
            
        Returns:
            list: Paths to saved score files.
        """
        import os
        
        output_dir = os.path.join(self.output_dir, 'scores')
        if frame_info is not None:
            output_dir = os.path.join(output_dir, frame_info)
        os.makedirs(output_dir, exist_ok=True)
        
        score_paths = []
        
        for i, score in enumerate(scores):
            if score < self.conf_threshold: continue # Don't save less then conf
            score_path = f"{output_dir}/score_{i}.txt"
            with open(score_path, 'w') as f:
                f.write(str(score.item()))
            score_paths.append(score_path)
        return True
if __name__ == "__main__":

    # Example usage
    object_name = "AMF1"
    output_dir = '/nas/project_data/B1_Behavior/rush/ados-objects/object_pose_mega/rtdt/cnos/cropped_new_sam'
    wrapper = InferenceWrapper(
        conf_threshold=0.01,
        output_dir=output_dir
    )
    


    # Load templates
    wrapper.load_templates(f"/nas/project_data/B1_Behavior/rush/ados-objects/object_pose_mega/rtdt/cnos/cnos/rendered_objects/{object_name}")
    
    # Run inference
    frame_folder = "/nas/project_data/B1_Behavior/rush/ados-objects/object_pose_mega/rtdt/hand-object/media/output"

    frames = [os.path.join(frame_folder, path) for path in os.listdir(frame_folder)]

    for frame in frames:

        detections, scores = wrapper.run(frame)
        # Get masks and scores
        masks = wrapper.get_masks(detections)
        scores = wrapper.get_scores(detections)
        
        frame_info = os.path.basename(frame).split('.')[0]
        # Save masks
        wrapper.save_masks(detections,frame_info)
        wrapper.save_bbox(detections, object_name, frame_info)
        wrapper.save_scores(scores, frame_info)
        torch.cuda.empty_cache()