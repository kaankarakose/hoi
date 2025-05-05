import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
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

import cv2
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    #print("mask2rle",mask.shape)
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

def visualize(rgb, detections, object_names=None, save_path=None):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    all_masks = []
    all_object_ids = []
    
    # Collect all masks and object IDs from all detections
    for obj_idx, detection_list in enumerate(detections):
        if detection_list is None:
            continue
            
        masks = getattr(detection_list, 'masks')
        object_ids = getattr(detection_list, 'object_ids')
        
        for mask_idx in range(len(detection_list)):
            all_masks.append(masks[mask_idx])
            # Use obj_idx to identify which object type this mask belongs to
            all_object_ids.append(obj_idx)
    
    # Generate colors for visualization
    colors = distinctipy.get_colors(len(object_names) if object_names else max(all_object_ids) + 1)
    alpha = 0.33
    
    # Visualize all masks
    for idx, (mask, obj_id) in enumerate(zip(all_masks, all_object_ids)):
        # Convert mask to boolean mask
        mask = mask > 0.5
        
        edge = canny(mask.astype(np.uint8) * 255)
        edge = binary_dilation(edge, np.ones((2, 2)))
        
        r = int(255*colors[obj_id][0])
        g = int(255*colors[obj_id][1])
        b = int(255*colors[obj_id][2])
        
        img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
        img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
        img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
        img[edge, :] = 255
        
        # Add a label if possible
        if object_names:
            # Find topmost point of the mask
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                top_y = y_coords.min()
                center_x = x_coords.mean().astype(int)
                
                # Add label text
                label = object_names[obj_id]
                cv2.putText(img, label, (center_x, top_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (r, g, b), 2)
    
    img = Image.fromarray(np.uint8(img))
    if save_path:
        img.save(save_path)
    return img

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
    
    if not np.any(rows) or not np.any(cols):
        # Return empty bbox if mask is empty
        return {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}
    
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

class MultiObjectInferenceWrapper:
    """
    A wrapper around the Cnos inference model that can handle multiple objects efficiently.
    """
    
    def __init__(self, 
                 conf_threshold=0.52,
                 output_dir=None,
                 gpu_id=0,
                 session_name=None,
                 camera_view=None,
                 frame_type=None):
        """
        Initialize the MultiObjectInferenceWrapper.
        
        Args:
            conf_threshold (float): Confidence threshold for detections.
            output_dir (str): Directory to save results.
            gpu_id (int): GPU ID to use for inference (default: 0).
            session_name (str, optional): Name of the session for directory structure.
            camera_view (str, optional): Camera view for directory structure.
            frame_type (str, optional): Frame type (L_frames/R_frames) for directory structure.
        """
        self.conf_threshold = conf_threshold
        self.output_dir = output_dir
        self.gpu_id = gpu_id
        
        # Set session directory structure information
        self.session_name = session_name
        self.camera_view = camera_view
        self.frame_type = frame_type
        
        # Set the GPU device for this wrapper
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            logging.info(f"Using GPU: {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
        
        # Initialize model
        with initialize(version_base=None, config_path="./configs"):
            self.cfg = compose(config_name='run_inference.yaml')
        
        # Initialize similarity metric
        self.metric = Similarity()
        
        # Initialize model
        #logging.info(f"Initializing model on GPU {gpu_id}")
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
        
        #logging.info(f"Moving models to {self.device} (GPU {gpu_id}) done!")
        
        # Initialize processor for templates
        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        self.proposal_processor = CropResizePad(processing_config.image_size)
        
        # Store reference features for multiple objects
        self.ref_feats_dict = {}
        self.object_names = []
        self.scores_dict = {}
        
    def load_templates(self, template_paths_dict):
        """
        Load templates for multiple objects.
        
        Args:
            template_paths_dict (dict): Dictionary mapping object names to template paths.
        
        Returns:
            dict: Dictionary mapping object names to reference features.
        """
        for object_name, template_path in template_paths_dict.items():
            if template_path.endswith('.npz'):
                data = np.load(template_path)
                self.ref_feats_dict[object_name] = torch.from_numpy(data['ref_feats']).to(self.device)
                self.object_names.append(object_name)
            else:
                logging.warning(f"Template path for {object_name} must be an .npz file")
                
        #logging.info(f"Loaded templates for {len(self.ref_feats_dict)} objects: {', '.join(self.object_names)}")
        return self.ref_feats_dict
    
    def run_single_frame(self, rgb_image, custom_conf_threshold=None):
        """
        Run inference on a single frame for all loaded objects.
        
        Args:
            rgb_image: Either a path to an RGB image or a PIL Image object.
            custom_conf_threshold (float, optional): Custom confidence threshold.
            
        Returns:
            tuple: (all_detections, frame_name, rgb)
                - all_detections: Dictionary mapping object names to detection results.
                - frame_name: Name of the processed frame.
                - rgb: Processed RGB image.
        """
        if not self.ref_feats_dict:
            raise ValueError("Templates not loaded.")
        
        # Use custom parameters if provided
        conf_threshold = custom_conf_threshold if custom_conf_threshold is not None else self.conf_threshold

        # Load RGB image
        if isinstance(rgb_image, str):
            rgb = Image.open(rgb_image).convert("RGB")
            rgb_path = rgb_image
            frame_name = os.path.basename(rgb_image).split('.')[0]
        else:
            rgb = rgb_image
            rgb_path = "memory_image"
            frame_name = "memory_image"
        
        #logging.info(f'Processing {frame_name}, input image size: {rgb.size}')
        
        # Generate masks with segmentor model (this is common for all objects)
        #logging.info('Generating masks with segmentor model...')
        raw_detections = self.model.segmentor_model.generate_masks(np.array(rgb))
        base_detections = Detections(raw_detections)
        
        # Generate descriptors (this is common for all objects)
        #logging.info('Generating descriptors...')
        descriptors = self.model.descriptor_model.forward(np.array(rgb), base_detections)
        
        # Dictionary to store detections for each object
        all_detections = {}
        
        # For each object, calculate similarity scores and filter detections
        for object_name in self.object_names:
            try:
                # Wrap the entire process in a try-except to catch any unexpected errors
                ref_feats = self.ref_feats_dict[object_name]
                #logging.info(f"Processing {object_name}: ref_feats shape={ref_feats.shape}, device={ref_feats.device}")
                #logging.info(f"descriptors shape={descriptors.shape}, device={descriptors.device}")
                
                if descriptors.numel() == 0:
                    logging.warning(f"Empty descriptors tensor for {object_name}")
                    all_detections[object_name] = None
                    continue
                    
                if ref_feats.numel() == 0:
                    logging.warning(f"Empty reference features for {object_name}")
                    all_detections[object_name] = None
                    continue
                
                # Calculate similarity scores for this object
                #logging.info(f'Calculating similarity scores for {object_name}...')
                try:
                    scores = self.metric(descriptors[:, None, :], ref_feats[None, :, :])
                    #logging.info(f"Scores tensor shape: {scores.shape}")
                except Exception as e:
                    #logging.error(f"Error in metric calculation for {object_name}: {str(e)}")
                    all_detections[object_name] = None
                    continue
                
                # Check if scores tensor is valid
                if scores.numel() == 0 or scores.shape[-1] == 0:
                    #logging.warning(f"Empty scores tensor for {object_name}")
                    all_detections[object_name] = None
                    continue
                    
                # Get top-k scores per detection
                try:
                    k_value = min(10, scores.shape[-1])
                    #logging.info(f"Using k_value={k_value} for first topk")
                    score_per_detection = torch.topk(scores, k=k_value, dim=-1)[0]
                    #logging.info(f"Score_per_detection shape after first topk: {score_per_detection.shape}")
                    score_per_detection = torch.mean(score_per_detection, dim=-1)
                    #logging.info(f"Score_per_detection shape after mean: {score_per_detection.shape}")
                    
                    # Before using topk, make sure score_per_detection has valid elements
                    num_elements = score_per_detection.numel()
                    #logging.info(f"Score_per_detection has {num_elements} elements")
                    
                    if num_elements == 0:
                        logging.warning(f"Empty score_per_detection tensor for {object_name}")
                        all_detections[object_name] = None
                        continue
                    
                    # Additional safety check - make sure tensor is at least 1D
                    if len(score_per_detection.shape) == 0:
                        #logging.warning(f"score_per_detection is a scalar, not a tensor with dimensions")
                        # Convert to a 1D tensor with one element
                        score_per_detection = score_per_detection.unsqueeze(0)
                        #logging.info(f"Converted to shape: {score_per_detection.shape}")
                    
                    # Use min to ensure k is not larger than the tensor size
                    k_value = min(num_elements, score_per_detection.size(0))
                    #logging.info(f"Using k_value={k_value} for second topk with tensor size {score_per_detection.size(0)}")
                    
                    if k_value == 0:
                        #logging.warning(f"k_value is 0 for {object_name}, cannot perform topk")
                        all_detections[object_name] = None
                        continue
                        
                    scores, index = torch.topk(score_per_detection, k=k_value, dim=-1)
                    #logging.info(f"Final scores shape: {scores.shape}, index shape: {index.shape}")
                except Exception as e:
                    #logging.error(f"Error in topk processing for {object_name}: {str(e)}")
                    #logging.error(f"score_per_detection: shape={score_per_detection.shape if 'score_per_detection' in locals() else 'not created'}")
                    all_detections[object_name] = None
                    continue
            except Exception as e:
                #logging.error(f"Unexpected error processing {object_name}: {str(e)}")
                all_detections[object_name] = None
                continue
                
            try:
                # Clone base detections for this object
                #logging.info(f"Cloning base detections with index shape: {index.shape}")
                obj_detections = base_detections.clone()
                
                try:
                    # Convert to CPU and ensure proper integer type before filtering
                    #logging.info(f"Preparing index for filtering: shape={index.shape}, device={index.device}, dtype={index.dtype}")
                    
                    # Ensure we pass a CPU-based index to filter method
                    # The filter method itself also handles this, but we do it here for clarity
                    if index.device.type != 'cpu':
                        index = index.cpu()
                    
                    # Ensure it's the right datatype for indexing
                    if index.dtype not in [torch.int32, torch.int64, torch.long]:
                        index = index.long()
                    
                    #logging.info(f"Using cleaned index for filtering: device={index.device}, dtype={index.dtype}")
                    
                    # Pre-check attribute devices by moving all tensors to CPU before filtering
                    for key in obj_detections.keys:
                        attr = getattr(obj_detections, key)
                        if torch.is_tensor(attr) and attr is not None and attr.device.type != 'cpu':
                            #logging.info(f"Moving attribute '{key}' from {attr.device} to CPU before filtering")
                            setattr(obj_detections, key, attr.cpu())
                    
                    # Now filter with prepared index and attributes
                    obj_detections.filter(index)
                    #logging.info(f"Successfully filtered detections with index")
                except Exception as e:
                    logging.error(f"Error in first filter operation: {str(e)}")
                    # Add detailed debug info
                    #logging.error(f"Index shape: {index.shape}, device: {index.device}, dtype: {index.dtype}")
                    # Check all attributes of obj_detections and log their devices
                    for key in obj_detections.keys:
                        attr = getattr(obj_detections, key)
                        #if attr is not None and torch.is_tensor(attr) and hasattr(attr, 'device'):
                            #logging.error(f"Attribute '{key}' device: {attr.device}, shape: {attr.shape}, dtype: {attr.dtype}")
                    all_detections[object_name] = None
                    continue
                    
                # Filter by confidence threshold
                mask = scores > conf_threshold
                #logging.info(f"Confidence mask sum: {torch.sum(mask)} out of {mask.numel()}")
                
                if torch.any(mask):
                    try:
                        keep_indices = torch.nonzero(mask).squeeze(-1)
                        #logging.info(f"keep_indices shape: {keep_indices.shape}, numel: {keep_indices.numel()}")
                        
                        # Handle case where there's only one detection
                        if keep_indices.numel() == 1:
                            keep_indices = keep_indices.unsqueeze(0)
                            #logging.info(f"Converted single element to shape: {keep_indices.shape}")
                        
                        #logging.info(f"Preparing keep_indices for filtering: shape={keep_indices.shape}, device={keep_indices.device}, dtype={keep_indices.dtype}")
                        
                        # Ensure proper device and dtype
                        if keep_indices.device.type != 'cpu':
                            keep_indices = keep_indices.cpu()
                            
                        if keep_indices.dtype not in [torch.int32, torch.int64, torch.long]:
                            keep_indices = keep_indices.long()
                            
                        #logging.info(f"Using prepared keep_indices: device={keep_indices.device}, dtype={keep_indices.dtype}")
                        
                        # Pre-check attribute devices by moving all tensors to CPU before filtering
                        for key in obj_detections.keys:
                            attr = getattr(obj_detections, key)
                            if torch.is_tensor(attr) and attr is not None and attr.device.type != 'cpu':
                                #logging.info(f"Moving attribute '{key}' from {attr.device} to CPU before second filtering")
                                setattr(obj_detections, key, attr.cpu())
                        
                        # Filter with prepared indices
                        obj_detections.filter(keep_indices)
                        #logging.info(f"Successfully filtered with keep_indices")
                        
                        filtered_scores = scores[keep_indices]
                        #logging.info(f"Filtered scores shape: {filtered_scores.shape}")
                        
                        # Add attributes to detections
                        obj_detections.add_attribute("scores", filtered_scores)
                        obj_detections.add_attribute("object_ids", torch.zeros_like(filtered_scores))
                        #logging.info(f"Successfully added attributes")
                        
                        # Convert to numpy for storage
                        obj_detections.to_numpy()
                        #logging.info(f"Successfully converted to numpy")
                        
                        # Store filtered scores for this object
                        self.scores_dict[object_name] = filtered_scores
                        
                        # Store detections for this object
                        all_detections[object_name] = obj_detections
                        
                        #logging.info(f'Found {len(filtered_scores)} detections for {object_name}')
                    except Exception as e:
                        #logging.error(f"Error filtering detections for {object_name}: {str(e)}")
                        #if 'keep_indices' in locals():
                            #logging.error(f"keep_indices shape: {keep_indices.shape}, values: {keep_indices}")
                        all_detections[object_name] = None
                else:
                    #logging.info(f'No detections above threshold for {object_name}')
                    all_detections[object_name] = None
            except Exception as e:
                #logging.error(f"Unexpected error in detection filtering for {object_name}: {str(e)}")
                all_detections[object_name] = None
        
        return all_detections, frame_name, rgb
    
    def process_frame(self, rgb_image, custom_conf_threshold=None, visualize_output=True):
        """
        Process a single frame with all objects and save results.
        
        Args:
            rgb_image: Either a path to an RGB image or a PIL Image object.
            custom_conf_threshold (float, optional): Custom confidence threshold.
            visualize_output (bool): Whether to visualize and save output.
            
        Returns:
            dict: Dictionary mapping object names to detection results.
        """
        try:
            # Run inference for all objects
            all_detections, frame_name, rgb = self.run_single_frame(rgb_image, custom_conf_threshold)
            
            # Create a boolean flag to track if at least one object was successfully processed
            any_object_processed = False
            
            # Save results for each object
            for object_name, detections in all_detections.items():
                if detections is not None:
                    # Set up session information
                    session_name = self.session_name if hasattr(self, 'session_name') and self.session_name else "unknown_session"
                    camera_view = self.camera_view if hasattr(self, 'camera_view') and self.camera_view else "unknown_camera"
                    frame_type = self.frame_type if hasattr(self, 'frame_type') and self.frame_type else "unknown_frame_type"
                    
                    # Check if output_dir already contains session information
                    output_parts = self.output_dir.split(os.sep)
                    path_has_session_info = (session_name in output_parts and 
                                            camera_view in output_parts and 
                                            frame_type in output_parts)
                    
                    # Set the base directory for this object
                    if path_has_session_info:
                        # Base output directory already includes session info
                        base_dir = self.output_dir
                    else:
                        # Need to add session info to path
                        base_dir = os.path.join(self.output_dir, session_name, camera_view, frame_type)
                    
                    # Create visualization directory (only do this once)
                    if visualize_output and not any_object_processed:
                        vis_output = os.path.join(base_dir, 'frames')
                        try:
                            os.makedirs(vis_output, exist_ok=True)
                            logging.info(f"Created visualization directory: {vis_output}")
                            
                            # Visualize combined results only once
                            if any(all_detections.values()):
                                try:
                                    vis_img = visualize(
                                        rgb, 
                                        [all_detections[obj_name] for obj_name in self.object_names], 
                                        self.object_names,
                                        save_path=f"{vis_output}/{frame_name}.png"
                                    )
                                    #logging.info(f"Saved visualization to {vis_output}/{frame_name}.png")
                                except Exception as e:
                                    logging.error(f"Error in visualization: {str(e)}")
                        except OSError as e:
                            logging.error(f"Failed to create visualization directory: {str(e)}")
                
                    # Save masks, bboxes and scores, handling errors properly
                    try:
                        # Save masks
                        self.save_masks(detections, object_name, frame_name, session_name, camera_view, frame_type)
                        # Save bounding boxes
                        self.save_bbox(detections, object_name, frame_name, session_name, camera_view, frame_type)
                        # Save scores
                        if object_name in self.scores_dict and self.scores_dict[object_name] is not None:
                            self.save_scores(self.scores_dict[object_name], object_name, frame_name, session_name, camera_view, frame_type)
                        else:
                            logging.warning(f"No scores to save for {object_name}")
                            
                        # Mark that at least one object was processed
                        any_object_processed = True
                    except (ValueError, IOError) as e:
                        logging.error(f"Error saving data for {object_name}: {str(e)}")
                        # Continue processing other objects instead of raising and stopping
                        continue
        except Exception as e:
            logging.error(f"Error running single frame inference: {str(e)}")
            return {}
        
        return all_detections
    
    def set_session_info(self, session_name, camera_view, frame_type):
        """
        Set the session information for directory structure.
        
        Args:
            session_name (str): Name of the session.
            camera_view (str): Camera view (e.g., cam_top, cam_side_r).
            frame_type (str): Frame type (L_frames or R_frames).
        """
        self.session_name = session_name
        self.camera_view = camera_view
        self.frame_type = frame_type
        logging.info(f"Set session info: {session_name}/{camera_view}/{frame_type}")

    
    def save_masks(self, detections, object_name, frame_info, session_name=None, camera_view=None, frame_type=None):
        """
        Save masks from detections as RLE (Run-Length Encoding) format.
        
        Args:
            detections (Detections): Detection object.
            object_name (str): Name of the object.
            frame_info (str): Frame information for directory naming.
            session_name (str, optional): Session name for directory structure.
            camera_view (str, optional): Camera view for directory structure.
            frame_type (str, optional): Frame type (L_frames/R_frames) for directory structure.
            
        Returns:
            bool: True if successfully saved.
            
        Raises:
            ValueError: If detections are None or masks can't be retrieved.
            IOError: If there's an error saving masks to files.
        """
        if detections is None:
            raise ValueError(f"Cannot save masks for {object_name}: detections are None")
            
        try:
            masks = getattr(detections, 'masks')
            scores = getattr(detections, 'scores')
        except AttributeError as e:
            raise ValueError(f"Cannot retrieve masks or scores for {object_name}: {str(e)}")
        
        # Ensure we have all required components
        if not (session_name and camera_view and frame_type):
            logging.warning(f"Missing session info: session={session_name}, camera={camera_view}, frame_type={frame_type}")
            session_name = session_name or "unknown_session"
            camera_view = camera_view or "unknown_camera"
            frame_type = frame_type or "unknown_frame_type"
        
        # Check if output_dir already contains the session/camera/frame_type structure
        output_parts = self.output_dir.split(os.sep)
        if (session_name in output_parts and 
            camera_view in output_parts and 
            frame_type in output_parts):
            # Path already contains structure
            output_dir = os.path.join(self.output_dir, object_name, 'masks', frame_info)
        else:
            # Add full path structure
            output_dir = os.path.join(self.output_dir, session_name, camera_view, frame_type, object_name, 'masks', frame_info)
        
        logging.info(f"Saving masks to: {output_dir}")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise IOError(f"Failed to create directory for masks: {output_dir}. Error: {str(e)}")
        
        # Dictionary to store all masks with their scores
        masks_data = {}
        saved_files = []
        
        try:
            for i, mask in enumerate(masks):
                # Convert mask to RLE format
                rle = mask_to_rle(mask)
                
                # Save individual RLE to a file
                rle_path = f"{output_dir}/mask_{i}.rle"
                with open(rle_path, 'w') as f:
                    json.dump({
                        'counts': rle['counts'],
                        'size': rle['size'],
                        'score': float(scores[i])
                    }, f, cls=NumpyEncoder)
                
                saved_files.append(rle_path)
                
                # Add to the combined dictionary
                masks_data[f"mask_{i}"] = {
                    'rle': rle,
                    'score': float(scores[i])
                }
            
            # Save all masks in a single JSON file
            all_masks_path = f"{output_dir}/all_masks.json"
            with open(all_masks_path, 'w') as f:
                json.dump(masks_data, f, cls=NumpyEncoder)
                
            saved_files.append(all_masks_path)
            
            # Verify files were saved correctly
            for file_path in saved_files:
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    raise IOError(f"Mask file was not saved properly: {file_path}")
                    
            logging.info(f"Successfully saved {len(masks)} masks for {object_name} in {output_dir}")
            return True
            
        except Exception as e:
            raise IOError(f"Error saving masks for {object_name}: {str(e)}")
    
    def save_bbox(self, detections, object_name, frame_info, session_name=None, camera_view=None, frame_type=None):
        """
        Save bounding boxes and object names as a JSON file.

        Args:
            detections (Detections): Detection object
            object_name (str): Name of the detected object
            frame_info (str): Frame identifier for filename
            session_name (str, optional): Session name for directory structure.
            camera_view (str, optional): Camera view for directory structure.
            frame_type (str, optional): Frame type (L_frames/R_frames) for directory structure.

        Returns:
            bool: True if save was successful
            
        Raises:
            ValueError: If detections are None or masks can't be retrieved.
            IOError: If there's an error saving bounding boxes to file.
        """
        if detections is None:
            raise ValueError(f"Cannot save bounding boxes for {object_name}: detections are None")
            
        try:
            masks = getattr(detections, 'masks')
        except AttributeError as e:
            raise ValueError(f"Cannot retrieve masks for {object_name}: {str(e)}")
        
        # Ensure we have all required components
        if not (session_name and camera_view and frame_type):
            logging.warning(f"Missing session info: session={session_name}, camera={camera_view}, frame_type={frame_type}")
            session_name = session_name or "unknown_session"
            camera_view = camera_view or "unknown_camera"
            frame_type = frame_type or "unknown_frame_type"
            
        # Check if output_dir already contains the session/camera/frame_type structure
        output_parts = self.output_dir.split(os.sep)
        if (session_name in output_parts and 
            camera_view in output_parts and 
            frame_type in output_parts):
            # Path already contains structure
            output_dir = os.path.join(self.output_dir, object_name, 'bbox')
        else:
            # Add full path structure
            output_dir = os.path.join(self.output_dir, session_name, camera_view, frame_type, object_name, 'bbox')
            
        logging.info(f"Saving bounding boxes to: {output_dir}")
            
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise IOError(f"Failed to create directory for bounding boxes: {output_dir}. Error: {str(e)}")

        bboxes = []
        try:
            for mask in masks:
                bbox = mask_to_bbox(mask)
                bboxes.append({
                    'label': object_name,
                    'bbox_modal': bbox
                })

            json_path = os.path.join(output_dir, f'{frame_info}.json')
            with open(json_path, 'w') as f:
                json.dump(bboxes, f, indent=2)
                
            # Verify file was saved correctly
            if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
                raise IOError(f"Bounding box file was not saved properly: {json_path}")
                
            logging.info(f"Successfully saved {len(bboxes)} bounding boxes for {object_name} in {json_path}")
            return True
            
        except Exception as e:
            raise IOError(f"Error saving bounding boxes for {object_name}: {str(e)}")
        
    def save_scores(self, scores, object_name, frame_info, session_name=None, camera_view=None, frame_type=None):
        """
        Save scores as individual text files.
        
        Args:
            scores (torch.Tensor): Tensor containing scores.
            object_name (str): Name of the object.
            frame_info (str): Frame information for directory naming.
            session_name (str, optional): Session name for directory structure.
            camera_view (str, optional): Camera view for directory structure.
            frame_type (str, optional): Frame type (L_frames/R_frames) for directory structure.
            
        Returns:
            bool: True if successfully saved.
            
        Raises:
            ValueError: If scores are None or empty.
            IOError: If there's an error saving scores to files.
        """
        if scores is None:
            raise ValueError(f"Cannot save scores for {object_name}: scores are None")
            
        if isinstance(scores, torch.Tensor) and scores.numel() == 0:
            raise ValueError(f"Cannot save scores for {object_name}: scores tensor is empty")
        
        # Ensure we have all required components
        if not (session_name and camera_view and frame_type):
            logging.warning(f"Missing session info: session={session_name}, camera={camera_view}, frame_type={frame_type}")
            session_name = session_name or "unknown_session"
            camera_view = camera_view or "unknown_camera"
            frame_type = frame_type or "unknown_frame_type"
            
        # Check if output_dir already contains the session/camera/frame_type structure
        output_parts = self.output_dir.split(os.sep)
        if (session_name in output_parts and 
            camera_view in output_parts and 
            frame_type in output_parts):
            # Path already contains structure
            output_dir = os.path.join(self.output_dir, object_name, 'scores', frame_info)
        else:
            # Add full path structure
            output_dir = os.path.join(self.output_dir, session_name, camera_view, frame_type, object_name, 'scores', frame_info)
            
        logging.info(f"Saving scores to: {output_dir}")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise IOError(f"Failed to create directory for scores: {output_dir}. Error: {str(e)}")
        
        saved_files = []
        try:
            saved_count = 0
            for i, score in enumerate(scores):
                if score < self.conf_threshold: 
                    continue
                
                score_path = f"{output_dir}/score_{i}.txt"
                with open(score_path, 'w') as f:
                    f.write(str(score.item()))
                    
                saved_files.append(score_path)
                saved_count += 1
            
            # Verify at least one file was saved if there were scores above threshold
            above_threshold = sum(1 for s in scores if s >= self.conf_threshold)
            if above_threshold > 0 and saved_count == 0:
                raise IOError(f"No score files were saved for {object_name} despite having {above_threshold} scores above threshold")
            
            # Verify files were saved correctly
            for file_path in saved_files:
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    raise IOError(f"Score file was not saved properly: {file_path}")
            
            logging.info(f"Successfully saved {saved_count} scores for {object_name} in {output_dir}")
            return True
            
        except Exception as e:
            raise IOError(f"Error saving scores for {object_name}: {str(e)}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run multi-object Cnos inference')
    parser.add_argument('--conf_threshold', type=float, default=0.01, help='Confidence threshold')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--templates_dir', type=str, required=True, help='Templates base directory')
    parser.add_argument('--objects', type=str, nargs='+', required=True, help='List of object names')
    parser.add_argument('--frame_folder', type=str, required=True, help='Folder containing frames')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--session_name', type=str, help='Session name for directory structure')
    parser.add_argument('--camera_view', type=str, help='Camera view for directory structure')
    parser.add_argument('--frame_type', type=str, help='Frame type (L_frames/R_frames) for directory structure')
    
    args = parser.parse_args()
    
    # Initialize the wrapper
    wrapper = MultiObjectInferenceWrapper(
        conf_threshold=args.conf_threshold,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        session_name=args.session_name,
        camera_view=args.camera_view,
        frame_type=args.frame_type
    )
    
    # Load templates for all objects
    template_paths = {}
    for object_name in args.objects:
        template_path = os.path.join(args.templates_dir, object_name)
        if os.path.exists(template_path):
            template_paths[object_name] = template_path
        else:
            logging.warning(f"Template path for {object_name} does not exist: {template_path}")
    
    wrapper.load_templates(template_paths)
    
    # Get all frames
    frames = [os.path.join(args.frame_folder, path) for path in os.listdir(args.frame_folder) 
              if path.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process all frames
    results = wrapper.process_multiple_frames(frames, num_workers=args.num_workers)
    
    logging.info(f"Processing complete. Results saved to {args.output_dir}")