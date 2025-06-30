import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from src.model.utils import Detections
from scipy.ndimage import distance_transform_edt

class MergedDetection:
    """Class to store information about merged masks"""
    def __init__(self, mask, component_indices, score, descriptor):
        """Initialize MergedDetection with validation for mask dimensions
        
        Args:
            mask: The merged binary mask (must be 2D)
            component_indices: Indices of the original detections that were merged
            score: Similarity score with the reference object
            descriptor: Feature descriptor for the merged mask
            
        Raises:
            ValueError: If mask is not 2D
        """
        # Validate mask dimensions
        if isinstance(mask, torch.Tensor):
            if len(mask.shape) != 2:
                raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
        elif isinstance(mask, np.ndarray):
            if len(mask.shape) != 2:
                raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
        else:
            raise ValueError(f"Mask must be either torch.Tensor or np.ndarray, got {type(mask)}")
            
        self.mask = mask
        self.component_indices = component_indices
        self.score = score
        self.descriptor = descriptor


class ObjectAwareMaskMerger:
    def __init__(self, descriptor_model, similarity_metric, score_threshold=0.41):
        """
        Initialize the mask merger.
        
        Args:
            descriptor_model: Model to extract descriptors from masks
            similarity_metric: Function to compute similarity between descriptors
            score_threshold: Minimum score improvement required to merge masks
            iou_threshold: Minimum IoU required to consider merging masks
        """
        self.descriptor_model = descriptor_model
        self.similarity_metric = similarity_metric
        self.score_threshold = score_threshold
    


    def check_if_neighbors(self, mask1, mask2, distance_threshold=20) -> bool:
        """
        Check if two binary masks are neighbors or close to each other.
        Returns True if any pixel in mask1 is within distance_threshold pixels of any pixel in mask2.
        
        Parameters:
        - mask1, mask2: Binary masks (PyTorch tensors or NumPy arrays)
        - distance_threshold: Maximum distance (in pixels) to consider masks as neighbors
        
        Returns:
        - bool: True if masks are neighbors, False otherwise
        """
        # Validate mask dimensions
        # if isinstance(mask1, torch.Tensor) and len(mask1.shape) != 2:
        #     raise ValueError(f"1D mask detected in check_if_neighbors, mask1 shape: {mask1.shape}")
        # if isinstance(mask2, torch.Tensor) and len(mask2.shape) != 2:
        #     raise ValueError(f"1D mask detected in check_if_neighbors, mask2 shape: {mask2.shape}")
        # if isinstance(mask1, np.ndarray) and len(mask1.shape) != 2:
        #     raise ValueError(f"1D mask detected in check_if_neighbors, mask1 shape: {mask1.shape}")
        # if isinstance(mask2, np.ndarray) and len(mask2.shape) != 2:
        #     raise ValueError(f"1D mask detected in check_if_neighbors, mask2 shape: {mask2.shape}")

        # Convert to PyTorch tensors if they are NumPy arrays
        if isinstance(mask1, np.ndarray):
            mask1 = torch.from_numpy(mask1)
        if isinstance(mask2, np.ndarray):
            mask2 = torch.from_numpy(mask2)
            
        # Move tensors to CPU if they're on CUDA
        if mask1.device.type == 'cuda':
            mask1 = mask1.cpu()
        if mask2.device.type == 'cuda':
            mask2 = mask2.cpu()
            
        # Final validation after conversions
        if len(mask1.shape) != 2:
            raise ValueError(f"1D mask after conversion in check_if_neighbors, mask1 shape: {mask1.shape}")
        if len(mask2.shape) != 2:
            raise ValueError(f"1D mask after conversion in check_if_neighbors, mask2 shape: {mask2.shape}")
        
        # Convert to numpy for distance transform
        mask1_np = mask1.numpy().astype(np.uint8)
        mask2_np = mask2.numpy().astype(np.uint8)
        

        
        # Get distance from every pixel to the nearest True pixel in mask1
        dist_from_mask1 = distance_transform_edt(~mask1_np)
        
        # Check if any True pixel in mask2 is within the threshold distance of mask1
        neighboring_pixels = mask2_np & (dist_from_mask1 <= distance_threshold)
        
        return neighboring_pixels.any()
    
    def merge_masks(self, 
                   image: np.ndarray, 
                   detections: Detections, 
                   ref_features: torch.Tensor,
                   conf_threshold: float) -> List[MergedDetection]:
        """
        Merge masks that likely belong to the same object.
        
        Args:
            image: RGB image as numpy array
            detections: List of detection objects with masks
            ref_features: Reference features to compare against [N_objects, D]
            
        Returns:
            List of MergedDetection objects
        """
        # Extract masks from detections
        masks = [det.masks for det in detections]
        num_masks = len(masks)
        
        if num_masks == 0:
            return None
        
        print('Number of masks:', num_masks)
        
        # Generate descriptors for individual masks (this is likely a bottleneck)
        descriptors = self.descriptor_model.forward(image, detections)  # [N_masks, DinoV2]

        # Compute initial similarity scores with reference features - vectorized operation
        # Shape: [N_masks, N_objects]
        # Reshape tensors once instead of using None index multiple times
        descriptors_reshaped = descriptors.unsqueeze(1)  # [N_masks, 1, D]
        ref_features_reshaped = ref_features.unsqueeze(0)  # [1, N_objects, D]
        initial_scores = self.similarity_metric(descriptors_reshaped, ref_features_reshaped).squeeze()

        # Find best match for each mask
        best_match_indices = torch.argmax(initial_scores, dim=1)  # [N_masks]
        best_match_scores = torch.gather(
            initial_scores, 1, 
            best_match_indices.unsqueeze(1)
        ).squeeze()  # [N_masks]
        
        # Pre-filter by confidence threshold - speeds up by reducing candidates early
        valid_mask_indices = torch.where(best_match_scores >= conf_threshold / 2)[0].cpu().numpy()
        
        if len(valid_mask_indices) == 0:
            return None
            
        # Only create MergedDetections for masks that pass the threshold
        merged_detections = []
        for i in valid_mask_indices:
            current_mask = masks[i]
            merged_detections.append(MergedDetection(
                mask=current_mask,
                component_indices=[i],
                score=best_match_scores[i].item(),
                descriptor=descriptors[i].detach().cpu().numpy()
            ))

        # Early return if only one mask
        if len(merged_detections) == 1:
            return merged_detections
        
        # Cache for neighbor checks to avoid repeated computations
        neighbor_cache = {}
        
        # Try merging masks
        improved = True
        iteration = 0
        max_iterations = min(num_masks * 2, 20)  # Avoid excessive iterations, cap at 20
        
        # Track which masks have been merged
        merged_mask_indices = set()
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Prepare potential merges to evaluate
            potential_merges = []
            
            # First pass: Find all potential merges without computing descriptors yet
            for i in range(len(merged_detections)):
                if i in merged_mask_indices:
                    continue
                    
                merged_i = merged_detections[i]
                
                for j in range(i + 1, len(merged_detections)):  # Start from i+1 to avoid duplicate checks
                    if j in merged_mask_indices:
                        continue
                        
                    merged_j = merged_detections[j]
                    
                    # Check if masks are neighbors using cached results if available
                    cache_key = tuple(sorted([id(merged_i.mask), id(merged_j.mask)]))
                    if cache_key in neighbor_cache:
                        are_neighbors = neighbor_cache[cache_key]
                    else:
                        are_neighbors = self.check_if_neighbors(merged_i.mask, merged_j.mask)
                        neighbor_cache[cache_key] = are_neighbors
                    
                    if not are_neighbors:
                        continue
                    
                    # Add to potential merges list
                    potential_merges.append((i, j))
            
            # Second pass: Evaluate all potential merges in batch if possible
            best_merges = {}  # i -> (j, improvement, merged_mask, merged_descriptor)
            
            for i, j in potential_merges:
                merged_i = merged_detections[i]
                merged_j = merged_detections[j]
                
                # Create a merged mask
                if isinstance(merged_i.mask, torch.Tensor):
                    # Handle PyTorch tensors
                    mask1 = merged_i.mask.cpu() if merged_i.mask.device.type == 'cuda' else merged_i.mask
                    mask2 = merged_j.mask.cpu() if merged_j.mask.device.type == 'cuda' else merged_j.mask
                    
                    # Fast path: use logical_or directly
                    merged_mask = torch.logical_or(mask1, mask2)
                
                # Skip additional validation to improve speed
                
                # Create a temporary detection with the merged mask
                merged_detection = {
                    "mask": merged_mask,
                    "bbox": self._mask_to_bbox(merged_mask)
                }
                
                # Generate descriptor for the merged mask
                merged_descriptor = self.descriptor_model.forward(
                    np.array(image), merged_detection
                )[0]  # [D]
                
                # Compute similarity score for the merged mask - avoid reshaping multiple times
                merged_descriptor_reshaped = merged_descriptor.unsqueeze(0).unsqueeze(1)
                merged_score = self.similarity_metric(
                    merged_descriptor_reshaped,
                    ref_features_reshaped
                ).squeeze()
                
                # Get score for the best matching object
                best_obj_idx = best_match_indices[merged_i.component_indices[0]].item()
                merged_obj_score = merged_score[best_obj_idx].item()
                
                # Calculate improvement
                improvement = merged_obj_score - max(merged_i.score, merged_j.score)
                
                # Track the best merge for each mask i
                if improvement > 0:  # Only consider positive improvements
                    if i not in best_merges or improvement > best_merges[i][1]:
                        best_merges[i] = (j, improvement, merged_mask, merged_descriptor)
            
            # Apply all the best merges
            # Sort by improvement to prioritize most beneficial merges
            sorted_merges = sorted(best_merges.items(), key=lambda x: x[1][1], reverse=True)
            
            for i, (j, improvement, merged_mask, merged_descriptor) in sorted_merges:
                # Skip if either mask has already been merged
                if i in merged_mask_indices or j in merged_mask_indices:
                    continue
                
                improved = True
                merged_i = merged_detections[i]
                merged_j = merged_detections[j]
                
                # Create a new merged detection
                new_merged = MergedDetection(
                    mask=merged_mask,
                    component_indices=merged_i.component_indices + merged_j.component_indices,
                    score=merged_i.score + improvement,
                    descriptor=merged_descriptor.detach().cpu().numpy()
                )
                
                # Mark the merged detections as processed
                merged_mask_indices.add(j)
                
                # Replace the current detection with the merged one
                merged_detections[i] = new_merged
            
            # Remove the merged detections - more efficient than creating a new list
            merged_detections = [md for idx, md in enumerate(merged_detections) 
                               if idx not in merged_mask_indices]
            merged_mask_indices = set()  # Reset for next iteration
            
            # Early termination if no improvements found
            if not improved:
                break
        
        return merged_detections


    def merge_masks_v2(self, 
                   image: np.ndarray, 
                   detections: Detections, 
                   ref_features: torch.Tensor,
                   conf_threshold: float) -> List[MergedDetection]:
        """
        Merge masks that likely belong to the same object.
        
        Args:
            image: RGB image as numpy array
            detections: List of detection objects with masks
            ref_features: Reference features to compare against [N_images, D]
            
        Returns:
            List of MergedDetection objects
        """
        # Extract masks from detections

        # Extract masks and ensure they're 2D
        masks = []
        for det in detections:
            mask = det.masks
            # # Check if mask is 1D and reshape it if necessary
            # if len(mask.shape) == 1:
            #     raise ValueError("1D mask in merge masks")
            masks.append(mask)

        
        num_masks = len(masks)
        print('Number of masks:',num_masks)
        # Generate descriptors for individual masks
        descriptors = self.descriptor_model.forward(image, detections)  # [N_masks, DinoV2]

        # Compute initial similarity scores with reference features
        # Shape: [N_masks, N_objects]
        initial_scores = self.similarity_metric(
            descriptors[:, None, :],  ## N_masks , 1 , D
            ref_features[None, :, :] ## 1 , 512, D -
        ).squeeze()

        
        # Find best match for each mask
        best_match_indices = torch.argmax(initial_scores, dim=1)  # [N_masks]
        best_match_scores = torch.gather(
            initial_scores, 1, 
            best_match_indices.unsqueeze(1)
        ).squeeze()  # [N_masks]
        
        # Initialize merged detections with individual masks
        merged_detections = []
        for i in range(num_masks):
            # Apply confidence threshold filter
            if best_match_scores[i].item() < conf_threshold / 2:
               continue # Skip masks with low confidence scores
            
            # Debug check to ensure mask is 2D before creating MergedDetection
            current_mask = masks[i]
                                
            merged_detections.append(MergedDetection(
                mask=current_mask,
                component_indices=[i],
                score=best_match_scores[i].item(),
                descriptor=descriptors[i].detach().cpu().numpy()
            ))

        #No mask detected
        if merged_detections == []:
            return None
        #1 mask only
        if len(merged_detections) == 1:
            return merged_detections
        
        # Track which masks have been merged
        merged_mask_indices = set()
        
        # Sort merged detections by score
        merged_detections.sort(key=lambda x: x.score, reverse=True)

        # Try merging masks
        improved = True
        iteration = 0
        max_iterations =min(num_masks * 2, 20)  # Avoid infinite loops
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Iterate through all pairs of current merged detections
            for i in range(len(merged_detections)):
                if i in merged_mask_indices:
                    continue
                    
                merged_i = merged_detections[i]
                best_improvement = 0
                best_merge_idx = -1
                best_merged_mask = None
                best_merged_descriptor = None
                
                for j in range(len(merged_detections)):
                    if j == i or j in merged_mask_indices:
                        continue
                        
                    merged_j = merged_detections[j]
                    

                    # Check if masks are neighbors or close to each other
                    are_neighbors = self.check_if_neighbors(merged_i.mask, merged_j.mask)
                    if not are_neighbors:
                        continue
                    
                    # Create a merged mask
                    if isinstance(merged_i.mask, torch.Tensor):
                        # Handle PyTorch tensors
                        mask1 = merged_i.mask.cpu() if merged_i.mask.device.type == 'cuda' else merged_i.mask
                        mask2 = merged_j.mask.cpu() if merged_j.mask.device.type == 'cuda' else merged_j.mask
                        
                        # Validate mask dimensions before merging
                        # if len(mask1.shape) != 2:
                        #     raise ValueError(f"1D mask detected before merging, mask1 shape: {mask1.shape}")
                        # if len(mask2.shape) != 2:
                        #     raise ValueError(f"1D mask detected before merging, mask2 shape: {mask2.shape}")
                            
                        merged_mask = torch.logical_or(mask1, mask2)
                        
                        # if isinstance(merged_mask, torch.Tensor) and (merged_mask.shape[0] == 1 or merged_mask.shape[1] == 1):
                        #     raise ValueError('merged mask has 1 in dimension!')
                        # # Validate merged mask dimensions
                        # if len(merged_mask.shape) != 2:
                        #     raise ValueError(f"1D mask created after merging, merged_mask shape: {merged_mask.shape}")
       
                    
                    # Create a temporary detection with the merged mask
                    merged_detection = {
                        "mask": merged_mask,
                        "bbox": self._mask_to_bbox(merged_mask)
                    }
                    
                    # Generate descriptor for the merged mask
                    merged_descriptor = self.descriptor_model.forward(
                        np.array(image), merged_detection
                    )[0]  # [D]
                    
                    # Compute similarity score for the merged mask
                    merged_score = self.similarity_metric(
                        merged_descriptor.unsqueeze(0)[:, None, :],
                        ref_features[None, :, :]
                    ).squeeze()
                    
                    # Get score for the best matching object
                    best_obj_idx = best_match_indices[merged_i.component_indices[0]].item()
                    merged_obj_score = merged_score[best_obj_idx].item()
                    
                    # Calculate improvement
                    improvement = merged_obj_score - max(merged_i.score, merged_j.score)
                    
                    if improvement > best_improvement: #and improvement > self.score_threshold:
                        best_improvement = improvement
                        best_merge_idx = j
                        best_merged_mask = merged_mask
                        best_merged_descriptor = merged_descriptor
                        
                
                # If we found a good merge, perform it
                if best_merge_idx >= 0:
                    improved = True
                    merged_j = merged_detections[best_merge_idx]
                    
                    # Create a new merged detection
                    new_merged = MergedDetection(
                        mask=best_merged_mask,
                        component_indices=merged_i.component_indices + merged_j.component_indices,
                        score=merged_i.score + best_improvement,
                        descriptor=best_merged_descriptor.detach().cpu().numpy()
                    )
                    
                    # Mark the merged detections as processed
                    merged_mask_indices.add(best_merge_idx)
                    
                    # Replace the current detection with the merged one
                    merged_detections[i] = new_merged
            
            # Remove the merged detections
            merged_detections = [md for idx, md in enumerate(merged_detections) 
                               if idx not in merged_mask_indices]
            merged_mask_indices = set()  # Reset for next iteration
        
        return merged_detections
    
    def _mask_to_bbox(self, mask) -> Tuple[int, int, int, int]:
        """
        Convert binary mask to bounding box (x1, y1, x2, y2).
        Handles PyTorch tensors.
        """        
        # Move to CPU if on CUDA
        if mask.device.type == 'cuda':
            mask = mask.cpu()
            
        # Check if mask is 1D and print a diagnostic message
        if len(mask.shape) == 1:
            print(f"ERROR: 1D mask detected in _mask_to_bbox with shape {mask.shape}")
            raise ValueError(f"1D mask detected in _mask_to_bbox")
            
        # Use PyTorch operations
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        
        rows_indices = torch.where(rows)[0]
        cols_indices = torch.where(cols)[0]
        
        if len(rows_indices) == 0 or len(cols_indices) == 0:
            return 0, 0, 1, 1  # Default small bbox for empty masks
            
        y1, y2 = int(rows_indices[0].item()), int(rows_indices[-1].item())
        x1, x2 = int(cols_indices[0].item()), int(cols_indices[-1].item())
        
        return torch.from_numpy(np.array([x1, y1, x2, y2]))


class MaskMergingPipeline:
    def __init__(self, segmentor_model, descriptor_model, ref_feats, similarity_metric, conf_threshold):
        """
        Initialize the pipeline.
        
        Args:
            segmentor_model: Model to generate initial masks (e.g., SAM)
            descriptor_model: Model to extract descriptors from masks
            ref_feats: Reference features for known objects
            similarity_metric: Function to compute similarity
        """
        self.segmentor_model = segmentor_model
        self.descriptor_model = descriptor_model
        self.ref_feats = ref_feats
        self.similarity_metric = similarity_metric
        self.mask_merger = ObjectAwareMaskMerger(
            descriptor_model=descriptor_model,
            similarity_metric=similarity_metric
        )
        self.conf_threshold = conf_threshold

    
    def process_image(self, image: np.ndarray) -> Tuple[List[MergedDetection], torch.Tensor]:
        """
        Process an image to detect and identify objects.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            merged_detections: List of merged detections
            scores: Similarity scores for each merged detection
        """
        print('\nGenerating masks with segmentor...')
        detections = self.segmentor_model.generate_masks(image)

        detections = Detections(detections)
        
        # # Validate that masks from segmentor are 2D
        # if hasattr(detections, 'masks'):
        #     masks = detections.masks
        #     if isinstance(masks, list):
        #         for i, mask in enumerate(masks):
        #             if len(mask.shape) != 2:
        #                 raise ValueError(f"1D mask detected from segmentor at index {i}, shape: {mask.shape}")

        #             if mask.shape[0] == 1 or mask.shape[1] == 1:
        #                 raise ValueError(f"Found single mask with shape {mask.shape} at {i}")

        #     elif isinstance(masks, torch.Tensor) or isinstance(masks, np.ndarray):
        #         if len(masks.shape) < 3:  # Should be batch_size x height x width
        #             raise ValueError(f"Invalid mask dimensions from segmentor, shape: {masks.shape}")

        print('\nMerging masks that belong to the same object...')
        merged_detections = self.mask_merger.merge_masks(
            image, detections, self.ref_feats, self.conf_threshold
        )
        # No mask detected
        if merged_detections == []:
            return None, None
        
        # Convert merged detection descriptors to tensor
        merged_descriptors = torch.tensor(
            np.stack([md.descriptor for md in merged_detections]), 
            device=self.ref_feats.device
        )
        
        print('\nCalculating final similarity scores...')
        scores = self.similarity_metric(
            merged_descriptors[:, None, :], 
            self.ref_feats[None, :, :]
        )


        boxes = []
        for md in merged_detections:
            # Convert numpy mask to tensor for _mask_to_bbox
            mask_tensor = md.mask
            bbox = self.mask_merger._mask_to_bbox(mask_tensor)
            boxes.append(bbox)
        
        # Stack boxes into a tensor
        boxes = torch.stack(boxes) if boxes else torch.zeros((0, 4), dtype=torch.long)
        
        # Stack masks into a tensor, ensuring they're all 2D
        processed_masks = []
        for md in merged_detections:
            mask = md.mask
            # Convert to CPU tensor if needed
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu()
            # Ensure mask is 2D
            if len(mask.shape) == 1:
                raise ValueError("1D mask detected in process_image")
            processed_masks.append(mask)
        
        if processed_masks:
            masks = torch.stack(processed_masks).float()
        else:
            # Handle empty case
            masks = torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.float)

        # Extract scores from merged_detections
        md_scores = torch.tensor([md.score for md in merged_detections], device=scores.device)
        
        # Create a dictionary with all the detection data
        detection_data = {
            'masks': masks.to(scores.device),
            'boxes': boxes.to(scores.device)
        }
        

        # Convert to Detections object
        detections = Detections(detection_data)
        
        return detections, scores