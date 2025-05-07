"""
Flask web application for visualizing data from various loaders.
"""

import os
import sys
import numpy as np
import cv2
import base64
import json
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import loaders
from dataloaders.flow_loader import FlowLoader
from dataloaders.cnos_loader import CNOSLoader
from dataloaders.hamer_loader import HAMERLoader
from dataloaders.frame_loader import FrameLoader

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configuration
DATA_ROOT_DIR = "/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data"
FLOW_ROOT_DIR = "/nas/project_data/B1_Behavior/rush/kaan/old_method/processed_data"
DEFAULT_SESSION = "imi_session1_6"
DEFAULT_CAMERA = "cam_top"
DEFAULT_FRAME = 200
DEFAULT_FRAME_TYPE = "L_frames"  # For CNOS loader (L_frames or R_frames)

# Initialize loaders
flow_loader = FlowLoader(
    session_name=DEFAULT_SESSION,
    data_root_dir=FLOW_ROOT_DIR,
    feature_types=['direction', 'brightness']
)

cnos_loader = CNOSLoader(
    session_name=DEFAULT_SESSION,
    data_root_dir=DATA_ROOT_DIR
)

hamer_loader = HAMERLoader(
    session_name=DEFAULT_SESSION,
    data_root_dir=DATA_ROOT_DIR
)

frame_loader = FrameLoader(
    session_name=DEFAULT_SESSION,
    data_root_dir=DATA_ROOT_DIR
)

def create_mask(shape, x1, y1, x2, y2):
    """Create a rectangular mask for testing"""
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask

def image_to_base64(image_array):
    """Convert numpy array to base64 encoded image for display"""
    if image_array is None:
        return ""
    
    # Convert to uint8 if not already
    if image_array.dtype != np.uint8:
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
    
    # Handle different channel formats
    if len(image_array.shape) == 2:
        # Convert grayscale to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        # Convert RGBA to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    
    # Convert to PIL Image
    image = Image.fromarray(image_array)
    
    # Save to bytes buffer
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    
    # Convert to base64
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    
    return f"data:image/png;base64,{img_base64}"

@app.route('/api/flow', methods=['POST'])
def visualize_flow_data():
    """Get flow data visualization using object masks from CNOS"""
    try:
        # Get parameters from request
        camera_view = request.form.get('camera_view', 'cam_top')
        frame_idx = int(request.form.get('frame_idx', 0))
        object_name = request.form.get('object_name', '')
        feature_type = request.form.get('feature_type', 'direction')
        
        # Load flow data
        flow_data = flow_loader.load_features(camera_view=camera_view, frame_idx=frame_idx)
        
        if not flow_data or not flow_data.get('success', False):
            return {
                'success': False,
                'message': f'Failed to load flow data for camera {camera_view}, frame {frame_idx}'
            }
        
        # Get mask from CNOS data if object name is provided
        mask = None
        if object_name:
            # Load masks from CNOS loader for both frame types (to ensure we find the object)
            for frame_type in ['L_frames', 'R_frames']:
                cnos_data = cnos_loader.load_masks(camera_view=camera_view, frame_idx=frame_idx, load_masks=True)
                if cnos_data and cnos_data.get('success', False) and frame_type in cnos_data:
                    objects = cnos_data[frame_type].get('objects', {})
                    if object_name in objects and 'max_score_mask' in objects[object_name]:
                        mask = objects[object_name]['max_score_mask']
                        print(f"Using mask for {object_name} from {frame_type}")
                        break
            
            # Properly place the mask in full-sized frame using HAMER crop_bbox
            if mask is not None and 'flow_data' in flow_data and flow_data['flow_data'] is not None:
                # Double-check that flow_data['flow_data'] has a valid shape
                try:
                    flow_height, flow_width = flow_data['flow_data'].shape[:2]
                    
                    # Create a full-sized mask (same size as flow data)
                    full_mask = np.zeros((flow_height, flow_width), dtype=np.uint8)
                except (AttributeError, TypeError) as e:
                    print(f"Error getting flow dimensions: {e}")
                    return {
                        'success': False,
                        'message': f'Error processing flow data: {str(e)}'
                    }
                
                # Try to get crop_bbox from HAMER for both hand types
                crop_bbox = None
                hand_types = ['left', 'right']
                for hand_type in hand_types:
                    bbox = hamer_loader.get_hand_crop_bbox(camera_view, frame_idx, hand_type)
                    if bbox is not None:
                        crop_bbox = bbox
                        print(f"Using crop_bbox for object {object_name} from {hand_type} hand: {crop_bbox}")
                        break
                
                if crop_bbox is not None:
                    x1, y1, x2, y2 = crop_bbox
                    # Make sure the crop coordinates are valid for the frame size
                    x1 = max(0, min(x1, flow_width - 1))
                    y1 = max(0, min(y1, flow_height - 1))
                    x2 = max(x1 + 1, min(x2, flow_width))
                    y2 = max(y1 + 1, min(y2, flow_height))
                    
                    # Resize mask to the bounding box size
                    if mask is not None:
                        crop_width = x2 - x1
                        crop_height = y2 - y1
                        
                        # Resize mask to match crop dimensions if needed
                        if mask.shape[0] != crop_height or mask.shape[1] != crop_width:
                            print(f"Resizing mask to match crop dimensions: {mask.shape} -> {crop_height}x{crop_width}")
                            mask_resized = cv2.resize(mask.astype(np.uint8), (crop_width, crop_height), 
                                              interpolation=cv2.INTER_NEAREST)
                        else:
                            mask_resized = mask.astype(np.uint8)
                        
                        # Create a full-sized mask
                        full_mask = np.zeros((flow_height, flow_width), dtype=np.uint8)
                        
                        # Place the resized mask in the appropriate position
                        try:
                            full_mask[y1:y2, x1:x2] = mask_resized
                        except Exception as e:
                            print(f"Error placing mask: {e}")
                            print(f"Mask shape: {mask_resized.shape}, Target region: {y1}:{y2}, {x1}:{x2}")
                        
                        # Use the full-sized mask for further processing
                        mask = full_mask
                
                # For flow processing, if flow data is available, make sure mask dimensions match exactly
                if flow_data and flow_data.get('success') and flow_data.get('hsv_data') is not None and mask is not None:
                    # Get the HSV dimensions
                    hsv_height, hsv_width = flow_data['hsv_data'].shape[:2]
                    
                    # Check if mask dimensions match HSV dimensions
                    if mask.shape[0] != hsv_height or mask.shape[1] != hsv_width:
                        print(f"Final resize for flow processing: {mask.shape} -> {hsv_height}x{hsv_width}")
                        mask = cv2.resize(mask.astype(np.uint8), (hsv_width, hsv_height),
                                       interpolation=cv2.INTER_NEAREST)
                else:
                    # Fallback if no crop_bbox is available
                    print("No crop_bbox found, resizing mask to full frame")
                    mask = cv2.resize(mask.astype(np.uint8), (flow_width, flow_height), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Validate feature type
        if feature_type not in ['direction', 'brightness']:
            feature_type = 'all'
        
        # Make sure we have flow data for HSV dimensions
        if mask is not None and flow_data.get('hsv_data') is not None:
            # Get HSV dimensions to ensure mask matches exactly
            hsv_height, hsv_width = flow_data['hsv_data'].shape[:2]
            mask_height, mask_width = mask.shape[:2]
            
            # Final check to ensure mask and HSV dimensions match exactly
            if mask_height != hsv_height or mask_width != hsv_width:
                print(f"Final resize to match HSV dimensions: {mask.shape} -> {hsv_height}x{hsv_width}")
                mask = cv2.resize(mask.astype(np.uint8), (hsv_width, hsv_height), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Process flow features with desired feature type
        try:
            flow_result = flow_loader.process_flow_features(
                camera_view=camera_view,
                frame_idx=frame_idx,
                mask=mask,
                feature_type=feature_type,
                # Use median for more robust results when analyzing objects
                aggregation_method='median' if object_name else 'mean',
                # Use temporal information for more stable results
                temporal_window=3 if feature_type in ['all', 'direction'] else 1
            )
        except Exception as e:
            print(f"Error processing flow features: {e}")
            return {
                'success': False,
                'message': f'Error visualizing flow data: {str(e)}'
            }
        
        # Create visualization
        try:
            # Check if flow data is available
            if 'flow_data' not in flow_data or flow_data['flow_data'] is None:
                return {
                    'success': False,
                    'message': 'Flow data is not available'
                }
            
            # Create visualization
            flow_image = flow_data['flow_data'].copy()
            
            # Draw mask outline with safety checks
            if mask is not None:
                # Ensure mask and flow_image have compatible dimensions
                if mask.shape[:2] != flow_image.shape[:2]:
                    print(f"Resizing mask to match flow image: {mask.shape[:2]} -> {flow_image.shape[:2]}")
                    mask = cv2.resize(mask.astype(np.uint8), (flow_image.shape[1], flow_image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
                
                mask_outline = np.zeros_like(flow_image)
                mask_outline[mask > 0] = (0, 255, 0)
                flow_vis = cv2.addWeighted(flow_image, 1.0, mask_outline, 0.5, 0)
            else:
                flow_vis = flow_image
            
            # Add arrow for direction if moving
            if flow_result.get('success', False) and flow_result.get('is_moving', False):
                # Safety checks to make sure all required fields exist
                if all(k in flow_result for k in ['center_x', 'center_y', 'avg_dir_scaled']):
                    center_x = flow_result['center_x']
                    center_y = flow_result['center_y']
                    arrow_length = 50
                    dx = flow_result['avg_dir_scaled'][0] * arrow_length
                    dy = flow_result['avg_dir_scaled'][1] * arrow_length
                    cv2.arrowedLine(flow_vis, 
                                   (center_x, center_y), 
                                   (int(center_x + dx), int(center_y + dy)), 
                                   (0, 0, 255), 2)
            
            # Convert to base64 for display
            flow_vis_b64 = image_to_base64(flow_vis)
            
            return {
                'success': True,
                'visualization': flow_vis_b64,
                'flow_data': flow_result if flow_result.get('success', False) else None
            }
            
        except Exception as e:
            print(f"Error in flow visualization: {e}")
            return {
                'success': False,
                'message': f"Error visualizing flow data: {str(e)}"
            }
    
    except Exception as e:
        print(f"Error in visualize_flow_data: {e}")
        return {
            'success': False,
            'message': f"Error processing flow data: {str(e)}"
        }

@app.route('/api/cnos', methods=['POST'])
def visualize_cnos_data():
    """Get CNOS data visualization"""
    try:
        # Get parameters from request
        camera_view = request.form.get('camera_view', 'cam_top')
        frame_idx = int(request.form.get('frame_idx', 100))
        frame_type = request.form.get('frame_type', DEFAULT_FRAME_TYPE)
        
        # Load CNOS features with frame type
        features = cnos_loader.load_masks(camera_view=camera_view, frame_idx=frame_idx, load_masks=True)
        
        if not features or not features.get('success', False):
            return {
                'success': False,
                'message': f"Failed to load CNOS data for {camera_view}, frame {frame_idx}, frame type {frame_type}"
            }
        
        # Create visualization
        if 'rgb_frame' in features and features['rgb_frame'] is not None:
            vis_image = features['rgb_frame'].copy()
        else:
            # Create blank image if no RGB frame
            vis_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Dictionary to store masks for later use by flow features
        object_masks = {}
        
        # Add object segmentation masks if available
        if 'objects' in features and features['objects']:
            # Iterate through object names
            for obj_name, obj_data in features['objects'].items():
                if not obj_data or 'masks' not in obj_data or not obj_data['masks']:
                    continue
                    
                # Get the max score mask index
                max_score_idx = obj_data.get('max_score_mask_idx', 0)
                if max_score_idx >= len(obj_data['masks']):
                    max_score_idx = 0  # Fallback to first mask if index is invalid
                
                # Get the mask with highest score
                if len(obj_data['masks']) > max_score_idx:
                    mask = obj_data['masks'][max_score_idx]
                    score = obj_data['scores'][max_score_idx] if 'scores' in obj_data and len(obj_data['scores']) > max_score_idx else 0.0
                    
                    # Store mask for potential use with flow features
                    object_masks[obj_name] = mask
                    
                    # Create random color for visualization
                    color = list(np.random.random(size=3) * 255)
                    
                    # Apply mask with transparency
                    mask_viz = mask.astype(np.uint8)
                    colored_mask = np.zeros_like(vis_image)
                    colored_mask[mask_viz > 0] = color
                    
                    # Blend with original image
                    vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask, 0.5, 0)
                    
                    # Find centroid of mask for label placement
                    moments = cv2.moments(mask_viz)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        # Add object name and score text
                        label = f"{obj_name}: {score:.2f}"
                        cv2.putText(vis_image, label, (cx, cy), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert to base64 for display
        vis_image_b64 = image_to_base64(vis_image)
        
        return {
            'success': True,
            'visualization': vis_image_b64,
            'data': features
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Error visualizing CNOS data: {str(e)}"
        }

def visualize_hamer_data(camera_view, frame_idx, mask_coords=None):
    """Get HAMER data visualization"""
    try:
        # Load HAMER features
        features = hamer_loader.load_features(camera_view, frame_idx)
        
        if not features.get('success', False):
            return {
                'success': False,
                'message': f"Failed to load HAMER data for {camera_view}, frame {frame_idx}"
            }
        
        # Create visualization
        if 'rgb_frame' in features and features['rgb_frame'] is not None:
            vis_image = features['rgb_frame'].copy()
        else:
            # Create blank image if no RGB frame
            vis_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Dictionary to track hand bounding boxes for possible flow analysis
        hand_bboxes = {}
        
        # Process left hand
        if 'left_hand' in features and features['left_hand'].get('success', False):
            # Draw bounding box if available
            if features['left_hand'].get('crop_bbox') is not None:
                x1, y1, x2, y2 = features['left_hand']['crop_bbox']
                # Store bbox for possible flow analysis
                hand_bboxes['left_hand'] = (x1, y1, x2, y2)
                # Draw rectangle for crop_bbox
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add label
                cv2.putText(vis_image, "Left Hand", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw keypoints if available (vertices is the 3D mesh points)
            if features['left_hand'].get('vertices') is not None:
                vertices = features['left_hand']['vertices']
                # Draw a subset of vertices to avoid clutter
                step = max(1, len(vertices) // 50)  # Show ~50 points max
                for i in range(0, len(vertices), step):
                    if i < len(vertices):
                        vertex = vertices[i]
                        if len(vertex) >= 2:
                            x, y = int(vertex[0]), int(vertex[1])
                            cv2.circle(vis_image, (x, y), 2, (255, 0, 0), -1)
        
        # Process right hand
        if 'right_hand' in features and features['right_hand'].get('success', False):
            # Draw bounding box if available
            if features['right_hand'].get('crop_bbox') is not None:
                x1, y1, x2, y2 = features['right_hand']['crop_bbox']
                # Store bbox for possible flow analysis
                hand_bboxes['right_hand'] = (x1, y1, x2, y2)
                # Draw rectangle for crop_bbox
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Add label
                cv2.putText(vis_image, "Right Hand", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw keypoints if available
            if features['right_hand'].get('vertices') is not None:
                vertices = features['right_hand']['vertices']
                # Draw a subset of vertices to avoid clutter
                step = max(1, len(vertices) // 50)  # Show ~50 points max
                for i in range(0, len(vertices), step):
                    if i < len(vertices):
                        vertex = vertices[i]
                        if len(vertex) >= 2:
                            x, y = int(vertex[0]), int(vertex[1])
                            cv2.circle(vis_image, (x, y), 2, (255, 0, 255), -1)
        
        # Convert to base64 for display
        vis_image_b64 = image_to_base64(vis_image)
        
        return {
            'success': True,
            'visualization': vis_image_b64,
            'data': features
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Error visualizing HAMER data: {str(e)}"
        }

def visualize_combined_data(camera_view, frame_idx, object_name=None, frame_type=DEFAULT_FRAME_TYPE, feature_type='all'):
    """Combine visualizations from all loaders"""
    # 1. First load all raw data to integrate properly
    frame_data = frame_loader.load_features(camera_view=camera_view, frame_idx=frame_idx)
    #print(frame_data.keys())

    cnos_data = cnos_loader.load_masks(camera_view=camera_view, frame_idx=frame_idx, load_masks=True)
    #print(cnos_data.keys())
    #print('-----------------')
    hamer_data = hamer_loader.load_features(camera_view=camera_view, frame_idx=frame_idx)
    #print(hamer_data.keys())
    #print('-----')
    flow_data = flow_loader.load_features(camera_view=camera_view, frame_idx=frame_idx)
    #print(flow_data['success'])
    #print('-----')
    # Validate feature type
    if feature_type not in ['all', 'direction', 'brightness']:
        feature_type = 'all'
    
    success = flow_data.get('success', False)
    #print(success)
    if not success:
        return {
            'success': False,
            'message': "Failed to load one or more data sources"
        }
    
    # 2. Create base visualization with RGB frame
    combined_vis = frame_loader.load_frame(camera_view=camera_view, frame_idx=frame_idx)

    print(combined_vis.shape)
    
    # 3. Process and add object masks from CNOS
    object_flow_results = {}
    object_mask = None
    
    # First check if we have specified object_name to focus on
    if object_name and frame_type in cnos_data and 'objects' in cnos_data[frame_type]:
        objects = cnos_data[frame_type].get('objects', {})
        if object_name in objects and 'max_score_mask' in objects[object_name]:
            # Extract the mask for the selected object
            object_mask = objects[object_name]['max_score_mask']
            print(f"Using mask for {object_name} from {frame_type}")
            
            # Properly place the mask in full-sized frame using HAMER crop_bbox
            if 'flow_data' in flow_data and flow_data['flow_data'] is not None:
                flow_height, flow_width = flow_data['flow_data'].shape[:2]
                
                # Create a full-sized mask (same size as flow data)
                full_mask = np.zeros((flow_height, flow_width), dtype=np.uint8)
                
                # Try to get crop_bbox from HAMER for both hand types
                crop_bbox = None
                hand_types = ['left', 'right']
                for hand_type in hand_types:
                    bbox = hamer_loader.get_hand_crop_bbox(camera_view, frame_idx, hand_type)
                    if bbox is not None:
                        crop_bbox = bbox
                        print(f"Using crop_bbox from {hand_type} hand: {crop_bbox}")
                        break
                
                if crop_bbox is not None:
                    x1, y1, x2, y2 = crop_bbox
                    # Make sure the crop coordinates are valid for the frame size
                    x1 = max(0, min(x1, flow_width - 1))
                    y1 = max(0, min(y1, flow_height - 1))
                    x2 = max(x1 + 1, min(x2, flow_width))
                    y2 = max(y1 + 1, min(y2, flow_height))
                    
                    # Resize mask to the crop dimensions if needed
                    crop_width = x2 - x1
                    crop_height = y2 - y1
                    
                    if object_mask.shape[0] != crop_height or object_mask.shape[1] != crop_width:
                        print(f"Resizing mask from {object_mask.shape} to {crop_height}x{crop_width}")
                        object_mask_resized = cv2.resize(object_mask.astype(np.uint8), (crop_width, crop_height), 
                                            interpolation=cv2.INTER_NEAREST)
                    else:
                        object_mask_resized = object_mask.astype(np.uint8)
                    
                    # Place mask in the correct position in the full frame
                    try:
                        full_mask[y1:y2, x1:x2] = object_mask_resized[:y2-y1, :x2-x1]  # Handle potential boundary issues
                    except ValueError as e:
                        print(f"Error placing mask in full frame: {e}")
                        print(f"Mask shape: {object_mask_resized.shape}, Full mask region: {y1}:{y2}, {x1}:{x2}")
                    
                    # Replace the original mask with the properly positioned full mask
                    object_mask = full_mask
                else:
                    # Fallback if no crop_bbox is available
                    print("No crop_bbox found, resizing mask to full frame")
                    object_mask = cv2.resize(object_mask.astype(np.uint8), (flow_width, flow_height), 
                                            interpolation=cv2.INTER_NEAREST)
            
            # Process flow features with the selected object mask and feature type
            if flow_data.get('success', False):
                # Make sure we have flow data for HSV dimensions
                if flow_data.get('hsv_data') is not None:
                    try:
                        # Get HSV dimensions to ensure mask matches exactly
                        hsv_height, hsv_width = flow_data['hsv_data'].shape[:2]
                        mask_height, mask_width = object_mask.shape[:2]
                        
                        # Final check to ensure mask and HSV dimensions match exactly
                        if mask_height != hsv_height or mask_width != hsv_width:
                            print(f"Resizing combined view mask to match HSV dimensions: {object_mask.shape} -> {hsv_height}x{hsv_width}")
                            object_mask = cv2.resize(object_mask.astype(np.uint8), (hsv_width, hsv_height), 
                                              interpolation=cv2.INTER_NEAREST)
                    except (AttributeError, TypeError) as e:
                        print(f"Error resizing mask for HSV: {e}")
                        # If there's an error, we'll still try to process with the original mask
                
                try:
                    # Make sure flow_data has properly loaded before proceeding
                    if flow_data.get('flow_data') is not None and object_mask is not None:
                        # Double-check that the mask shape matches the flow data shape
                        flow_height, flow_width = flow_data['flow_data'].shape[:2]
                        if object_mask.shape[0] != flow_height or object_mask.shape[1] != flow_width:
                            print(f"Final mask resize to match flow dimensions: {object_mask.shape} -> {flow_height}x{flow_width}")
                            object_mask = cv2.resize(object_mask.astype(np.uint8), (flow_width, flow_height),
                                                  interpolation=cv2.INTER_NEAREST)
                    
                    flow_result = flow_loader.process_flow_features(
                        camera_view=camera_view,
                        frame_idx=frame_idx,
                        mask=object_mask,
                        feature_type=feature_type,
                        aggregation_method='median',  # Use median for more robust results
                        temporal_window=3 if feature_type in ['all', 'direction'] else 1  # Use temporal information
                    )
                except Exception as e:
                    print(f"Error processing flow features in combined view: {e}")
                    flow_result = {'success': False, 'error': str(e)}
                if flow_result:
                    object_flow_results[object_name] = flow_result
    
    # Process all visible objects in the scene
    if frame_type in cnos_data and 'objects' in cnos_data[frame_type]:
        obj_counter = 0
        for obj_name, obj_data in cnos_data[frame_type]['objects'].items():
            if not obj_data or 'max_score_mask' not in obj_data:
                continue
            
            # Get the mask for this object
            obj_mask = obj_data['max_score_mask']
            obj_score = obj_data.get('max_score', 0.0)
            
            # Generate a consistent color from object name
            obj_counter += 1
            color_seed = hash(obj_name) % 100 / 100.0
            color = [
                (color_seed * 255) % 255,
                ((color_seed * 3.7) * 255) % 255,
                ((color_seed * 7.9) * 255) % 255
            ]
            
            # Create a proper full-sized mask using HAMER crop_bbox
            img_height, img_width = combined_vis.shape[:2]
            
            # Convert the mask to uint8 format for processing
            mask_viz = obj_mask.astype(np.uint8)
            
            # Create a full-sized mask (same size as the combined_vis)
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            # Try to get crop_bbox from HAMER for both hand types
            crop_bbox = None
            hand_types = ['left', 'right'] if frame_type == 'L_frames' else ['right', 'left']
            for hand_type in hand_types:
                bbox = hamer_loader.get_hand_crop_bbox(camera_view, frame_idx, hand_type)
                if bbox is not None:
                    crop_bbox = bbox
                    print(f"Using crop_bbox for object {obj_name} from {hand_type} hand: {crop_bbox}")
                    break
                    
            if crop_bbox is not None:
                x1, y1, x2, y2 = crop_bbox
                # Make sure coordinates are valid for the frame size
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(x1 + 1, min(x2, img_width))
                y2 = max(y1 + 1, min(y2, img_height))
                
                # Resize mask to crop dimensions if needed
                crop_width = x2 - x1
                crop_height = y2 - y1
                
                if mask_viz.shape[0] != crop_height or mask_viz.shape[1] != crop_width:
                    print(f"Resizing mask from {mask_viz.shape} to {crop_height}x{crop_width}")
                    mask_viz_resized = cv2.resize(mask_viz, (crop_width, crop_height), 
                                         interpolation=cv2.INTER_NEAREST)
                else:
                    mask_viz_resized = mask_viz
                
                # Place mask in the correct position in full frame
                try:
                    full_mask[y1:y2, x1:x2] = mask_viz_resized[:y2-y1, :x2-x1]  # Handle boundary issues
                except ValueError as e:
                    print(f"Error placing mask for {obj_name} in full frame: {e}")
                    print(f"Mask shape: {mask_viz_resized.shape}, Region: {y1}:{y2}, {x1}:{x2}")
                    # Fallback - resize to full image
                    full_mask = cv2.resize(mask_viz, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            else:
                # Fallback if no crop_bbox is available
                print(f"No crop_bbox found for {obj_name}, resizing to full frame")
                full_mask = cv2.resize(mask_viz, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            
            # Apply color to the mask
            colored_mask = np.zeros_like(combined_vis)
            colored_mask[full_mask > 0] = color
            
            # Process flow for this object's mask
            if flow_data['success']:
                obj_flow = flow_loader.process_flow_features(
                    camera_view=camera_view,
                    frame_idx=frame_idx,
                    mask=obj_mask,
                    feature_type='direction',
                    aggregation_method='mean'
                )
                
                if obj_flow.get('success', False):
                    object_flow_results[obj_name] = obj_flow
                    
                    # If object is moving, draw flow direction arrow
                    if obj_flow.get('is_moving', False):
                        # Find centroid of mask
                        moments = cv2.moments(mask_viz)
                        if moments["m00"] != 0:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                            
                            # Draw arrow for flow direction
                            arrow_length = 30 * obj_flow['avg_len'] * 20  # Scale based on motion magnitude
                            dx = obj_flow['avg_dir_scaled'][0]
                            dy = obj_flow['avg_dir_scaled'][1]
                            cv2.arrowedLine(combined_vis, 
                                          (cx, cy), 
                                          (int(cx + dx), int(cy + dy)), 
                                          (255, 255, 0), 2)
                
                # Blend with combined image
                combined_vis = cv2.addWeighted(combined_vis, 1.0, colored_mask, 0.3, 0)
                
                # Add object name and score
                moments = cv2.moments(mask_viz)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    
                    # Add text for object and its motion if calculated
                    obj_text = f"{obj_name}: {obj_score:.2f}"
                    if obj_name in object_flow_results:
                        obj_flow = object_flow_results[obj_name]
                        motion_text = "(Moving)" if obj_flow.get('is_moving', False) else "(Static)"
                        obj_text = f"{obj_text} {motion_text}"
                        
                    cv2.putText(combined_vis, obj_text, (cx, cy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 4. Add hand bounding boxes from HAMER
    if hamer_data and 'left_hand' in hamer_data and hamer_data['left_hand'].get('success', False):
        if hamer_data['left_hand'].get('crop_bbox') is not None:
            x1, y1, x2, y2 = hamer_data['left_hand']['crop_bbox']
            cv2.rectangle(combined_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(combined_vis, "Left Hand", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # If there's flow data, analyze flow in the hand region
            if flow_data['success']:
                hand_mask = np.zeros_like(flow_data['flow_data'][:, :, 0], dtype=bool)
                hand_mask[y1:y2, x1:x2] = True
                
                hand_flow = flow_loader.process_flow_features(
                    camera_view=camera_view,
                    frame_idx=frame_idx,
                    mask=hand_mask,
                    feature_type='direction',
                    aggregation_method='mean'
                )
                
                if hand_flow.get('success', False) and hand_flow.get('is_moving', False):
                    # Draw flow direction for hand
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    arrow_length = 30
                    dx = hand_flow['avg_dir_scaled'][0]
                    dy = hand_flow['avg_dir_scaled'][1]
                    cv2.arrowedLine(combined_vis, 
                                   (center_x, center_y), 
                                   (int(center_x + dx), int(center_y + dy)), 
                                   (0, 255, 255), 2)
    
    if hamer_data and 'right_hand' in hamer_data and hamer_data['right_hand'].get('success', False):
        if hamer_data['right_hand'].get('crop_bbox') is not None:
            x1, y1, x2, y2 = hamer_data['right_hand']['crop_bbox']
            cv2.rectangle(combined_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(combined_vis, "Right Hand", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # If there's flow data, analyze flow in the hand region
            if flow_data['success']:
                hand_mask = np.zeros_like(flow_data['flow_data'][:, :, 0], dtype=bool)
                hand_mask[y1:y2, x1:x2] = True
                
                hand_flow = flow_loader.process_flow_features(
                    camera_view=camera_view,
                    frame_idx=frame_idx,
                    mask=hand_mask,
                    feature_type='direction',
                    aggregation_method='mean'
                )
                
                if hand_flow.get('success', False) and hand_flow.get('is_moving', False):
                    # Draw flow direction for hand
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    arrow_length = 30
                    dx = hand_flow['avg_dir_scaled'][0]
                    dy = hand_flow['avg_dir_scaled'][1]
                    cv2.arrowedLine(combined_vis, 
                                   (center_x, center_y), 
                                   (int(center_x + dx), int(center_y + dy)), 
                                   (255, 0, 255), 2)
    
    # 5. Add custom region flow analysis if requested
    if object_name:
        # Load masks from CNOS loader for both frame types (to ensure we find the object)
        for frame_type in ['L_frames', 'R_frames']:
            cnos_data = cnos_loader.load_masks(camera_view=camera_view, frame_idx=frame_idx, load_masks=True)
            if cnos_data and cnos_data.get('success', False) and frame_type in cnos_data:
                objects = cnos_data[frame_type].get('objects', {})
                if object_name in objects and 'max_score_mask' in objects[object_name]:
                    mask = objects[object_name]['max_score_mask']
                    print(f"Using mask for {object_name} from {frame_type}")
                    
                    # Process flow for this object's mask
                    if flow_data['success']:
                        obj_flow = flow_loader.process_flow_features(
                            camera_view=camera_view,
                            frame_idx=frame_idx,
                            mask=mask,
                            feature_type='direction',
                            aggregation_method='mean'
                        )
                        
                        if obj_flow.get('success', False):
                            object_flow_results[object_name] = obj_flow
                            
                            # If object is moving, draw flow direction arrow
                            if obj_flow.get('is_moving', False):
                                # Find centroid of mask
                                moments = cv2.moments(mask.astype(np.uint8))
                                if moments["m00"] != 0:
                                    cx = int(moments["m10"] / moments["m00"])
                                    cy = int(moments["m01"] / moments["m00"])
                                    
                                    # Draw arrow for flow direction
                                    arrow_length = 30 * obj_flow['avg_len'] * 20  # Scale based on motion magnitude
                                    dx = obj_flow['avg_dir_scaled'][0]
                                    dy = obj_flow['avg_dir_scaled'][1]
                                    cv2.arrowedLine(combined_vis, 
                                                   (cx, cy), 
                                                   (int(cx + dx), int(cy + dy)), 
                                                   (255, 255, 0), 2)
                    
                    # Draw mask outline
                    mask_outline = np.zeros_like(combined_vis)
                    mask_outline[mask > 0] = (0, 255, 0)
                    combined_vis = cv2.addWeighted(combined_vis, 1.0, mask_outline, 0.5, 0)
                    
                    # Add object name and score
                    moments = cv2.moments(mask.astype(np.uint8))
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        
                        # Add text for object and its motion if calculated
                        obj_text = f"{object_name}"
                        if object_name in object_flow_results:
                            obj_flow = object_flow_results[object_name]
                            motion_text = "(Moving)" if obj_flow.get('is_moving', False) else "(Static)"
                            obj_text = f"{obj_text} {motion_text}"
                            
                        cv2.putText(combined_vis, obj_text, (cx, cy), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 6. Add title and frame information
    cv2.putText(combined_vis, f"Combined Analysis - {camera_view} - Frame {frame_idx}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Convert to base64 for display
    combined_vis_b64 = image_to_base64(combined_vis)
    
    return {
        'success': True,
        'visualization': combined_vis_b64,
        'object_flow_results': object_flow_results
    }

@app.route('/')
def index():
    # Get valid frames from HAMERLoader
    valid_frames = hamer_loader.get_valid_frame_idx()
 
    # Get the initial camera view
    initial_camera = DEFAULT_CAMERA
    
    # Get valid frames for the initial camera
    initial_valid_frames = {
        'left': valid_frames.get(initial_camera, {}).get('left', []),
        'right': valid_frames.get(initial_camera, {}).get('right', [])
    }
    
    # Find a default frame that exists in both left and right hands if possible
    common_frames = set(initial_valid_frames['left']).intersection(set(initial_valid_frames['right']))
    default_frame = next(iter(common_frames), None) if common_frames else (
        initial_valid_frames['left'][0] if initial_valid_frames['left'] else (
            initial_valid_frames['right'][0] if initial_valid_frames['right'] else DEFAULT_FRAME
        )
    )
    
    return render_template(
        'index.html',
        valid_frames=valid_frames,
        default_frame=default_frame
    )

@app.route('/api/flow', methods=['POST'])
def get_flow_visualization():
    """API endpoint for flow visualization"""
    camera_view = request.form.get('camera_view', DEFAULT_CAMERA)
    frame_idx = int(request.form.get('frame_idx', DEFAULT_FRAME))
    object_name = request.form.get('object_name', '')
    feature_type = request.form.get('feature_type', 'all')
    
    # Check if frame is valid
    valid_frames = hamer_loader.get_valid_frame_idx()
    left_valid = frame_idx in valid_frames.get(camera_view, {}).get('left', [])
    right_valid = frame_idx in valid_frames.get(camera_view, {}).get('right', [])
    
    if not (left_valid or right_valid):
        return jsonify({
            'success': False,
            'message': f'Frame {frame_idx} is not a valid frame for camera {camera_view}. Please select a valid frame.'
        })
    
    result = visualize_flow_data(
        camera_view=camera_view, 
        frame_idx=frame_idx, 
        object_name=object_name if object_name else None,
        feature_type=feature_type
    )
    return jsonify(result)

@app.route('/api/cnos', methods=['POST'])
def get_cnos_visualization():
    """API endpoint for CNOS visualization"""
    camera_view = request.form.get('camera_view', DEFAULT_CAMERA)
    frame_idx = int(request.form.get('frame_idx', DEFAULT_FRAME))
    frame_type = request.form.get('frame_type', DEFAULT_FRAME_TYPE)
    mask_coords_str = request.form.get('mask_coords', '')
    
    # Check if frame is valid
    valid_frames = hamer_loader.get_valid_frame_idx()
    left_valid = frame_idx in valid_frames.get(camera_view, {}).get('left', [])
    right_valid = frame_idx in valid_frames.get(camera_view, {}).get('right', [])
    
    # For CNOS, we match the frame_type (L_frames/R_frames) with the corresponding hand
    if frame_type == 'L_frames' and not left_valid:
        return jsonify({
            'success': False,
            'message': f'Frame {frame_idx} is not a valid frame for left hand in camera {camera_view}.'
        })
    elif frame_type == 'R_frames' and not right_valid:
        return jsonify({
            'success': False,
            'message': f'Frame {frame_idx} is not a valid frame for right hand in camera {camera_view}.'
        })
    
    mask_coords = None
    if mask_coords_str:
        try:
            coords = mask_coords_str.split(',')
            if len(coords) == 4:
                mask_coords = [int(c) for c in coords]
        except ValueError:
            pass
    
    result = visualize_cnos_data(camera_view, frame_idx, frame_type, mask_coords)
    return jsonify(result)

@app.route('/api/hamer', methods=['POST'])
def get_hamer_visualization():
    """API endpoint for HAMER visualization"""
    data = request.json
    camera_view = data.get('camera_view', DEFAULT_CAMERA)
    frame_idx = int(data.get('frame_idx', DEFAULT_FRAME))
    
    return jsonify(visualize_hamer_data(camera_view, frame_idx))

@app.route('/api/combined', methods=['POST'])
def get_combined_visualization():
    """API endpoint for combined visualization"""
    camera_view = request.form.get('camera_view', DEFAULT_CAMERA)
    frame_idx = int(request.form.get('frame_idx', DEFAULT_FRAME))
    frame_type = request.form.get('frame_type', DEFAULT_FRAME_TYPE)
    object_name = request.form.get('object_name', '')
    feature_type = request.form.get('feature_type', 'all')
    
    # Check if frame is valid
    valid_frames = hamer_loader.get_valid_frame_idx()
    left_valid = frame_idx in valid_frames.get(camera_view, {}).get('left', [])
    right_valid = frame_idx in valid_frames.get(camera_view, {}).get('right', [])
    
    if not (left_valid or right_valid):
        return jsonify({
            'success': False,
            'message': f'Frame {frame_idx} is not a valid frame for camera {camera_view}. Please select a valid frame.'
        })
    
    # For combined visualization, ensure the frame matches the requested hand type
    if frame_type == 'L_frames' and not left_valid:
        return jsonify({
            'success': False,
            'message': f'Frame {frame_idx} is not a valid frame for left hand in camera {camera_view}.'
        })
    elif frame_type == 'R_frames' and not right_valid:
        return jsonify({
            'success': False,
            'message': f'Frame {frame_idx} is not a valid frame for right hand in camera {camera_view}.'
        })
    
    # Pass feature_type to the combined visualization
    result = visualize_combined_data(
        camera_view=camera_view, 
        frame_idx=frame_idx, 
        object_name=object_name if object_name else None, 
        frame_type=frame_type,
        feature_type=feature_type
    )
    return jsonify(result)

@app.route('/api/valid_frames', methods=['POST'])
def get_valid_frames():
    camera_view = request.form.get('camera_view', DEFAULT_CAMERA)
    valid_frames = hamer_loader.get_valid_frame_idx()
    
    camera_frames = valid_frames.get(camera_view, {'left': [], 'right': []})
    return jsonify(camera_frames)

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
