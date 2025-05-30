"""
Routes for motion-filtered data visualization
This module provides a unified approach using only MotionFilteredLoader
"""

import os
import cv2
import numpy as np
from flask import request, jsonify, current_app
from object_interaction_detection.dataloaders.motion_filtered_loader import MotionFilteredLoader, visualize_object_masks

def register_motion_routes(app, motion_loader):
    """Register all routes using the MotionFilteredLoader"""
    
    @app.route('/')
    def index():
        """Render main visualization page"""
        # Get valid frame indices from the motion loader
        valid_frames = motion_loader.combined_loader.cnos_loader.hamer_loader.get_valid_frame_idx()
        
        # Get default frame
        default_frame = 200
        if valid_frames.get('cam_top', {}).get('left', []):
            default_frame = valid_frames['cam_top']['left'][0]
        
        return current_app.render_template('index.html', 
                                           valid_frames=valid_frames, 
                                           default_frame=default_frame)
    
    @app.route('/api/valid-frames')
    def get_valid_frames():
        """Get valid frames for a camera view"""
        camera_view = request.args.get('camera_view', 'cam_top')
        valid_frames = motion_loader.combined_loader.cnos_loader.hamer_loader.get_valid_frame_idx()
        
        return jsonify({
            'success': True,
            'camera_view': camera_view,
            'valid_frames': valid_frames
        })
    
    @app.route('/api/flow-visualization', methods=['POST'])
    def get_flow_visualization():
        """API endpoint for flow visualization"""
        try:
            # Get parameters from request
            camera_view = request.form.get('camera_view', 'cam_top')
            frame_idx = int(request.form.get('frame_idx', 0))
            object_name = request.form.get('object_name', '')
            
            # Load features from motion loader
            features = motion_loader.load_features(camera_view=camera_view, frame_idx=frame_idx)
            
            if not features or not features.get('combined', {}).get('merged', {}).get('success', False):
                return jsonify({
                    'success': False,
                    'message': f'Failed to load data for camera {camera_view}, frame {frame_idx}'
                })
            
            # Get flow frame
            flow_frame = motion_loader.flow_loader._load_flow_frame(camera_view=camera_view, frame_idx=frame_idx)
            
            if flow_frame is None:
                return jsonify({
                    'success': False,
                    'message': f'Failed to load flow frame for camera {camera_view}, frame {frame_idx}'
                })
            
            # Get the merged mask and object ID map
            merged_mask = features['combined']['merged']['mask']
            object_id_map = features['combined']['merged']['object_id_map']
            
            # Process specific object if provided
            flow_info = {}
            
            if object_name and object_name in object_id_map:
                # Create mask for specific object
                obj_id = object_id_map[object_name]
                obj_mask = (merged_mask == obj_id)
                
                # Get activeness
                activeness = motion_loader.flow_loader.process_flow_in_mask_active_area(
                    camera_view=camera_view,
                    frame_idx=frame_idx,
                    mask=obj_mask
                )
                
                # Store flow info
                flow_info = {
                    'is_moving': activeness > motion_loader.motion_threshold,
                    'avg_dir': activeness,  # Simplification - could be expanded
                    'avg_len': activeness
                }
                
                # Create visualization with focus on this object
                activeness_map = {object_name: activeness}
                visualization = visualize_object_masks(
                    merged_mask, flow_frame, object_id_map, 
                    motion_loader.combined_loader.object_colors,
                    activeness_map=activeness_map, min_activeness=0.0  # Show all objects
                )
            else:
                # Get activeness for all objects
                activeness_result = motion_loader.get_activeness(camera_view, frame_idx)
                
                if activeness_result['success']:
                    activeness_map = activeness_result['activeness']
                    
                    # Create visualization
                    visualization = visualize_object_masks(
                        merged_mask, flow_frame, object_id_map, 
                        motion_loader.combined_loader.object_colors,
                        activeness_map=activeness_map, min_activeness=0.0  # Show all objects
                    )
                else:
                    # Fallback to simple visualization
                    visualization = visualize_object_masks(
                        merged_mask, flow_frame, object_id_map, 
                        motion_loader.combined_loader.object_colors
                    )
            
            # Convert visualization to base64
            from utils.image_utils import image_to_base64
            vis_base64 = image_to_base64(visualization)
            
            return jsonify({
                'success': True,
                'image': vis_base64,
                'flow_info': flow_info
            })
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Error processing flow data: {str(e)}'
            })
    
    @app.route('/api/cnos-visualization', methods=['POST'])
    def get_cnos_visualization():
        """API endpoint for CNOS visualization"""
        try:
            # Get parameters from request
            camera_view = request.form.get('camera_view', 'cam_top')
            frame_idx = int(request.form.get('frame_idx', 0))
            frame_type = request.form.get('frame_type', 'L_frames')
            
            # Load features from motion loader
            features = motion_loader.load_features(camera_view=camera_view, frame_idx=frame_idx)
            
            if not features or not features.get('combined', {}).get('merged', {}).get('success', False):
                return jsonify({
                    'success': False,
                    'message': f'Failed to load data for camera {camera_view}, frame {frame_idx}'
                })
            
            # Get the original frame
            frame = motion_loader._load_original_frame(camera_view=camera_view, frame_idx=frame_idx)
            
            if frame is None:
                return jsonify({
                    'success': False,
                    'message': f'Failed to load original frame for camera {camera_view}, frame {frame_idx}'
                })
            
            # Get object data from the combined loader's raw data
            cnos_data = motion_loader.combined_loader.cnos_loader.load_masks(
                camera_view=camera_view, frame_idx=frame_idx, load_masks=True
            )
            
            if not cnos_data or not cnos_data.get('success', False) or frame_type not in cnos_data:
                return jsonify({
                    'success': False,
                    'message': f'No {frame_type} data available for this frame'
                })
            
            # Get objects for the frame type
            objects = cnos_data[frame_type].get('objects', {})
            
            # Get the merged mask and object ID map
            merged_mask = features['combined']['merged']['mask']
            object_id_map = features['combined']['merged']['object_id_map']
            
            # Create visualization
            visualization = visualize_object_masks(
                merged_mask, frame, object_id_map, 
                motion_loader.combined_loader.object_colors
            )
            
            # Convert visualization to base64
            from utils.image_utils import image_to_base64
            vis_base64 = image_to_base64(visualization)
            
            # Extract object info for response
            objects_info = {}
            for obj_name, obj_data in objects.items():
                if 'max_score' in obj_data:
                    objects_info[obj_name] = {
                        'max_score': obj_data['max_score']
                    }
            
            return jsonify({
                'success': True,
                'image': vis_base64,
                'objects': objects_info
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Error processing CNOS data: {str(e)}'
            })
    
    @app.route('/api/hamer-visualization', methods=['POST'])
    def get_hamer_visualization():
        """API endpoint for HAMER visualization"""
        try:
            # Get parameters from request
            camera_view = request.form.get('camera_view', 'cam_top')
            frame_idx = int(request.form.get('frame_idx', 0))
            
            # Load features from motion loader
            features = motion_loader.load_features(camera_view=camera_view, frame_idx=frame_idx)
            
            if not features or not features.get('combined', {}).get('merged', {}).get('success', False):
                return jsonify({
                    'success': False,
                    'message': f'Failed to load data for camera {camera_view}, frame {frame_idx}'
                })
            
            # Get the original frame
            frame = motion_loader._load_original_frame(camera_view=camera_view, frame_idx=frame_idx)
            
            if frame is None:
                return jsonify({
                    'success': False,
                    'message': f'Failed to load original frame for camera {camera_view}, frame {frame_idx}'
                })
            
            # Get HAMER data from the combined loader's raw data
            hamer_data = motion_loader.combined_loader.cnos_loader.hamer_loader.load_features(
                camera_view=camera_view, frame_idx=frame_idx
            )
            
            # Create a visualization with hand meshes
            # Note: This is a simplified version - you may need to enhance this based on your needs
            visualization = frame.copy()
            
            # Draw hand bounding boxes if available
            if hamer_data.get('left_hand', {}).get('success', False):
                left_bbox = hamer_data['left_hand'].get('bbox')
                if left_bbox is not None:
                    x1, y1, x2, y2 = left_bbox
                    cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(visualization, "Left Hand", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if hamer_data.get('right_hand', {}).get('success', False):
                right_bbox = hamer_data['right_hand'].get('bbox')
                if right_bbox is not None:
                    x1, y1, x2, y2 = right_bbox
                    cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(visualization, "Right Hand", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convert visualization to base64
            from utils.image_utils import image_to_base64
            vis_base64 = image_to_base64(visualization)
            
            return jsonify({
                'success': True,
                'image': vis_base64,
                'hamer_info': {
                    'left_hand': {'success': hamer_data.get('left_hand', {}).get('success', False)},
                    'right_hand': {'success': hamer_data.get('right_hand', {}).get('success', False)}
                }
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Error processing HAMER data: {str(e)}'
            })
    
    @app.route('/api/combined-visualization', methods=['POST'])
    def get_combined_visualization():
        """API endpoint for combined visualization with motion filtering"""
        try:
            # Get parameters from request
            camera_view = request.form.get('camera_view', 'cam_top')
            frame_idx = int(request.form.get('frame_idx', 0))
            object_name = request.form.get('object_name', '')
            min_activeness = float(request.form.get('min_activeness', 0.25))
            
            # Load features from motion loader
            features = motion_loader.load_features(camera_view=camera_view, frame_idx=frame_idx)
            
            if not features or not features.get('motion_filtered', {}).get('success', False):
                return jsonify({
                    'success': False,
                    'message': f'Failed to load motion-filtered data for camera {camera_view}, frame {frame_idx}'
                })
            
            # Get the original frame
            frame = motion_loader._load_original_frame(camera_view=camera_view, frame_idx=frame_idx)
            
            if frame is None:
                return jsonify({
                    'success': False,
                    'message': f'Failed to load original frame for camera {camera_view}, frame {frame_idx}'
                })
            
            # Get the filtered mask and object ID map
            filtered_mask = features['motion_filtered']['mask']
            object_id_map = features['motion_filtered']['object_id_map']
            
            # Get activeness for all objects
            activeness_result = motion_loader.get_activeness(camera_view, frame_idx)
            
            if not activeness_result.get('success', False):
                return jsonify({
                    'success': False,
                    'message': 'Failed to calculate object activeness'
                })
            
            activeness_map = activeness_result['activeness']
            
            # Filter by specific object if provided
            if object_name and object_name in object_id_map:
                focused_activeness_map = {obj: score for obj, score in activeness_map.items() 
                                         if obj == object_name or score >= min_activeness}
            else:
                focused_activeness_map = activeness_map
            
            # Create visualization
            visualization = visualize_object_masks(
                filtered_mask, frame, object_id_map, 
                motion_loader.combined_loader.object_colors,
                activeness_map=focused_activeness_map, 
                min_activeness=min_activeness
            )
            
            # Convert visualization to base64
            from utils.image_utils import image_to_base64
            vis_base64 = image_to_base64(visualization)
            
            # Prepare active objects information
            active_objects = []
            for obj_name, activeness in activeness_map.items():
                active_objects.append({
                    'name': obj_name,
                    'activity': activeness,
                    'is_active': activeness >= min_activeness
                })
            
            # Sort by activeness (most active first)
            active_objects.sort(key=lambda x: x['activity'], reverse=True)
            
            return jsonify({
                'success': True,
                'image': vis_base64,
                'combined_info': {
                    'active_objects': active_objects,
                    'threshold': min_activeness
                }
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Error processing combined data: {str(e)}'
            })
