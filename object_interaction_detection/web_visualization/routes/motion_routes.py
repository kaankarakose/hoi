"""
Routes for motion-filtered data visualization
This module provides a unified approach using only MotionFilteredLoader
"""

import os
import cv2
import numpy as np
from flask import request, jsonify, current_app, render_template
from object_interaction_detection.dataloaders.motion_filtered_loader import MotionFilteredLoader, visualize_object_masks

def register_motion_routes(app, motion_loader, detection_manager):
    """Register all routes using the MotionFilteredLoader"""
    
    @app.route('/')
    def index():
        """Render main visualization page"""
        # Get valid frame indices from the motion loader
        valid_frames = motion_loader.combined_loader.cnos_loader.hamer_loader.get_valid_frame_idx()
        # Get default frame
        default_frame = 200
        if valid_frames:
            default_frame = valid_frames[0]
        # Get available sessions
        available_sessions = app.get_available_sessions()
        return render_template('index.html', 
                                valid_frames=valid_frames, 
                                default_frame=default_frame,
                                available_sessions=available_sessions,
                                default_session=motion_loader.session_name,
                                default_camera=motion_loader.camera_view)
    
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
            object_name = request.form.get('object_name', 'AMF1')
            
            # Load features from motion loader
            features = motion_loader.load_features(camera_view=camera_view, frame_idx=frame_idx)
            
            if not features or not features.get('combined', {}).get('merged', {}).get('success', False):
                return jsonify({
                    'success': False,
                    'message': f'Failed to load data for camera {camera_view}, frame {frame_idx}'
                })
            
            # Get flow frame
            flow_frame = motion_loader.flow_loader._load_flow_frame(camera_view=camera_view, frame_idx=frame_idx)
            #flow_frame = motion_loader._load_original_frame(camera_view=camera_view, frame_idx=frame_idx)[:, :, ::-1]  # Convert BGR to RGB
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
                print(object_name, object_id_map[object_name])
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
                'flow_info': flow_info,
                'object_name': object_name if object_name else None
            })
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Error processing flow data: {str(e)}'
            })
    @app.route('/api/available-sessions')
    def get_available_sessions_route():
        """Get list of available sessions"""
        try:
            # Get available sessions
            sessions = app.get_available_sessions()
            
            return jsonify({
                'success': True,
                'sessions': sessions,
                'current_session': motion_loader.session_name
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Error fetching available sessions: {str(e)}'
            })
            
    @app.route('/api/evaluation-data', methods=['POST'])
    def get_evaluation_data():
        """API endpoint to get evaluation data for a specific session and camera view"""
        try:
            # Get parameters from request
            session_name = request.form.get('session_name', motion_loader.session_name)
            camera_view = request.form.get('camera_view', motion_loader.camera_view)
            
            # Update the detection manager with the new session and camera
            detection_manager.session_name = session_name
            detection_manager.camera_view = camera_view
            
            # Load detection data using the DetectionManager
            detection_data = detection_manager.load_data(session_name, camera_view)
            
            if not detection_data:
                return jsonify({
                    'success': False,
                    'message': f'No detection data found for session {session_name}, camera {camera_view}'
                })
            
            # Convert to JSON response
            return jsonify({
                'success': True,
                'data': detection_data.to_dict()
            })
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Error loading evaluation data: {str(e)}'
            })
    