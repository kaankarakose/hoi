/**
 * API Service for Hand-Object Interaction Visualization
 * Handles all API calls to the backend
 */

class ApiService {
    /**
     * Get flow visualization data
     * @param {string} cameraView - Camera view (cam_top, cam_side_l, cam_side_r)
     * @param {number} frameIdx - Frame index
     * @param {string} objectName - Object name to mask (optional)
     * @param {string} featureType - Flow feature type (direction, brightness, all)
     * @returns {Promise} - Promise with the API response
     */
    static getFlowVisualization(cameraView, frameIdx, objectName = '', featureType = 'all') {
        const formData = new FormData();
        formData.append('camera_view', cameraView);
        formData.append('frame_idx', frameIdx);
        formData.append('object_name', objectName);
        formData.append('feature_type', featureType);
        
        return fetch('/api/flow-visualization', {
            method: 'POST',
            body: formData
        }).then(response => response.json());
    }
    
    /**
     * Get CNOS visualization data
     * @param {string} cameraView - Camera view (cam_top, cam_side_l, cam_side_r)
     * @param {number} frameIdx - Frame index
     * @param {string} frameType - Frame type (L_frames, R_frames)
     * @returns {Promise} - Promise with the API response
     */
    static getCNOSVisualization(cameraView, frameIdx, frameType = 'L_frames') {
        const formData = new FormData();
        formData.append('camera_view', cameraView);
        formData.append('frame_idx', frameIdx);
        formData.append('frame_type', frameType);
        
        return fetch('/api/cnos-visualization', {
            method: 'POST',
            body: formData
        }).then(response => response.json());
    }
    
    /**
     * Get HAMER visualization data
     * @param {string} cameraView - Camera view (cam_top, cam_side_l, cam_side_r)
     * @param {number} frameIdx - Frame index
     * @returns {Promise} - Promise with the API response
     */
    static getHAMERVisualization(cameraView, frameIdx) {
        const formData = new FormData();
        formData.append('camera_view', cameraView);
        formData.append('frame_idx', frameIdx);
        
        return fetch('/api/hamer-visualization', {
            method: 'POST',
            body: formData
        }).then(response => response.json());
    }
    
    /**
     * Get combined visualization data
     * @param {string} cameraView - Camera view (cam_top, cam_side_l, cam_side_r)
     * @param {number} frameIdx - Frame index
     * @param {string} objectName - Object name to focus on (optional)
     * @param {string} frameType - Frame type (L_frames, R_frames)
     * @param {string} featureType - Flow feature type (direction, brightness, all)
     * @returns {Promise} - Promise with the API response
     */
    static getCombinedVisualization(cameraView, frameIdx, objectName = '', frameType = 'L_frames', featureType = 'all') {
        const formData = new FormData();
        formData.append('camera_view', cameraView);
        formData.append('frame_idx', frameIdx);
        formData.append('object_name', objectName);
        formData.append('frame_type', frameType);
        formData.append('feature_type', featureType);
        
        return fetch('/api/combined-visualization', {
            method: 'POST',
            body: formData
        }).then(response => response.json());
    }
    
    /**
     * Get valid frames for a given camera view
     * @param {string} cameraView - Camera view (cam_top, cam_side_l, cam_side_r)
     * @returns {Promise} - Promise with the API response
     */
    static getValidFrames(cameraView) {
        return fetch(`/api/valid-frames?camera_view=${cameraView}`)
            .then(response => response.json());
    }
}
