/**
 * API Service for Hand-Object Interaction Visualization
 * Handles all API calls to the backend
 */

class ApiService {
    /**
     * Get flow visualization data
     * @param {string} sessionName - Name of the session
     * @param {string} cameraView - Camera view (cam_top, cam_side_l, cam_side_r)
     * @param {number} frameIdx - Frame index
     * @param {string} objectName - Object name to mask (optional)
     * @param {string} featureType - Flow feature type (direction, brightness, all)
     * @returns {Promise} - Promise with the API response
     */


    static getFlowVisualization(sessionName, cameraView, frameIdx, objectName = '', featureType = 'all') {
        const formData = new FormData();
        formData.append('session_name', sessionName); // Assuming a default session name, can be parameterized
        formData.append('camera_view', cameraView);
        formData.append('frame_idx', frameIdx);
        formData.append('object_name', objectName);
        formData.append('feature_type', featureType);
        
        return fetch('/api/flow-visualization', {
            method: 'POST',
            body: formData
        }).then(response => response.json());
    }
    
    static getCnosVisualization(sessionName, cameraView, frameIdx, objectName = '')
    {
        const formData = new FormData();
        formData.append('session_name', sessionName);
        formData.append('camera_view', cameraView);
        formData.append('frame_idx', frameIdx);
        formData.append('object_name', objectName);
        
        return fetch('/api/cnos-visualization', {
            method: 'POST',
            body: formData
        }).then(response => response.json());

    }

    static getEvaluationData(sessionName, cameraView) {
        /**
         * Get evaluation data for a given session and camera view
         * @param {string} sessionName - Name of the session
         * @param {string} cameraView - Camera view (cam_top, cam_side_l, cam_side_r)
         * @returns {Promise} - Promise with the API response
         */

        const formData = new FormData();
        formData.append('session_name', sessionName);
        formData.append('camera_view', cameraView);

        
        console.log('Sessiong', sessionName, 'camera',cameraView);
        return fetch('/api/evaluation-data', {
            method: 'POST',
            body: formData
        }).then(response => response.json());
    }
    
    static getEvaluationVisualization(sessionName, cameraView) {
        /**
         * Get evaluation data for a given session and camera view
         * @param {string} sessionName - Name of the session
         * @param {string} cameraView - Camera view (cam_top, cam_side_l, cam_side_r)
         * @returns {Promise} - Promise with the API response
         */
        const formData = new FormData();
        formData.append('session_name', sessionName);
        formData.append('camera_view', cameraView);

        return fetch('/api/evaulation-visualization', {
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
    
    /**
     * Get available sessions
     * @returns {Promise} - Promise with the API response containing available sessions
     */
    static getAvailableSessions() {
        return fetch('/api/available-sessions')
            .then(response => response.json());
    }
}
