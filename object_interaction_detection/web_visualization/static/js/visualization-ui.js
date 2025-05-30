/**
 * UI Controller for Hand-Object Interaction Visualization
 * Handles all UI interactions and updates
 */

class VisualizationUI {
    constructor() {
        this.initEventListeners();
        this.updateValidFrames();
    }
    
    /**
     * Initialize event listeners for UI controls
     */
    initEventListeners() {
        // Update button click handler
        $('#update-btn').click(() => this.updateAllVisualizations());
        
        // Camera change handler
        $('#camera-select').change(() => this.updateValidFrames());
        
        // Frame slider change handler
        $('#frame-select').on('input', () => {
            const frameValue = $('#frame-select').val();
            $('#frame-value').text(frameValue);
        });
        
        // Advanced: Auto-update when parameters change (optional)
        $('.auto-update').change(() => {
            if ($('#auto-update-toggle').prop('checked')) {
                this.updateAllVisualizations();
            }
        });
    }
    
    /**
     * Update the frame slider range based on valid frames for selected camera
     */
    updateValidFrames() {
        const cameraView = $('#camera-select').val();
        const frameType = $('#frame-type-select').val();
        
        // Show loading state
        $('#valid-frames-count').text('Loading...');
        
        ApiService.getValidFrames(cameraView)
            .then(data => {
                if (data.success) {
                    const leftFrames = data.valid_frames[cameraView]?.left || [];
                    const rightFrames = data.valid_frames[cameraView]?.right || [];
                    
                    // Update count display
                    $('#valid-frames-count').text(
                        `Left: ${leftFrames.length} frames / Right: ${rightFrames.length} frames`
                    );
                    
                    // Determine which frames to use based on frame type
                    const frames = frameType === 'L_frames' ? leftFrames : rightFrames;
                    
                    if (frames.length > 0) {
                        // Update slider range
                        $('#frame-select').attr('min', frames[0]);
                        $('#frame-select').attr('max', frames[frames.length - 1]);
                        
                        // Set current value to first frame if outside range
                        const currentVal = parseInt($('#frame-select').val());
                        if (currentVal < frames[0] || currentVal > frames[frames.length - 1]) {
                            $('#frame-select').val(frames[0]);
                            $('#frame-value').text(frames[0]);
                        }
                    } else {
                        // No valid frames
                        $('#valid-frames-count').text('No valid frames found');
                    }
                } else {
                    $('#valid-frames-count').text('Error loading frames');
                }
            })
            .catch(error => {
                console.error('Error fetching valid frames:', error);
                $('#valid-frames-count').text('Error loading frames');
            });
    }
    
    /**
     * Get selected parameters from UI
     */
    getSelectedParameters() {
        return {
            cameraView: $('#camera-select').val(),
            frameIdx: parseInt($('#frame-select').val()),
            frameType: $('#frame-type-select').val(),
            objectName: $('#object-select').val(),
            featureType: $('#flow-feature-type').val()
        };
    }
    
    /**
     * Update all visualizations with current parameters
     */
    updateAllVisualizations() {
        const params = this.getSelectedParameters();
        
        this.fetchFlowVisualization(params);
    }
    
    /**
     * Fetch and update flow visualization
     */
    fetchFlowVisualization(params) {
        // Show loading state
        $('#flow-loading').removeClass('d-none');
        $('#flow-error').addClass('d-none');
        
        ApiService.getFlowVisualization(
            params.cameraView,
            params.frameIdx,
            params.objectName,
            params.featureType
        )
        .then(data => {
            $('#flow-loading').addClass('d-none');
            
            if (data.success) {
                // Update image
                $('#flow-image').attr('src', data.image);
                
                // Update data info
                let infoHTML = '';
                if (data.flow_info) {
                    infoHTML += `<p>Camera: ${params.cameraView}</p>`;
                    infoHTML += `<p>Frame: ${params.frameIdx}</p>`;
                    
                    if (params.objectName) {
                        infoHTML += `<p>Object: ${params.objectName}</p>`;
                    }
                    
                    if (data.flow_info.is_moving !== undefined) {
                        infoHTML += `<p>Moving: ${data.flow_info.is_moving ? 'Yes' : 'No'}</p>`;
                    }
                    
                    if (data.flow_info.avg_dir !== undefined) {
                        infoHTML += `<p>Avg. Direction: ${data.flow_info.avg_dir.toFixed(2)} degrees</p>`;
                    }
                    
                    if (data.flow_info.avg_len !== undefined) {
                        infoHTML += `<p>Avg. Speed: ${data.flow_info.avg_len.toFixed(4)}</p>`;
                    }
                    
                    if (data.flow_info.avg_brightness !== undefined) {
                        infoHTML += `<p>Avg. Brightness: ${data.flow_info.avg_brightness.toFixed(4)}</p>`;
                    }
                }
                
                $('#flow-data').html(infoHTML);
            } else {
                // Show error
                $('#flow-error').removeClass('d-none').text(data.message || 'Error loading flow data');
                $('#flow-image').attr('src', '');
                $('#flow-data').html('<p>Failed to load flow data.</p>');
            }
        })
        .catch(error => {
            console.error('Error fetching flow visualization:', error);
            $('#flow-loading').addClass('d-none');
            $('#flow-error').removeClass('d-none').text('Network error loading flow data');
            $('#flow-image').attr('src', '');
            $('#flow-data').html('<p>Failed to load flow data.</p>');
        });
    }
    
   
 
}

// Initialize UI when document is ready
$(document).ready(() => {
    // Create global visualization UI instance
    window.visualizationUI = new VisualizationUI();
});
