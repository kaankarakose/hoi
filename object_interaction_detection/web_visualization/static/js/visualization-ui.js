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
        this.fetchCNOSVisualization(params);
        this.fetchHAMERVisualization(params);
        this.fetchCombinedVisualization(params);
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
    
    /**
     * Fetch and update CNOS visualization
     */
    fetchCNOSVisualization(params) {
        // Show loading state
        $('#cnos-loading').removeClass('d-none');
        $('#cnos-error').addClass('d-none');
        
        ApiService.getCNOSVisualization(
            params.cameraView,
            params.frameIdx,
            params.frameType
        )
        .then(data => {
            $('#cnos-loading').addClass('d-none');
            
            if (data.success) {
                // Update image
                $('#cnos-image').attr('src', data.image);
                
                // Update data info
                let infoHTML = '';
                if (data.objects) {
                    infoHTML += `<p>Camera: ${params.cameraView}</p>`;
                    infoHTML += `<p>Frame: ${params.frameIdx}</p>`;
                    infoHTML += `<p>Frame Type: ${params.frameType}</p>`;
                    infoHTML += '<p>Objects:</p><ul>';
                    
                    Object.entries(data.objects).forEach(([objName, objData]) => {
                        infoHTML += `<li>${objName}: Score ${objData.max_score.toFixed(3)}</li>`;
                    });
                    
                    infoHTML += '</ul>';
                }
                
                $('#cnos-data').html(infoHTML);
            } else {
                // Show error
                $('#cnos-error').removeClass('d-none').text(data.message || 'Error loading CNOS data');
                $('#cnos-image').attr('src', '');
                $('#cnos-data').html('<p>Failed to load CNOS data.</p>');
            }
        })
        .catch(error => {
            console.error('Error fetching CNOS visualization:', error);
            $('#cnos-loading').addClass('d-none');
            $('#cnos-error').removeClass('d-none').text('Network error loading CNOS data');
            $('#cnos-image').attr('src', '');
            $('#cnos-data').html('<p>Failed to load CNOS data.</p>');
        });
    }
    
    /**
     * Fetch and update HAMER visualization
     */
    fetchHAMERVisualization(params) {
        // Show loading state
        $('#hamer-loading').removeClass('d-none');
        $('#hamer-error').addClass('d-none');
        
        ApiService.getHAMERVisualization(
            params.cameraView,
            params.frameIdx
        )
        .then(data => {
            $('#hamer-loading').addClass('d-none');
            
            if (data.success) {
                // Update image
                $('#hamer-image').attr('src', data.image);
                
                // Update data info
                let infoHTML = '';
                if (data.hamer_info) {
                    infoHTML += `<p>Camera: ${params.cameraView}</p>`;
                    infoHTML += `<p>Frame: ${params.frameIdx}</p>`;
                    
                    if (data.hamer_info.left_hand && data.hamer_info.left_hand.success) {
                        infoHTML += '<p>Left Hand: Detected</p>';
                    } else {
                        infoHTML += '<p>Left Hand: Not detected</p>';
                    }
                    
                    if (data.hamer_info.right_hand && data.hamer_info.right_hand.success) {
                        infoHTML += '<p>Right Hand: Detected</p>';
                    } else {
                        infoHTML += '<p>Right Hand: Not detected</p>';
                    }
                }
                
                $('#hamer-data').html(infoHTML);
            } else {
                // Show error
                $('#hamer-error').removeClass('d-none').text(data.message || 'Error loading HAMER data');
                $('#hamer-image').attr('src', '');
                $('#hamer-data').html('<p>Failed to load HAMER data.</p>');
            }
        })
        .catch(error => {
            console.error('Error fetching HAMER visualization:', error);
            $('#hamer-loading').addClass('d-none');
            $('#hamer-error').removeClass('d-none').text('Network error loading HAMER data');
            $('#hamer-image').attr('src', '');
            $('#hamer-data').html('<p>Failed to load HAMER data.</p>');
        });
    }
    
    /**
     * Fetch and update combined visualization
     */
    fetchCombinedVisualization(params) {
        // Show loading state
        $('#combined-loading').removeClass('d-none');
        $('#combined-error').addClass('d-none');
        
        ApiService.getCombinedVisualization(
            params.cameraView,
            params.frameIdx,
            params.objectName,
            params.frameType,
            params.featureType
        )
        .then(data => {
            $('#combined-loading').addClass('d-none');
            
            if (data.success) {
                // Update image
                $('#combined-image').attr('src', data.image);
                
                // Update data info
                let infoHTML = '';
                if (data.combined_info) {
                    infoHTML += `<p>Camera: ${params.cameraView}</p>`;
                    infoHTML += `<p>Frame: ${params.frameIdx}</p>`;
                    
                    if (data.combined_info.active_objects) {
                        infoHTML += '<p>Active Objects:</p><ul>';
                        data.combined_info.active_objects.forEach(obj => {
                            infoHTML += `<li>${obj.name}: Activity ${obj.activity.toFixed(3)}</li>`;
                        });
                        infoHTML += '</ul>';
                    }
                }
                
                $('#combined-data').html(infoHTML);
            } else {
                // Show error
                $('#combined-error').removeClass('d-none').text(data.message || 'Error loading combined data');
                $('#combined-image').attr('src', '');
                $('#combined-data').html('<p>Failed to load combined data.</p>');
            }
        })
        .catch(error => {
            console.error('Error fetching combined visualization:', error);
            $('#combined-loading').addClass('d-none');
            $('#combined-error').removeClass('d-none').text('Network error loading combined data');
            $('#combined-image').attr('src', '');
            $('#combined-data').html('<p>Failed to load combined data.</p>');
        });
    }
}

// Initialize UI when document is ready
$(document).ready(() => {
    // Create global visualization UI instance
    window.visualizationUI = new VisualizationUI();
});
