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
        
        // Session change handler
        $('#session-select').change(() => {
            // When session changes, we need to update valid frames
            this.updateValidFrames();
        });
        
        // Camera change handler
        $('#camera-select').change(() => this.updateValidFrames());
        
        // Frame slider change handler
        $('#frame-select').on('input', () => {
            const frameValue = $('#frame-select').val();
            $('#frame-value').text(frameValue);
        });
        
        // Fetch evaluation data button
        $('#fetch-evaluation-btn').click(() => this.fetchEvaluationData());
        
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
        // Get frameType directly, use 'L_frames' as default if not found
        const frameType = $('#frame-type-select').length ? $('#frame-type-select').val() : 'L_frames';
        
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
            sessionName: $('#session-select').val(),
            cameraView: $('#camera-select').val(),
            frameIdx: parseInt($('#frame-select').val()),
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
        
        // We don't need to get these from DOM elements as they could come from other sources like evaluation data
        // Default to empty/default values if not specified
        const objectName = '';
        const featureType = 'all';
        
        ApiService.getFlowVisualization(
            params.cameraView,
            params.frameIdx,
            objectName,
            featureType
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
                    
                    // Only show object info if there is a specific object in the data
                    if (data.object_name) {
                        infoHTML += `<p>Object: ${data.object_name}</p>`;
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
    
    // Fetch evaluation data when the dedicated button is clicked
    $('#fetch-evaluation-btn').click(function() {
        const params = window.visualizationUI.getSelectedParameters();
        $('#evaluation-loading').removeClass('d-none');
        $('#evaluation-error').addClass('d-none');
        
        ApiService.getEvaluationData(
            params.sessionName,
            params.cameraView,
            params.frameIdx
        )
        .then(data => {
            $('#evaluation-loading').addClass('d-none');
            
            if (data.success) {
                // Update evaluation data display
                displayEvaluationData(data.data);
            } else {
                // Show error
                $('#evaluation-error').removeClass('d-none').text(data.message || 'Error loading evaluation data');
            }
        })
        .catch(error => {
            console.error('Error fetching evaluation data:', error);
            $('#evaluation-loading').addClass('d-none');
            $('#evaluation-error').removeClass('d-none').text('Network error loading evaluation data');
        });
    });
    
    // Display function for evaluation data
    function displayEvaluationData(data) {
        // General metrics
        let metricsHTML = '<h6>General Metrics</h6>';
        if (data.metrics) {
            metricsHTML += '<table class="table table-sm">';
            metricsHTML += '<tbody>';
            for (const [key, value] of Object.entries(data.metrics)) {
                metricsHTML += `<tr><td>${key}</td><td>${value}</td></tr>`;
            }
            metricsHTML += '</tbody></table>';
        } else {
            metricsHTML += '<div>No metrics available</div>';
        }
        $('#general-metrics').html(metricsHTML);
        
        // Object detection timeline
        let timelineHTML = '<h6>Object Detection Timeline</h6>';
        if (data.timeline) {
            // Implement timeline visualization here
            timelineHTML += '<div>Timeline data available (visualization pending)</div>';
        } else {
            timelineHTML += '<div>No timeline data available</div>';
        }
        $('#detection-timeline').html(timelineHTML);
        
        // Object hit details
        let hitsHTML = '<h6>Object Hit Details</h6>';
        if (data.object_hits && data.object_hits.length > 0) {
            hitsHTML += '<table class="table table-sm table-striped">';
            hitsHTML += '<thead><tr><th>Object</th><th>First Frame</th><th>Last Frame</th><th>Duration</th></tr></thead>';
            hitsHTML += '<tbody>';
            for (const hit of data.object_hits) {
                hitsHTML += `<tr><td>${hit.object_name || 'Unknown'}</td>`;
                hitsHTML += `<td>${hit.first_frame || 'N/A'}</td>`;
                hitsHTML += `<td>${hit.last_frame || 'N/A'}</td>`;
                hitsHTML += `<td>${hit.duration || 'N/A'}</td></tr>`;
            }
            hitsHTML += '</tbody></table>';
        } else {
            hitsHTML += '<div>No object hit data available</div>';
        }
        $('#object-hits').html(hitsHTML);
    }
});
