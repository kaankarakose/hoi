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
        
        // Get session name, camera view, and frame index from UI controls
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
        this.fetchCnosVisualization(params);
        this.fetchEvaluationData();
        
        this.fetchEvaluationVisualization(params.sessionName, params.cameraView);
        
    }
    
    /**
     * Fetch and update flow visualization
     */
    fetchFlowVisualization(params) {
        // Show loading state
        $('#flow-loading').removeClass('d-none');
        $('#flow-error').addClass('d-none');
        
        // Use the object name from params if available (from visualizeObject method)
        const objectName = params.objectName || '';
        const featureType = 'all';
        ApiService.getFlowVisualization(
            params.sessionName,
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
    fetchCnosVisualization(params) {
        // Show loading state
        $('#cnos-loading').removeClass('d-none');
        $('#cnos-error').addClass('d-none');
        // Use the object name from params if available (from visualizeObject method)
        const objectName = params.objectName || '';
        ApiService.getCnosVisualization(
            params.sessionName,
            params.cameraView,
            params.frameIdx,
            objectName
        )
        .then(data => {
            $('#cnos-loading').addClass('d-none');
            
            if (data.success) {
                // Update image
                $('#cnos-image').attr('src', data.image);
                
                // Update data info
                let infoHTML = '';
                if (data.cnos_info) {
                    infoHTML += `<p>Camera: ${params.cameraView}</p>`;
                    infoHTML += `<p>Frame: ${params.frameIdx}</p>`;
                    
                    // Only show object info if there is a specific object in the data
                    if (data.object_name) {
                        infoHTML += `<p>Object: ${data.object_name}</p>`;
                    }        
                }
                
                $('#cnos-data').html(infoHTML);
            } else {
                // Show error
                $('#cnos-error').removeClass('d-none').text(data.message || 'Error loading CNOS data');
                $('#cnos-image').attr('src', '');
                $('#cnos-data').html('<p>Failed to load CNOS data.</p>');
            }
        }).catch(error => {
            console.error('Error fetching CNOS visualization:', error);
            $('#cnos-loading').addClass('d-none');
            $('#cnos-error').removeClass('d-none').text('Network error loading CNOS data');
            $('#cnos-image').attr('src', '');
            $('#cnos-data').html('<p>Failed to load CNOS data.</p>');
        }
        );
    }

            /**
     * Fetch evaluation data for the current session and camera
     */
    fetchEvaluationData() {
        const params = this.getSelectedParameters();
        // Show loading state
        console.log('Fetching evaluation data with params:', params);
        $('#evaluation-loading').removeClass('d-none');
        $('#evaluation-error').addClass('d-none');
        
        ApiService.getEvaluationData(
            params.sessionName,
            params.cameraView,
        )
        .then(data => {
            $('#evaluation-loading').addClass('d-none');
            
            if (data.success) {
                // Update evaluation data display
                this.displayEvaluationData(data.data);
                console.log('Evaluation data loaded:', data.data);
                console.log(data.data.first_detection_times);
                
                //this.fetchEvaluationData(params.sessionName, params.cameraView, );
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
    }
    

    /** Fetch evaluation visualization - Fixed typo and improved error handling */
    fetchEvaluationVisualization(sessionName, camera_view) {
        /**
         * Fetch evaluation visualization data for a given session and camera view
         * @param {string} sessionName - Name of the session
         * @param {string} camera_view - Camera view (cam_top, cam_side_l, cam_side_r)
         * @returns {Promise} - Promise with the API response
         */

        $('#eval-loading').removeClass('d-none');
        $('#eval-error').addClass('d-none');
        $('#eval-image').hide(); // Hide previous image
        
        console.log('Fetching evaluation visualization for session:', sessionName, 'camera:', camera_view);
        
        return ApiService.getEvaluationVisualization(sessionName, camera_view)
            .then(data => {
                $('#eval-loading').addClass('d-none');
                
                if (data.success && data.image) {
                    // Update the image using the existing function
                    window.updateDetectionVisualization(data.image, 'Loaded');
                } else {
                    $('#eval-error')
                        .removeClass('d-none')
                        .text(data.message || 'Error loading evaluation visualization');
                    $('#eval-image').attr('src', '');
                    // Show placeholder
                    $('#eval-placeholder').show();
                }
                
                return data; // Return data for potential chaining
            })
            .catch(error => {
                $('#eval-loading').addClass('d-none');
                $('#eval-error')
                    .removeClass('d-none')
                    .text('Network error: Failed to load visualization');
                console.error('Error fetching evaluation visualization:', error);
                throw error; // Re-throw for caller handling
            });
    }

    /**
     * Display evaluation data in the UI and set up object interaction visualization
     * @param {Object} data - Evaluation data from the backend
     */
    displayEvaluationData(data) {
        // Store the original data for reference
        this.evaluationData = data;
        
        // Set FPS constant for time-to-frame conversion
        const FPS = 30;
        
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
        
        // Object detection timeline with clickable objects
        let timelineHTML = '<h6>Object Detection/Annotation Times</h6>';
        
        if (data.first_detection_times && Object.keys(data.first_detection_times).length > 0) {
            timelineHTML += '<table class="table table-sm table-striped">';
            timelineHTML += '<thead><tr><th>Object</th><th>First Detection Time</th><th>First Annotation Time</th><th>Actions</th></tr></thead>';
            timelineHTML += '<tbody>';
            
            for (const [objectName, detectionTime] of Object.entries(data.first_detection_times)) {
                const annotationTime = data.first_annotation_times?.[objectName] || 'N/A';
                const detectionFrame = Math.round(detectionTime * FPS);
                const annotationFrame = typeof annotationTime === 'number' ? Math.round(annotationTime * FPS) : 'N/A';
                
                timelineHTML += `<tr>`;
                timelineHTML += `<td>${objectName}</td>`;
                timelineHTML += `<td>${detectionTime.toFixed(2)} sec (frame ${detectionFrame})</td>`;
                timelineHTML += `<td>${typeof annotationTime === 'number' ? annotationTime.toFixed(2) + ' sec (frame ' + annotationFrame + ')' : annotationTime}</td>`;
                timelineHTML += `<td>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-info visualize-object" data-object="${objectName}">Visualize Object</button>
                        <button class="btn btn-outline-secondary visualize-at-frame" data-object="${objectName}" data-frame="${detectionFrame}">At Detection</button>
                        ${typeof annotationTime === 'number' ? `<button class="btn btn-outline-secondary visualize-at-frame" data-object="${objectName}" data-frame="${annotationFrame}">At Annotation</button>` : ''}
                    </div>
                </td>`;
                timelineHTML += `</tr>`;
            }
            
            timelineHTML += '</tbody></table>';
        } else {
            timelineHTML += '<div>No object detection timing data available</div>';
        }
        
        $('#detection-timeline').html(timelineHTML);
        
        // Set up event handlers for the visualize buttons
        $('.visualize-object').click((event) => {
            const objectName = $(event.currentTarget).data('object');
            this.visualizeObject(objectName);
        });
        
        // Set up event handlers for the visualization at specific frames
        $('.visualize-at-frame').click((event) => {
            const objectName = $(event.currentTarget).data('object');
            const frameIdx = $(event.currentTarget).data('frame');
            this.visualizeObjectAtFrame(objectName, frameIdx);
        });
        
        // Object hit details 
        let hitsHTML = '<h6>Object Hit Details</h6>';
        if (data.object_hits && Object.keys(data.object_hits).length > 0) {
            hitsHTML += '<div class="accordion" id="objectHitsAccordion">';
            
            let index = 0;
            for (const [objectName, hits] of Object.entries(data.object_hits)) {
                if (hits && hits.length > 0) {
                    hitsHTML += `
                        <div class="accordion-item">
                            <h2 class="accordion-header" id="heading${index}">
                                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                                        data-bs-target="#collapse${index}" aria-expanded="false" aria-controls="collapse${index}">
                                    ${objectName} (${hits.length} hits)
                                </button>
                            </h2>
                            <div id="collapse${index}" class="accordion-collapse collapse" aria-labelledby="heading${index}">
                                <div class="accordion-body">
                                    <table class="table table-sm">
                                        <thead>
                                            <tr>
                                                <th>Detection Time</th>
                                                <th>Annotation Time</th>
                                                <th>Time Diff</th>
                                                <th>Activeness</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                    `;
                    
                    for (const hit of hits) {
                        hitsHTML += `<tr>
                            <td>${hit.detection_time?.toFixed(2)} sec</td>
                            <td>${hit.annotation_time?.toFixed(2)} sec</td>
                            <td>${hit.time_diff?.toFixed(2)} sec</td>
                            <td>${hit.activeness?.toFixed(4)}</td>
                        </tr>`;
                    }
                    
                    hitsHTML += `
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    `;
                    index++;
                }
            }
            
            hitsHTML += '</div>';
        } else {
            hitsHTML += '<div>No object hit data available</div>';
        }
        
        $('#object-hits').html(hitsHTML);
    }
    



    /**
     * Visualize a specific object at a specific frame
     * @param {string} objectName - Name of the object to visualize
     * @param {number} frameIdx - Frame index to visualize at
     */
    visualizeObjectAtFrame(objectName, frameIdx) {
        const params = this.getSelectedParameters();
        
        // Update slider UI first
        $('#frame-select').val(frameIdx);
        $('#frame-value').text(frameIdx);
        
        // Update params with object and frame
        params.objectName = objectName;
        params.frameIdx = frameIdx;
        params.sessionName = $('#session-select').val();
        params.cameraView = $('#camera-select').val();

        // Fetch updated visualization
        this.fetchFlowVisualization(params);
        this.fetchCnosVisualization(params);
    }
}

// Initialize UI when document is ready
$(document).ready(() => {
    // Create global visualization UI instance
    window.visualizationUI = new VisualizationUI();
});
