/**
 * Configuration for Hand-Object Interaction Dataloaders
 * Central place to define defaults and settings
 */

const LoaderConfig = {
    // Default settings
    DEFAULT_SESSION: "imi_session1_6",
    DEFAULT_CAMERA: "cam_top",
    DEFAULT_FRAME: 200,
    DEFAULT_FRAME_TYPE: "L_frames",
    
    // Available camera views
    CAMERA_VIEWS: [
        { id: "cam_top", name: "Top Camera" },
        { id: "cam_side_l", name: "Left Side Camera" },
        { id: "cam_side_r", name: "Right Side Camera" }
    ],
    
    // Available objects
    OBJECTS: [
        "AMF1", "AMF2", "AMF3", "BOX", "CUP", 
        "DINOSAUR", "FIRETRUCK", "HAIRBRUSH", "PINCER", "WRENCH"
    ],
    
    // Flow feature types
    FLOW_FEATURE_TYPES: [
        { id: "all", name: "All Features" },
    ],
    
    // Threshold settings
    THRESHOLDS: {
        SCORE: 0.40,       // Object detection confidence threshold
        MOTION: 0.05,      // Motion detection threshold
        ACTIVITY: 0.25     // Object activity threshold for visibility
    }
};
