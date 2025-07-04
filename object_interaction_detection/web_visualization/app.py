"""
Flask web application for visualizing hand-object interaction data
using the MotionFilteredLoader.
"""

import os
import sys
import logging

from flask import Flask

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the MotionFilteredLoader
from dataloaders.motion_filtered_loader import MotionFilteredLoader
from dataloaders.combined_loader import CombinedLoader, visualize_object_masks_combined

# Import routes
from routes.motion_routes import register_motion_routes

from utils.detection import DetectionManager


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__
          , static_folder='static',
          template_folder='templates',
          static_url_path='/static')
app.config['JSON_SORT_KEYS'] = False

# Configuration
DATA_ROOT_DIR = "/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data"
FLOW_ROOT_DIR = "/nas/project_data/B1_Behavior/rush/kaan/old_method/processed_data"
EVALUATION_ROOT_DIR = "/nas/project_data/B1_Behavior/rush/kaan/hoi/outputs/evaluation/0_5"
ANNOTATION_ROOT_DIR = "/nas/project_data/B1_Behavior/rush/kaan/hoi/annotations"
DEFAULT_SESSION = "imi_session1_6"
DEFAULT_CAMERA = "cam_top"
DEFAULT_FRAME = 200

# Function to get available sessions
def get_available_sessions():
    """Get list of available sessions from annotation directory"""
    import os
    try:
        # List all directories in the annotation root
        sessions = [d for d in os.listdir(ANNOTATION_ROOT_DIR) 
                   if os.path.isdir(os.path.join(ANNOTATION_ROOT_DIR, d))]
        return sorted(sessions)
    except Exception as e:
        logger.error(f"Error getting available sessions: {str(e)}")
        return []
# Add the get_available_sessions function to the Flask app
app.get_available_sessions = get_available_sessions
# Set configuration for the motion filtered loader
config = {
    'score_threshold': 0.40,    # CNOS confidence threshold
    'motion_threshold': 0.05,   # Threshold for determining motion
    'frames_dir': f"{DATA_ROOT_DIR}/orginal_frames"
}

# Initialize the MotionFilteredLoader
motion_loader = MotionFilteredLoader(
    session_name=DEFAULT_SESSION,
    data_root_dir=DATA_ROOT_DIR, 
    config=config
)


logger.info(f"Initialized MotionFilteredLoader for session {DEFAULT_SESSION}")
# Initialize the CombinedLoader for visualization
# Create a configuration with a specific score threshold

# Initialize the loader
combined_loader = CombinedLoader(session_name=DEFAULT_SESSION,
                                data_root_dir=DATA_ROOT_DIR,
                                config=config)
logger.info(f"Initialized CombinedLoader for session {DEFAULT_SESSION}")
# Initialize the DetectionManager
detection_manager = DetectionManager(data_dir=EVALUATION_ROOT_DIR, 
                                     session_name=DEFAULT_SESSION, 
                                     camera_view=DEFAULT_CAMERA)

logger.info(f"Initialized DetectionManager for session {DEFAULT_SESSION} and camera {DEFAULT_CAMERA}")

# Register routes
register_motion_routes(app, motion_loader, combined_loader, detection_manager)

if __name__ == '__main__':
    logger.info("Starting web visualization server")
    app.run(debug=True, host='0.0.0.0', port=8899)
    logger.info("Server stopped")
