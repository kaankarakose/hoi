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

# Import routes
from routes.motion_routes import register_motion_routes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configuration
DATA_ROOT_DIR = "/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data"
FLOW_ROOT_DIR = "/nas/project_data/B1_Behavior/rush/kaan/old_method/processed_data"
DEFAULT_SESSION = "imi_session1_6"
DEFAULT_CAMERA = "cam_top"
DEFAULT_FRAME = 200

# Set configuration for the motion filtered loader
config = {
    'score_threshold': 0.40,    # CNOS confidence threshold
    'motion_threshold': 0.05,   # Threshold for determining motion
    'temporal_window': 2,       # Window for optical flow aggregation
    'frames_dir': f"{DATA_ROOT_DIR}/orginal_frames"
}

# Initialize the MotionFilteredLoader
motion_loader = MotionFilteredLoader(
    session_name=DEFAULT_SESSION, 
    data_root_dir=DATA_ROOT_DIR, 
    config=config
)

logger.info(f"Initialized MotionFilteredLoader for session {DEFAULT_SESSION}")

# Register routes
register_motion_routes(app, motion_loader)

if __name__ == '__main__':
    logger.info("Starting web visualization server")
    app.run(debug=True, host='0.0.0.0', port=8888)
    logger.info("Server stopped")
