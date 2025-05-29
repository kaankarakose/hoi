"""
Main evaluation script for object interaction detection.

This script provides command-line functionality to evaluate
detection results against ground truth annotations.
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any
from pathlib import Path

# Use absolute imports instead of relative imports
import sys
sys.path.append('/nas/project_data/B1_Behavior/rush/kaan/hoi')

from object_interaction_detection.evaluation.evaluation_base import EvaluationConfig, EvaluationResult, Evaluator
from object_interaction_detection.evaluation.utils.annotation_parser import AnnotationData
from object_interaction_detection.evaluation.utils.detection_parser import DetectionData

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_available_sessions(annotation_root: str = None, detection_root: str = None) -> Dict[str, List[str]]:
    """
    Get lists of available sessions from annotation and detection directories.
    
    Args:
        annotation_root: Root directory for annotations
        detection_root: Root directory for detection results
        
    Returns:
        Dictionary with 'annotations' and 'detections' keys, each containing a list of session names
    """
    result = {
        'annotations': [],
        'detections': []
    }
    
    # Check annotations
    if annotation_root:
        try:
            result['annotations'] = [
                d for d in os.listdir(annotation_root)
                if os.path.isdir(os.path.join(annotation_root, d))
            ]
        except Exception as e:
            logger.error(f"Error reading annotation directory: {e}")
    
    # Check detections
    if detection_root:
        try:
            result['detections'] = [
                d for d in os.listdir(detection_root)
                if os.path.isdir(os.path.join(detection_root, d))
            ]
        except Exception as e:
            logger.error(f"Error reading detection directory: {e}")
    
    return result


def get_camera_views_for_session(session_name: str, detection_root: str) -> List[str]:
    """
    Get a list of available camera views for a given session.
    
    Args:
        session_name: Name of the session
        detection_root: Root directory for detection results
        
    Returns:
        List of camera view names
    """
    session_dir = os.path.join(detection_root, session_name)
    
    if not os.path.exists(session_dir):
        logger.warning(f"Session directory not found: {session_dir}")
        return []
    
    return [
        d for d in os.listdir(session_dir)
        if os.path.isdir(os.path.join(session_dir, d))
    ]


def main():
    """Main entry point for evaluation script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate object interaction detection results")
    
    # Session/camera selection
    parser.add_argument("--session", type=str, help="Specific session to evaluate")
    parser.add_argument("--all-sessions", action="store_true", help="Evaluate all available sessions", default = True)
    
    # Configuration
    parser.add_argument("--activeness-threshold", type=float, default=0.01, help="Activeness threshold (0-1)")
    parser.add_argument("--energy-threshold", type=float, default=0.01, help="Energy threshold (0-1)")
    parser.add_argument("--time-window", type=float, default=0.5, help="Time window in seconds for matching")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    
    # Paths
    parser.add_argument("--annotation-root", type=str, 
                        default="/nas/project_data/B1_Behavior/rush/kaan/hoi/annotations",
                        help="Root directory for annotations")
    parser.add_argument("--detection-root", type=str, 
                        default="/nas/project_data/B1_Behavior/rush/kaan/hoi/outputs/tracking_multi",
                        help="Root directory for detection results")
    parser.add_argument("--output-root", type=str, 
                        default="/nas/project_data/B1_Behavior/rush/kaan/hoi/outputs/evaluation_multi",
                        help="Output directory for results")
    
    # Extra options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging", default = False)
    parser.add_argument("--list-sessions", action="store_true", help="List available sessions and exit", default = False)
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # List available sessions if requested
    if args.list_sessions:
        available = get_available_sessions(args.annotation_root, args.detection_root)
        print("\nAvailable sessions in annotation directory:")
        for session in sorted(available['annotations']):
            print(f"  - {session}")
        
        print("\nAvailable sessions in detection directory:")
        for session in sorted(available['detections']):
            print(f"  - {session}")
            

    
    
    # Create configuration
    config = EvaluationConfig(
        activeness_threshold=args.activeness_threshold,
        energy_threshold=args.energy_threshold,
        time_window=args.time_window,
        fps=args.fps,
        annotation_root=args.annotation_root,
        detection_root=args.detection_root,
        output_root=args.output_root
    )
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Determine sessions to evaluate
    sessions_to_evaluate = []
    
    if args.all_sessions:
        # Get intersection of available sessions in both directories
        available = get_available_sessions(args.annotation_root, args.detection_root)
        sessions_to_evaluate = list(set(available['annotations']).intersection(set(available['detections'])))
        
        if not sessions_to_evaluate:
            logger.error("No sessions found in both annotation and detection directories.")
            return
    elif args.session:
        # Use specified session
        sessions_to_evaluate = [args.session]
    else:
        logger.error("Must specify either --session or --all-sessions")
        return
    

    
  
    # Evaluate all sessions for this camera
    results = evaluator.evaluate_multiple_sessions_multi_camera(
        sessions_to_evaluate,
        save_results=True
    )
    print(f"\n=== Evaluation complete for multi camera ===\n")
    
    # Output path information
    print(f"\nResults saved to: {args.output_root}")
    print("You can visualize the results by running:")
    print(f"  python -m object_interaction_detection.evaluation.visualize --results-dir {args.output_root}\n")


if __name__ == "__main__":
    main()

