"""
Base evaluation module for object interaction detection.

This module provides base classes and utilities for evaluating
object interaction detection against ground truth annotations.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path

# Add the project root to path to ensure imports work
sys.path.append('/nas/project_data/B1_Behavior/rush/kaan/hoi')

# Import evaluation utilities
from object_interaction_detection.evaluation.utils.detection_parser import DetectionData, Detection
from object_interaction_detection.evaluation.utils.annotation_parser import AnnotationData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""
    # Detection thresholds
    activeness_threshold: float = 0.25  # Threshold for considering an object as active
    energy_threshold: float = 0.3  # Threshold for object energy
    
    # Time window settings
    time_window: float = None # Time window in seconds for matching detections to annotations
    fps: float = 30.0  # Frames per second in the video
    
    # Paths
    annotation_root: str = "/nas/project_data/B1_Behavior/rush/kaan/hoi/annotations"
    detection_root: str = "/nas/project_data/B1_Behavior/rush/kaan/hoi/outputs/tracking"
    output_root: str = "/nas/project_data/B1_Behavior/rush/kaan/hoi/outputs/evaluation"

@dataclass
class DetectionHit:
    """Represents a hit between a detection and an annotation."""
    object_name: str
    annotation_time: float  # Start time of the annotation segment
    detection_time: float   # Time of the detection
    time_diff: float        # Time difference (detection - annotation)
    hit_within_window: bool  # Whether the hit is within the configured time window
    activeness: float       # Activeness value of the detection
    energy: float           # Energy value of the detection
    isFirstPlayDetection: bool # Whether the detection is the first play detection
    
    def to_dict(self) -> Dict:
        """Convert hit to dictionary for serialization."""
        return {
            'object_name': self.object_name,
            'annotation_time': self.annotation_time,
            'detection_time': self.detection_time,
            'time_diff': self.time_diff,
            'hit_within_window': self.hit_within_window,
            'activeness': self.activeness,
            'energy': self.energy,
            'isFirstPlayDetection': self.isFirstPlayDetection
        }

class EvaluationResult:
    """Results from evaluation of detection against annotations."""
    
    def __init__(self, session_name: str = None, camera_view: str = None):
        self.session_name = session_name
        self.camera_view = camera_view
        
        # Overall metrics
        self.metrics = {}
        
        # Per-object metrics
        self.object_metrics = {}
        
        # Per-object hits
        self.object_hits: Dict[str, List[DetectionHit]] = {}
        
        # Time to first detection (per object)
        self.first_detection_times = {}
        
        # First annotation times (per object and overall)
        self.first_annotation_times = {}
        self.first_overall_annotation_time = None
    
    def add_metric(self, name: str, value: float):
        """Add a global metric."""
        self.metrics[name] = value
    
    def add_object_metric(self, object_name: str, name: str, value: float):
        """Add a metric for a specific object."""
        if object_name not in self.object_metrics:
            self.object_metrics[object_name] = {}
        self.object_metrics[object_name][name] = value
    
    def add_hit(self, hit: DetectionHit):
        """Add a detection hit."""
        if hit.object_name not in self.object_hits:
            self.object_hits[hit.object_name] = []
        self.object_hits[hit.object_name].append(hit)
    
    def set_first_detection_time(self, object_name: str, time: float):
        """Set the first detection time for an object."""
        self.first_detection_times[object_name] = time
    
    def set_first_annotation_time(self, object_name: str, time: float):
        """Set the first annotation time for an object."""
        self.first_annotation_times[object_name] = time
        
        # Update overall first annotation time if this is earlier
        if self.first_overall_annotation_time is None or time < self.first_overall_annotation_time:
            self.first_overall_annotation_time = time
    
    def compute_summary_metrics(self):
        """Compute summary metrics from hits and other data."""
        # Count total annotations and hits
        total_annotations = len(self.first_annotation_times)

        total_hit_objects = sum(1 for obj_name, hits in self.object_hits.items() 
                               if any(hit.hit_within_window for hit in hits))
        
        # Add object count metrics
        self.add_metric('total_annotations', total_annotations)
        self.add_metric('total_hit_objects', total_hit_objects)
        
        # Compute average time difference for hits within window
        time_diffs = []
        for obj_name, hits in self.object_hits.items():
            for hit in hits:
                if hit.hit_within_window:
                    time_diffs.append(hit.time_diff)
        
        if time_diffs:
            avg_time_diff = np.mean(time_diffs)
            std_time_diff = np.std(time_diffs)
            self.add_metric('avg_time_diff', avg_time_diff)
            self.add_metric('std_time_diff', std_time_diff)
        
        
    
    def to_dict(self) -> Dict:
        """Convert results to dictionary for serialization."""
        return {
            'session_name': self.session_name,
            'camera_view': self.camera_view,
            'metrics': self.metrics,
            'object_metrics': self.object_metrics,
            'object_hits': {obj: [hit.to_dict() for hit in hits] 
                           for obj, hits in self.object_hits.items()},
            'first_detection_times': self.first_detection_times,
            'first_annotation_times': self.first_annotation_times,
            'first_overall_annotation_time': self.first_overall_annotation_time
        }
    
    def save_to_json(self, file_path: str):
        """Save results to a JSON file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved evaluation results to {file_path}")
    
    def print_summary(self):
        """Print a summary of the evaluation results."""
        # Create a visually distinct header
        print(f"\n{'=' * 80}")
        total_hit_objects = self.metrics.get('total_hit_objects', 0)
        total_annotations = self.metrics.get('total_annotations', 0)
        success_ratio = f"{total_hit_objects}/{total_annotations}"
        
        print(f"SESSION: {self.session_name} - {self.camera_view} | Objects detected: {success_ratio}")
        print(f"{'=' * 80}\n")
        
        # Print overall metrics
        print("📊 OVERALL METRICS:")
        if self.metrics:
            for name, value in self.metrics.items():
                print(f"  {name}: {value:.2f}")
        
        # Display a count of objects with successful detections
        hit_objects = sum(1 for obj_name, hits in self.object_hits.items() if any(hit.hit_within_window for hit in hits))
        total_objects = len(self.first_annotation_times)
        print(f"\n📋 DETECTION SUMMARY: {hit_objects}/{total_objects} objects detected within time window")
        
        # Display info about first play detection - this should be boolean per session
        first_play_detected = any(any(hit.isFirstPlayDetection for hit in hits) for obj_name, hits in self.object_hits.items())
        first_play_status = "✅ DETECTED" if first_play_detected else "❌ NOT DETECTED"
        print(f"\n🌟 FIRST PLAY DETECTION: {first_play_status}")
        
        # Print annotation timing
        if self.first_overall_annotation_time is not None:
            print(f"\n⏱️ First annotation at: {self.first_overall_annotation_time:.2f}s")
        
        # Print per-object results in a tabular format
        print("\n🔍 OBJECT RESULTS:")
        # Print header
        print(f"  {'Object':<15} {'Status':<8} {'Anno Time':<10} {'Detect Time':<12} {'Time Diff':<10}")
        print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*12} {'-'*10}")
        
        for obj_name in sorted(set(list(self.object_metrics.keys()) + list(self.first_annotation_times.keys()))):
            obj_metrics = self.object_metrics.get(obj_name, {})
            first_anno_time = self.first_annotation_times.get(obj_name)
            first_detect_time = self.first_detection_times.get(obj_name)
            
            # Determine if this object had a hit within the window
            hits_for_obj = self.object_hits.get(obj_name, [])
            hit_status = "✅ OK" if any(hit.hit_within_window for hit in hits_for_obj) else "❌ NOT"
            
            # Format times
            anno_time_str = f"{first_anno_time:.2f}s" if first_anno_time is not None else "N/A"
            detect_time_str = f"{first_detect_time:.2f}s" if first_detect_time is not None else "N/A"
            
            # Calculate time difference if both times exist
            time_diff_str = "N/A"
            if first_anno_time is not None and first_detect_time is not None:
                time_diff = first_detect_time - first_anno_time
                time_diff_str = f"{time_diff:.2f}s"
            
            # Print row
            print(f"  {obj_name:<15} {hit_status:<8} {anno_time_str:<10} {detect_time_str:<12} {time_diff_str:<10}")

class Evaluator:
    """Main class for evaluating detections against annotations."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
    
    def evaluate_session(self, session_name: str, camera_view: str) -> EvaluationResult:
        """
        Evaluate detections for a session and camera against annotations.
        
        Args:
            session_name: Name of the session (e.g., 'imi_session1_2')
            camera_view: Name of the camera view (e.g., 'cam_side_l')
            
        Returns:
            EvaluationResult with metrics and hits
        """
        logger.info(f"Evaluating session {session_name}, camera {camera_view}")
        
        # Initialize result
        result = EvaluationResult(session_name, camera_view)
        
        # Load detection data
        detection_data = DetectionData(session_name=session_name, camera_view=camera_view,
                                      detection_root=self.config.detection_root)
 
        # Process detections with energy
        logger.info("Processing detections with energy calculation")
        detections = detection_data.process_with_energy(
            energy_threshold=self.config.energy_threshold,
            activeness_threshold=self.config.activeness_threshold
        )
        
        # Load annotation data
        logger.info("Loading annotation data")
        annotation_data = AnnotationData(session_name=session_name)
        annotation_data.set_fps(self.config.fps)
        
        # Find first annotations and compare with detections
        self._evaluate_object_identities(result, annotation_data, detections)
        
        # Compute summary metrics based on hits
        result.compute_summary_metrics()
        
        return result

    def evaluate_session_multi_camera(self, session_name: str) -> EvaluationResult:
        """
        Evaluate detections for a session against annotations.
        
        Args:
            session_name: Name of the session (e.g., 'imi_session1_2')
            
        Returns:
            EvaluationResult with metrics and hits
        """
        logger.info(f"Evaluating session {session_name}, multi")
        
        # Initialize result
        result = EvaluationResult(session_name)
        
        # Load detection data
        detection_data = DetectionData(session_name=session_name,
                                      detection_root=self.config.detection_root)
 
        # Process detections with energy
        logger.info("Processing detections with energy calculation")
        detections = detection_data.process_with_energy(
            energy_threshold=self.config.energy_threshold,
            activeness_threshold=self.config.activeness_threshold
        )
        
        # Load annotation data
        logger.info("Loading annotation data")
        annotation_data = AnnotationData(session_name=session_name)
        annotation_data.set_fps(self.config.fps)
        
        # Find first annotations and compare with detections
        self._evaluate_object_identities(result, annotation_data, detections)
        
        # Compute summary metrics based on hits
        result.compute_summary_metrics()
        
        return result
      

    def _evaluate_object_identities(self, result: EvaluationResult, 
                                   annotation_data: AnnotationData,
                                   detections: List[Detection]):
        """
        Evaluate object identity detections against annotations.
        
        Args:
            result: EvaluationResult to store results in
            annotation_data: Loaded annotation data
            detections: List of processed detections
        """

        ## First Play Detection
        first_play_annotated_time = annotation_data.get_first_segment("ObjectIdentity").start_time # this is what we care for First Play Detection
        first_played_objects = annotation_data.get_first_played_object(return_all=True, combine=False) # this is what we care for First Play Detection
        #
        counted_already = False # Need to count each object only once
        # Group detections by object
        detections_by_object = {}
        for detection in detections:
            if detection.object_name not in detections_by_object:
                detections_by_object[detection.object_name] = []
            detections_by_object[detection.object_name].append(detection)
        
        # Get all annotated objects
        all_objects = annotation_data.get_all_objects()
        
        # Process each annotated object
        for obj_name in all_objects:
            # Find first annotation time for this object
            first_anno_time = None
            for segment in annotation_data.get_all_segments("ObjectIdentity"):
                if obj_name in segment.get_objects():
                    first_anno_time = segment.start_time
                    break
            
            if first_anno_time is None:
                logger.warning(f"No annotation found for object {obj_name}")
                continue
            
            # Store first annotation time
            result.set_first_annotation_time(obj_name, first_anno_time)
            
            # Get detections for this object
            obj_detections = detections_by_object.get(obj_name, [])
            
            # Find the first detection
            if obj_detections:
                # Sort by frame index
                obj_detections.sort(key=lambda d: d.frame_idx)
    
                first_detection = obj_detections[0]
                
                # Convert frame to time
                first_detection_time = first_detection.frame_idx / self.config.fps
                
                # Store first detection time
                result.set_first_detection_time(obj_name, first_detection_time)
                
                # Calculate time difference
                time_diff = first_detection_time - first_anno_time
                
                # Check if detection is within time window of annotation
                hit_within_window = abs(time_diff) <= self.config.time_window
                
                # Check if detection is the first play detection
            
                if hit_within_window and first_anno_time == first_play_annotated_time and obj_name in first_played_objects and not counted_already:
                    counted_already = True
                    isFirstPlayDetection = True
                else:
                    isFirstPlayDetection = False

                # Create a hit record
                hit = DetectionHit(
                    object_name=obj_name,
                    annotation_time=first_anno_time,
                    detection_time=first_detection_time,
                    time_diff=time_diff,
                    hit_within_window=hit_within_window,
                    activeness=first_detection.activeness,
                    energy=first_detection.energy,
                    isFirstPlayDetection= isFirstPlayDetection
                )
                
                # Add hit to results
                result.add_hit(hit)
            else:
                logger.warning(f"No detections found for annotated object {obj_name}")
    
    def evaluate_multiple_sessions(self, session_list: List[str], camera_view: str,
                                  save_results: bool = True) -> Dict[str, EvaluationResult]:
        """
        Evaluate multiple sessions for a camera view.
        
        Args:
            session_list: List of session names to evaluate
            camera_view: Camera view to evaluate
            save_results: Whether to save results to JSON files
            
        Returns:
            Dictionary mapping session names to EvaluationResults
        """
        results = {}
        
        for session_name in session_list:
            try:
                result = self.evaluate_session(session_name, camera_view)
                results[session_name] = result
                
                # Print summary
                result.print_summary()
                
                # Save to JSON if requested
                if save_results:
                    output_dir = os.path.join(self.config.output_root, session_name, camera_view)
                    os.makedirs(output_dir, exist_ok=True)
                    file_path = os.path.join(output_dir, f"{session_name}_{camera_view}_evaluation.json")
                    result.save_to_json(file_path)
                    logger.info(f"Saved results to {file_path}")
            except Exception as e:
                logger.error(f"Error evaluating session {session_name}, camera {camera_view}: {e}")
        
        # Print aggregate summary across all sessions
        self.print_aggregate_summary(results, camera_view)
        
        return results
    

    def evaluate_multiple_sessions_multi_camera(self, session_list: List[str],
                                  save_results: bool = True) -> Dict[str, EvaluationResult]:
        """
        Evaluate multiple sessions for multi camera.
        
        Args:
            session_list: List of session names to evaluate
            save_results: Whether to save results to JSON files
            
        Returns:
            Dictionary mapping session names to EvaluationResults
        """
        results = {}
        for session_name in session_list:
            try:
                result = self.evaluate_session_multi_camera(session_name)
                results[session_name] = result
                
                # Print summary
                result.print_summary()
                
                # Save to JSON if requested
                if save_results:
                    output_dir = os.path.join(self.config.output_root, session_name)
                    os.makedirs(output_dir, exist_ok=True)
                    file_path = os.path.join(output_dir, f"{session_name}_multi_evaluation.json")
                    result.save_to_json(file_path)
                    logger.info(f"Saved results to {file_path}")
            except Exception as e:
                logger.error(f"Error evaluating session {session_name}, multi camera: {e}")
        
        # Print aggregate summary across all sessions
        self.print_aggregate_summary(results,camera_view = "multi")
        
        return results
        
    def print_aggregate_summary(self, results: Dict[str, EvaluationResult], camera_view: str) -> None:
        """Print an aggregate summary of results across all sessions.
        
        Args:
            results: Dictionary mapping session names to EvaluationResults
            camera_view: The camera view being evaluated
        """
        if not results:
            logger.warning("No results to summarize")
            return
        
        print(f"\n{'#' * 100}")
        print(f"## AGGREGATE SUMMARY FOR CAMERA: {camera_view}")
        print(f"{'#' * 100}\n")
        
        # Set up summary metrics
        successful_sessions = 0
        all_sessions_data = []
        # For calculating overall average time diff
        all_time_diffs = []  # Store all time diffs across sessions
        print(f"\n{'=' * 80}")
        print(f"{'SESSION':<20} {'SUCCESS RATIO':<15} {'STATUS':<10} {'TIME DIFF' :<10} {'DETECTED OBJECTS':<20}")
        print(f"{'-'*20} {'-'*15} {'-'*10} {'-'*20}")
        
        for session_name, result in sorted(results.items()):
            # Get detected objects count
            hits = result.object_hits
            hit_objects = sum(1 for obj_name, obj_hits in hits.items() if any(hit.hit_within_window for hit in obj_hits))
            total_objects = len(result.first_annotation_times)
            
            # Calculate success percentage
            success_rate = hit_objects / total_objects if total_objects > 0 else 0
            success_ratio = f"{hit_objects}/{total_objects}"
            
            # Check if any first play detection exists (should be boolean per session)
            first_play_detection = any(any(hit.isFirstPlayDetection for hit in obj_hits) for obj_name, obj_hits in hits.items())

            #Time diff per


            # Calculate time diff for this session - FIXED VERSION
            session_time_diffs = []
            for obj_name, obj_hits in hits.items():
                for hit in obj_hits:  # Iterate through individual hits
                    if hit.hit_within_window:
                        session_time_diffs.append(abs(hit.time_diff))
                        all_time_diffs.append(abs(hit.time_diff))  # Add to overall collection
            # Calculate average time diff for this session
            avg_time_diff = sum(session_time_diffs) / len(session_time_diffs) if session_time_diffs else 0


            # Store data for overall calculation
            all_sessions_data.append({
                'session': session_name,
                'hit_objects': hit_objects,
                'total_objects': total_objects,
                'success_rate': success_rate,
                'first_play_detection': first_play_detection,
                'avg_time_diff': avg_time_diff
            })
            

            
            # Status indicator - consider a session successful only if we detect first play
            status = "✅ OK" if first_play_detection else "❌ NOT"
            
            # Print session summary row
            print(f"{session_name:<20} {success_ratio:<15} {status:<10} {avg_time_diff:<15.2f} {hit_objects}")
        
        # Calculate and display overall stats
        total_sessions = len(results)
        total_hit_objects = sum(data['hit_objects'] for data in all_sessions_data)
        total_objects = sum(data['total_objects'] for data in all_sessions_data)
        overall_success_rate = total_hit_objects / total_objects if total_objects > 0 else 0
        
        # Count sessions with first play detection
        sessions_with_first_play = sum(1 for data in all_sessions_data if data.get('first_play_detection', False))

        ##
        # Calculate overall average time diff across ALL hits
        overall_avg_time_diff = sum(all_time_diffs) / len(all_time_diffs) if all_time_diffs else 0
        
        # Calculate average of session averages (alternative metric)
        session_avg_time_diff = sum(data['avg_time_diff'] for data in all_sessions_data) / len(all_sessions_data) if all_sessions_data else 0
        
        

        print(f"\n{'=' * 80}")
        print(f"OVERALL DETECTION RATE: {total_hit_objects}/{total_objects} ({overall_success_rate:.2f})")
        print(f"SESSIONS WITH FIRST PLAY DETECTION: {sessions_with_first_play}/{total_sessions} sessions")
        print(f"OVERALL AVERAGE TIME DIFF: {overall_avg_time_diff:.2f}s (across {len(all_time_diffs)} hits)")
        print(f"AVERAGE OF SESSION AVERAGES: {session_avg_time_diff:.2f}s")
        print(f"{'=' * 80}\n")
    
    def add_metric(self, name: str, value: float):
        """Add a global metric."""
        self.metrics[name] = value
    
    def add_object_metric(self, object_name: str, metric_name: str, value: float):
        """Add a per-object metric."""
        if object_name not in self.per_object_metrics:
            self.per_object_metrics[object_name] = {}
        self.per_object_metrics[object_name][metric_name] = value
    
    def add_timing_error(self, object_name: str, error_seconds: float):
        """Add a timing error for an object."""
        if object_name not in self.timing_errors:
            self.timing_errors[object_name] = []
        self.timing_errors[object_name].append(error_seconds)
    
    def set_confusion_matrix(self, matrix: np.ndarray, labels: List[str]):
        """Set the confusion matrix."""
        self.confusion_matrix = {
            'matrix': matrix,
            'labels': labels
        }
    
    def add_session_metric(self, session_name: str, metric_name: str, value: float):
        """Add a per-session metric."""
        if session_name not in self.session_metrics:
            self.session_metrics[session_name] = {}
        self.session_metrics[session_name][metric_name] = value
    
    def to_dict(self) -> Dict:
        """Convert results to a dictionary."""
        return {
            'metrics': self.metrics,
            'per_object_metrics': self.per_object_metrics,
            'timing_errors': {k: {'mean': np.mean(v), 'std': np.std(v), 'values': v} 
                             for k, v in self.timing_errors.items() if v},
            'confusion_matrix': self.confusion_matrix,
            'session_metrics': self.session_metrics
        }
    
    def save_to_json(self, file_path: str):
        """Save results to a JSON file."""
        # Convert numpy types to Python native types for JSON serialization
        result_dict = self.to_dict()
        
        # Convert numpy arrays and special types
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        result_dict = convert_to_native(result_dict)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Saved evaluation results to {file_path}")
    
    def print_summary(self):
        """Print a summary of the evaluation results."""
        print("=== Evaluation Summary ===")
        
        if self.metrics:
            print("\nGlobal Metrics:")
            for name, value in self.metrics.items():
                print(f"  {name}: {value:.4f}")
        
        if self.per_object_metrics:
            print("\nPer-Object Metrics:")
            for obj_name, metrics in self.per_object_metrics.items():
                print(f"  {obj_name}:")
                for metric_name, value in metrics.items():
                    print(f"    {metric_name}: {value:.4f}")
        
        if self.timing_errors:
            print("\nTiming Errors (seconds):")
            for obj_name, errors in self.timing_errors.items():
                if errors:
                    print(f"  {obj_name}: mean={np.mean(errors):.4f}, std={np.std(errors):.4f}")
        
        if self.session_metrics:
            print("\nPer-Session Metrics:")
            for session_name, metrics in self.session_metrics.items():
                print(f"  {session_name}:")
                for metric_name, value in metrics.items():
                    print(f"    {metric_name}: {value:.4f}")
    
    def visualize_results(self, save_dir: Optional[str] = None):
        """
        Visualize evaluation results with plots.
        
        Args:
            save_dir: Directory to save plots. If None, plots will be displayed.
        """
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Plot global metrics
        if self.metrics:
            self._plot_global_metrics(save_dir)
        
        # Plot per-object metrics
        if self.per_object_metrics:
            self._plot_object_metrics(save_dir)
        
        # Plot timing errors
        if self.timing_errors:
            self._plot_timing_errors(save_dir)
        
        # Plot confusion matrix
        if self.confusion_matrix is not None:
            self._plot_confusion_matrix(save_dir)
    
    def _plot_global_metrics(self, save_dir: Optional[str] = None):
        """Plot global metrics."""
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = self.metrics
        names = list(metrics.keys())
        values = [metrics[name] for name in names]
        
        ax.bar(names, values)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Score')
        ax.set_title('Global Metrics')
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'global_metrics.png'))
            plt.close()
        else:
            plt.show()
    
    def _plot_object_metrics(self, save_dir: Optional[str] = None):
        """Plot per-object metrics."""
        # Get all unique metric names
        all_metrics = set()
        for metrics in self.per_object_metrics.values():
            all_metrics.update(metrics.keys())
        
        for metric_name in all_metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
            objects = []
            values = []
            
            for obj_name, metrics in self.per_object_metrics.items():
                if metric_name in metrics:
                    objects.append(obj_name)
                    values.append(metrics[metric_name])
            
            # Sort by value
            sorted_indices = np.argsort(values)[::-1]  # Descending
            objects = [objects[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
            
            ax.bar(objects, values)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Score')
            ax.set_title(f'Per-Object {metric_name}')
            ax.set_xticklabels(objects, rotation=45, ha='right')
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'object_{metric_name}.png'))
                plt.close()
            else:
                plt.show()
    
    def _plot_timing_errors(self, save_dir: Optional[str] = None):
        """Plot timing errors."""
        if not self.timing_errors:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        objects = []
        means = []
        stds = []
        
        for obj_name, errors in self.timing_errors.items():
            if errors:
                objects.append(obj_name)
                means.append(np.mean(errors))
                stds.append(np.std(errors))
        
        # Sort by absolute mean
        sorted_indices = np.argsort(np.abs(means))
        objects = [objects[i] for i in sorted_indices]
        means = [means[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]
        
        ax.bar(objects, means, yerr=stds, capsize=5)
        ax.set_ylabel('Timing Error (seconds)')
        ax.set_title('Detection Timing Errors')
        ax.set_xticklabels(objects, rotation=45, ha='right')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'timing_errors.png'))
            plt.close()
        else:
            plt.show()
    
    def _plot_confusion_matrix(self, save_dir: Optional[str] = None):
        """Plot confusion matrix."""
        if self.confusion_matrix is None:
            return
        
        matrix = self.confusion_matrix['matrix']
        labels = self.confusion_matrix['labels']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='Blues')
        
        # Show all ticks and label them
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                       ha="center", va="center", 
                       color="white" if matrix[i, j] > 0.5 else "black")
        
        ax.set_title("Confusion Matrix")
        fig.colorbar(im)
        fig.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
            plt.close()
        else:
            plt.show()
