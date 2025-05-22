"""
Visualization module for evaluation results.

This script provides command-line functionality to visualize
evaluation results from JSON files.
"""

import os
import json
import argparse
import logging
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_results(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load evaluation results from JSON files in the given directory.
    
    Args:
        results_dir: Root directory containing evaluation results
        
    Returns:
        Dictionary mapping session+camera to result data
    """
    results = {}
    
    # Find all result JSON files recursively
    json_files = glob.glob(os.path.join(results_dir, "**/*evaluation.json"), recursive=True)
    
    if not json_files:
        logger.warning(f"No evaluation result files found in {results_dir}")
        return results
    
    # Load each file
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Get session and camera from file path or contents
            session_name = data.get('session_name', 'unknown_session')
            camera_view = data.get('camera_view', 'unknown_camera')
            
            results[f"{session_name}_{camera_view}"] = data
            logger.info(f"Loaded results for {session_name}, camera {camera_view}")
        
        except Exception as e:
            logger.error(f"Error loading result file {file_path}: {e}")
    
    return results


def plot_hit_rates(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot hit rates (recall) for different sessions and camera views.
    
    Args:
        results: Dictionary mapping session+camera to result data
        save_path: Path to save plot image (optional)
    """
    hit_rates = {}
    
    # Extract hit rates
    for key, data in results.items():
        if 'metrics' in data and 'hit_rate' in data['metrics']:
            hit_rates[key] = data['metrics']['hit_rate']
    
    if not hit_rates:
        logger.warning("No hit rate data found in results")
        return
    
    # Sort by hit rate value
    hit_rates = {k: v for k, v in sorted(hit_rates.items(), key=lambda item: item[1], reverse=True)}
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(hit_rates)), list(hit_rates.values()), align='center')
    plt.xticks(range(len(hit_rates)), list(hit_rates.keys()), rotation=45, ha='right')
    plt.ylabel('Hit Rate')
    plt.title('Hit Rate by Session/Camera')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved hit rate plot to {save_path}")
    else:
        plt.show()


def plot_timing_errors(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot average timing errors for different sessions and camera views.
    
    Args:
        results: Dictionary mapping session+camera to result data
        save_path: Path to save plot image (optional)
    """
    timing_errors = {}
    
    # Extract timing errors
    for key, data in results.items():
        if 'metrics' in data and 'avg_time_diff' in data['metrics']:
            timing_errors[key] = data['metrics']['avg_time_diff']
    
    if not timing_errors:
        logger.warning("No timing error data found in results")
        return
    
    # Sort by absolute timing error value
    timing_errors = {k: v for k, v in sorted(timing_errors.items(), key=lambda item: abs(item[1]))}
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Use different colors for early/late detections
    bars = plt.bar(range(len(timing_errors)), list(timing_errors.values()), align='center')
    for i, v in enumerate(timing_errors.values()):
        if v < 0:
            bars[i].set_color('blue')  # Early detection
        else:
            bars[i].set_color('red')   # Late detection
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xticks(range(len(timing_errors)), list(timing_errors.keys()), rotation=45, ha='right')
    plt.ylabel('Timing Error (seconds)')
    plt.title('Average Timing Error by Session/Camera')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved timing error plot to {save_path}")
    else:
        plt.show()


def plot_object_hits(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot hit rates for different objects across all sessions.
    
    Args:
        results: Dictionary mapping session+camera to result data
        save_path: Path to save plot image (optional)
    """
    # Collect hit rates for all objects
    object_hits = {}
    
    for key, data in results.items():
        if 'object_metrics' in data:
            for obj_name, metrics in data['object_metrics'].items():
                if 'hit_rate' in metrics:
                    if obj_name not in object_hits:
                        object_hits[obj_name] = []
                    object_hits[obj_name].append(metrics['hit_rate'])
    
    if not object_hits:
        logger.warning("No object hit data found in results")
        return
    
    # Calculate average hit rate for each object
    avg_hits = {}
    for obj_name, hits in object_hits.items():
        avg_hits[obj_name] = np.mean(hits)
    
    # Sort by hit rate
    avg_hits = {k: v for k, v in sorted(avg_hits.items(), key=lambda item: item[1], reverse=True)}
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(avg_hits)), list(avg_hits.values()), align='center')
    plt.xticks(range(len(avg_hits)), list(avg_hits.keys()), rotation=45, ha='right')
    plt.ylabel('Average Hit Rate')
    plt.title('Average Hit Rate by Object')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved object hit plot to {save_path}")
    else:
        plt.show()


def create_summary_table(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """
    Create a summary table of evaluation results.
    
    Args:
        results: Dictionary mapping session+camera to result data
        save_path: Path to save table as CSV (optional)
    """
    # Prepare data for summary table
    summary_data = []
    
    for key, data in results.items():
        session = data.get('session_name', 'unknown')
        camera = data.get('camera_view', 'unknown')
        
        # Get metrics
        metrics = data.get('metrics', {})
        hit_rate = metrics.get('hit_rate', float('nan'))
        avg_time_diff = metrics.get('avg_time_diff', float('nan'))
        std_time_diff = metrics.get('std_time_diff', float('nan'))
        
        # Count objects
        total_objects = len(data.get('first_annotation_times', {}))
        detected_objects = sum(1 for metrics in data.get('object_metrics', {}).values() 
                              if metrics.get('hit_rate', 0) > 0)
        
        summary_data.append({
            'Session': session,
            'Camera': camera,
            'Total Objects': total_objects,
            'Detected Objects': detected_objects,
            'Hit Rate': hit_rate,
            'Avg Time Diff (s)': avg_time_diff,
            'Std Time Diff (s)': std_time_diff
        })
    
    if not summary_data:
        logger.warning("No data for summary table")
        return
    
    # Create dataframe and print
    df = pd.DataFrame(summary_data)
    print("\nSummary Table:")
    print(df.to_string(index=False))
    
    # Save if requested
    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Saved summary table to {save_path}")


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    
    # Input options
    parser.add_argument("--results-dir", type=str, 
                      default="/nas/project_data/B1_Behavior/rush/kaan/hoi/outputs/evaluation",
                      help="Directory containing evaluation result JSON files")
    
    # Output options
    parser.add_argument("--output-dir", type=str, 
                      help="Directory to save visualizations (default: results_dir/visualizations)")
    
    # Visualization options
    parser.add_argument("--all", action="store_true", 
                      help="Generate all visualizations")
    parser.add_argument("--hit-rates", action="store_true", 
                      help="Plot hit rates by session/camera")
    parser.add_argument("--timing-errors", action="store_true", 
                      help="Plot timing errors by session/camera")
    parser.add_argument("--object-hits", action="store_true", 
                      help="Plot hit rates by object")
    parser.add_argument("--summary-table", action="store_true", 
                      help="Create summary table")
    
    args = parser.parse_args()
    
    # Determine what to visualize
    visualize_hit_rates = args.all or args.hit_rates
    visualize_timing_errors = args.all or args.timing_errors
    visualize_object_hits = args.all or args.object_hits
    create_table = args.all or args.summary_table
    
    # If no specific visualization requested, do all
    if not (visualize_hit_rates or visualize_timing_errors or 
            visualize_object_hits or create_table):
        visualize_hit_rates = True
        visualize_timing_errors = True
        visualize_object_hits = True
        create_table = True
    
    # Set output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(args.results_dir, "visualizations")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        logger.error(f"No results found in {args.results_dir}")
        return
    
    # Generate visualizations
    if visualize_hit_rates:
        hit_rates_path = os.path.join(output_dir, "hit_rates.png")
        plot_hit_rates(results, hit_rates_path)
    
    if visualize_timing_errors:
        timing_errors_path = os.path.join(output_dir, "timing_errors.png")
        plot_timing_errors(results, timing_errors_path)
    
    if visualize_object_hits:
        object_hits_path = os.path.join(output_dir, "object_hits.png")
        plot_object_hits(results, object_hits_path)
    
    if create_table:
        summary_table_path = os.path.join(output_dir, "summary_table.csv")
        create_summary_table(results, summary_table_path)
    
    # Print information about saved visualizations
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
