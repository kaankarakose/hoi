#!/usr/bin/env python3
import os
import glob
import pandas as pd
from pathlib import Path
import argparse

class VideoManager:
    def __init__(self, base_dir="/nas/project_data/B1_Behavior/rush/mb_inhouse/mb_imi_zolti_/mb"):
        self.base_dir = Path(base_dir)
        print(f"Base directory: {base_dir}")

    def list_camera_views(self):
        """List all camera view directories (cam_side_l, cam_side_r, cam_top)"""
        return [d for d in self.base_dir.glob("cam_*") if d.is_dir()]
    
    def list_folders(self, camera_dir):
        """List all folders (calibration, imi, zolti) for a given camera view"""
        return [d for d in camera_dir.glob("*") if d.is_dir()]
    
    def list_persons(self, camera_dir):
        """List only person folders (imi, zolti) for a given camera view, excluding calibration"""
        return [d for d in camera_dir.glob("*") if d.is_dir() and d.name != "calibration"]
        
    def list_sessions(self, person_dir):
        """List all session directories for a given person"""
        return [d for d in person_dir.glob("session*") if d.is_dir()]
    
    def list_videos(self, session_dir):
        """List all video files in a session directory"""
        video_extensions = ["*.mp4", "*.avi", "*.mov"]
        videos = []
        for ext in video_extensions:
            videos.extend(session_dir.glob(ext))
        return videos
    
    def create_inventory(self):
        """Create a comprehensive inventory of all videos"""
        records = []
        
        for camera_dir in self.list_camera_views():
            camera_view = camera_dir.name
            
            for person_dir in self.list_persons(camera_dir):
                person = person_dir.name
                
                for session_dir in self.list_sessions(person_dir):
                    session = session_dir.name
                    
                    # Sort videos within this session directory
                    videos = sorted(self.list_videos(session_dir))
                    
                    # Assign index-based sync identifiers based on sort order
                    for index, video_file in enumerate(videos):
                        file_size_mb = video_file.stat().st_size / (1024 * 1024)
                        
                        # Use a combination of person, session, and index as the sync group
                        # This ensures videos at the same position in different camera folders are paired
                        sync_group = f"{person}_{session}_{index}"
                        
                        records.append({
                            "camera_view": camera_view,
                            "person": person,
                            "session": session,
                            "filename": video_file.name,
                            "path": str(video_file),
                            "size_mb": round(file_size_mb, 2),
                            "sync_group": sync_group,
                            "position_index": index
                        })
            
            # Handle calibration folder separately if it exists
            calibration_dir = camera_dir / "calibration"
            if calibration_dir.exists() and calibration_dir.is_dir():
                # Sort calibration videos
                videos = sorted(self.list_videos(calibration_dir))
                
                for index, video_file in enumerate(videos):
                    file_size_mb = video_file.stat().st_size / (1024 * 1024)
                    
                    # Use a combination of 'calibration' and index as the sync group
                    sync_group = f"calibration_{index}"
                    
                    records.append({
                        "camera_view": camera_view,
                        "person": "calibration",
                        "session": "",  # No session for calibration
                        "filename": video_file.name,
                        "path": str(video_file),
                        "size_mb": round(file_size_mb, 2),
                        "sync_group": sync_group,
                        "position_index": index
                    })
        
        return pd.DataFrame(records)
    
    def export_inventory(self, output_path="video_inventory.csv", include_sync_pairs=True):
        """Export the inventory to CSV with synchronized pair information
        
        Args:
            output_path (str): Path to save the inventory CSV
            include_sync_pairs (bool): Whether to include synchronized pair information
        """
        inventory = self.create_inventory()
        
        if include_sync_pairs:
            # Create columns for synchronized pairs
            inventory['has_synchronized_pairs'] = False
            inventory['synchronized_cam_side_l'] = ''
            inventory['synchronized_cam_side_r'] = ''
            inventory['synchronized_cam_top'] = ''
            
            # Identify synchronized pairs based on position in sorted directory listings
            # Group by sync_group which is based on person+session+position_index
            grouped_by_sync = inventory.groupby('sync_group')
            
            # Process each sync group
            for sync_group_name, group_data in grouped_by_sync:
                # Skip if there's only one camera view in this group
                camera_views = set(group_data['camera_view'])
                if len(camera_views) < 2:
                    continue
                    
                # Mark all videos in this sync group as having synchronized pairs
                mask = inventory['sync_group'] == sync_group_name
                inventory.loc[mask, 'has_synchronized_pairs'] = True
                
                # Create a lookup dictionary for each camera view's paths
                camera_paths = {}
                for camera in camera_views:
                    # Get the video path for this camera view
                    cam_data = group_data[group_data['camera_view'] == camera]
                    if not cam_data.empty:
                        camera_paths[camera] = cam_data.iloc[0]['path']
                
                # For each video in this sync group, add the paths to videos from other cameras
                for idx, row in inventory[mask].iterrows():
                    current_camera = row['camera_view']
                    # Add paths for each available camera view
                    for camera, path in camera_paths.items():
                        if camera != current_camera:
                            inventory.at[idx, f'synchronized_{camera}'] = path
        
        # Remove the position_index column as it was only used internally
        if 'position_index' in inventory.columns:
            inventory = inventory.drop('position_index', axis=1)
        
        inventory.to_csv(output_path, index=False)
        print(f"Inventory saved to {output_path}")
        return inventory
    
    def find_videos(self, camera=None, person=None, session=None, filename_pattern=None, sync_group=None):
        """Search for videos matching specific criteria"""
        inventory = self.create_inventory()
        
        if camera:
            inventory = inventory[inventory['camera_view'].str.contains(camera, case=False)]
        if person:
            inventory = inventory[inventory['person'].str.contains(person, case=False)]
        if session:
            inventory = inventory[inventory['session'].str.contains(session, case=False)]
        if filename_pattern:
            inventory = inventory[inventory['filename'].str.contains(filename_pattern, case=False)]
        if sync_group:
            inventory = inventory[inventory['sync_group'].str.contains(sync_group, case=False)]
            
        return inventory
        
    def find_synchronized_videos(self, sync_group=None, person=None, session=None):
        """Find synchronized videos across different camera angles"""
        inventory = self.create_inventory()
        
        # Filter by provided criteria
        if sync_group:
            inventory = inventory[inventory['sync_group'].str.contains(sync_group, case=False)]
        if person:
            inventory = inventory[inventory['person'].str.contains(person, case=False)]
        if session:
            inventory = inventory[inventory['session'].str.contains(session, case=False)]
        
        # Group by sync_group and filter for groups that have videos from all three cameras
        grouped = inventory.groupby('sync_group')
        complete_sets = []
        
        for group_name, group_data in grouped:
            camera_views = set(group_data['camera_view'])
            # Check if we have all three camera angles
            if len(camera_views) == 3 and 'cam_side_l' in camera_views and 'cam_side_r' in camera_views and 'cam_top' in camera_views:
                complete_sets.append(group_data)
        
        if not complete_sets:
            return pd.DataFrame()
        
        # Combine all complete sets into a single DataFrame
        result = pd.concat(complete_sets).sort_values(['sync_group', 'camera_view'])
        return result
        
    def find_all_synchronized_videos(self):
        """Find all synchronized videos across different camera angles without any filtering"""
        inventory = self.create_inventory()
        
        # Group by sync_group and filter for groups that have videos from all three cameras
        grouped = inventory.groupby('sync_group')
        complete_sets = []
        
        for group_name, group_data in grouped:
            camera_views = set(group_data['camera_view'])
            # Check if we have all three camera angles
            if len(camera_views) >= 2:  # Consider partial synchronization (at least 2 cameras)
                complete_sets.append(group_data)
        
        if not complete_sets:
            return pd.DataFrame()
        
        # Combine all complete sets into a single DataFrame
        result = pd.concat(complete_sets).sort_values(['sync_group', 'camera_view'])
        return result

def main():
    parser = argparse.ArgumentParser(description="Manage video recordings")
    parser.add_argument("--base-dir", default="/nas/project_data/B1_Behavior/rush/mb_inhouse/mb_imi_zolti_/mb", 
                        help="Base directory for video data")
    parser.add_argument("--inventory", action="store_true", 
                        help="Create and export inventory to CSV")
    parser.add_argument("--output", default="video_inventory.csv", 
                        help="Output path for inventory CSV")
    parser.add_argument("--camera", help="Filter by camera view (cam_side_l, cam_side_r, cam_top)")
    parser.add_argument("--person", help="Filter by person (imi, zolti, calibration)")
    parser.add_argument("--session", help="Filter by session (session1, session2)")
    parser.add_argument("--filename", help="Filter by filename pattern")
    parser.add_argument("--sync", action="store_true",
                        help="Find synchronized video sets across all cameras")
    parser.add_argument("--sync-group", help="Filter by sync group identifier")
    parser.add_argument("--no-sync-pairs", action="store_true",
                        help="Don't include synchronized pair information in the inventory")
    parser.add_argument("--verbose", action="store_true",
                        help="Print additional information during processing")
    
    args = parser.parse_args()
    
    manager = VideoManager(args.base_dir)
    
    if args.inventory:
        manager.export_inventory(args.output, not args.no_sync_pairs)
    elif args.sync:
        # Find synchronized videos across all cameras
        results = manager.find_synchronized_videos(
            sync_group=args.sync_group,
            person=args.person,
            session=args.session
        )
        
        if len(results) == 0:
            print("No synchronized video sets found matching the criteria.")
        else:
            # Count number of unique sync groups
            sync_count = results['sync_group'].nunique()
            print(f"Found {sync_count} synchronized video sets ({len(results)} videos total):")
            pd.set_option('display.max_colwidth', None)
            print(results[['sync_group', 'camera_view', 'person', 'session', 'filename', 'size_mb']])
    elif any([args.camera, args.person, args.session, args.filename, args.sync_group]):
        results = manager.find_videos(
            camera=args.camera,
            person=args.person,
            session=args.session,
            filename_pattern=args.filename,
            sync_group=args.sync_group
        )
        
        if len(results) == 0:
            print("No videos found matching the criteria.")
        else:
            print(f"Found {len(results)} videos:")
            pd.set_option('display.max_colwidth', None)
            print(results[['camera_view', 'person', 'session', 'filename', 'size_mb']])
    else:
        # Display summary statistics
        inventory = manager.create_inventory()
        
        # Count by camera view
        camera_counts = inventory['camera_view'].value_counts()
        print("\nVideos by camera view:")
        print(camera_counts)
        
        # Count by person
        person_counts = inventory['person'].value_counts()
        print("\nVideos by person:")
        print(person_counts)
        
        # Count by session
        session_counts = inventory['session'].value_counts()
        print("\nVideos by session:")
        print(session_counts)
        
        total_size_gb = inventory['size_mb'].sum() / 1024
        print(f"\nTotal storage used: {total_size_gb:.2f} GB")
        
if __name__ == "__main__":
    main()
