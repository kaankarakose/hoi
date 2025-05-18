#!/bin/bash

# Define the output directory
OUTPUT_DIR="/nas/project_data/B1_Behavior/rush/kaan/hoi/outputs/tracking"

# Define arrays for sessions and cameras
SESSIONS=(
    "imi_session1_2" "imi_session1_3" "imi_session1_4" "imi_session1_5" 
    "imi_session1_6" "imi_session1_7" "imi_session1_8" "imi_session1_9" 
    "imi_session2_1" "imi_session2_10" "imi_session2_11" "imi_session2_12" 
    "imi_session2_14" "imi_session2_15" )

CAMERAS=("cam_top" "cam_side_r" "cam_side_l")

# Count total number of combinations for progress tracking
TOTAL=$((${#SESSIONS[@]} * ${#CAMERAS[@]}))
CURRENT=0

echo "Starting processing of $TOTAL session-camera combinations..."

# Loop through each session and camera combination
for session in "${SESSIONS[@]}"; do
    for camera in "${CAMERAS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo "[$CURRENT/$TOTAL] Processing session: $session, camera: $camera"
        
        # Run the Python script with the current parameters
        python object_activeness_tracker.py \
            --session "$session" \
            --camera "$camera" \
            --output-dir "$OUTPUT_DIR"
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "✓ Completed $session - $camera"
        else
            echo "✗ Error processing $session - $camera"
        fi
        
        echo "-------------------------------------------"
    done
done

echo "All processing complete!"