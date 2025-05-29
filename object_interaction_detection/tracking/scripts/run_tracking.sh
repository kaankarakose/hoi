#!/bin/bash
#activate conda env
# source /home/kaan/miniconda3/etc/profile.d/conda.sh
# conda activate oid

# Check if parameters are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 num_part process_part"
    echo "  num_part: Total number of parts to split the processing into"
    echo "  process_part: Which part to process (0-indexed, from 0 to num_part-1)"
    exit 1
fi

# Get parameters
NUM_PART=$1
PROCESS_PART=$2

# Validate parameters
if [ $PROCESS_PART -ge $NUM_PART ]; then
    echo "Error: process_part ($PROCESS_PART) must be less than num_part ($NUM_PART)"
    exit 1
fi

# Define the output directory
OUTPUT_DIR="/nas/project_data/B1_Behavior/rush/kaan/hoi/outputs/tracking"

# Define arrays for sessions and cameras
# SESSIONS=(
#     "imi_session1_6" "imi_session1_7" "imi_session1_8" "imi_session1_9" 
#     "imi_session2_11" "imi_session2_12" 
#     "imi_session2_15"  "zolti_session1_17" 
#     "zolti_session1_18" "zolti_session1_19" "zolti_session1_20" "zolti_session2_21" 
#     "zolti_session2_24" "zolti_session2_25" )
## Define arrays for sessions and cameras
# SESSIONS=(
#    "zolti_session2_25" )
# SESSIONS=(
#    "imi_session1_2" "imi_session1_4" "imi_session1_5" 
#    "imi_session1_6" "imi_session1_7" "imi_session1_8" "imi_session1_9" 
#    "imi_session2_10" "imi_session2_11" "imi_session2_12" 
#    "imi_session2_14" "imi_session2_15"  "zolti_session1_16" "zolti_session1_17" 
#    "zolti_session1_18" "zolti_session1_19" "zolti_session1_20" "zolti_session2_21" 
#    "zolti_session2_22" "zolti_session2_23" "zolti_session2_24" "zolti_session2_25" )
SESSIONS=(
    "imi_session1_2" 
    "imi_session1_4"
    "imi_session1_5"
    "imi_session1_6"
    "imi_session1_7"
    "imi_session1_8"
    "imi_session1_9"
    "imi_session2_10"
    "imi_session2_11"
    "imi_session2_12"
    "imi_session2_14"
    "imi_session2_15"
    "zolti_session1_16"
    "zolti_session1_17"
    "zolti_session1_18"
    "zolti_session1_19"
    "zolti_session1_20"
    "zolti_session2_21"
    "zolti_session2_22"
    "zolti_session2_23"
    "zolti_session2_24"
    "zolti_session2_25"
    )

CAMERAS=("cam_side_l" "cam_top" )

# Create an array of all combinations
COMBINATIONS=()
for session in "${SESSIONS[@]}"; do
    for camera in "${CAMERAS[@]}"; do
        COMBINATIONS+=("$session:$camera")
    done
done

# Calculate total combinations and how to divide them
TOTAL=${#COMBINATIONS[@]}
COMBINATIONS_PER_PART=$(( (TOTAL + NUM_PART - 1) / NUM_PART ))  # Ceiling division

# Calculate start and end indices for this part
START_IDX=$((PROCESS_PART * COMBINATIONS_PER_PART))
END_IDX=$((START_IDX + COMBINATIONS_PER_PART - 1))

# Make sure end index doesn't exceed the total
if [ $END_IDX -ge $TOTAL ]; then
    END_IDX=$((TOTAL - 1))
fi

# Report part information
echo "Processing part $PROCESS_PART of $NUM_PART"
echo "Total combinations: $TOTAL"
echo "Processing combinations $START_IDX to $END_IDX"
echo "-------------------------------------------"

# Process assigned combinations
for i in $(seq $START_IDX $END_IDX); do
    # Split the combination back into session and camera
    IFS=':' read -r session camera <<< "${COMBINATIONS[$i]}"
    
    echo "[$(( i - START_IDX + 1 ))/$((END_IDX - START_IDX + 1))] Processing session: $session, camera: $camera"
    	
    echo "Memory before running:"
    free -h 
   
    python -c "import gc; gc.collect()"

    # Run the Python script with the current parameters
    python ../object_activeness_tracker.py \
        --session "$session" \
        --camera "$camera"
    sleep 30
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Completed $session - $camera"
    else
        echo "✗ Error processing $session - $camera"
    fi
    
    echo "-------------------------------------------"
done

echo "All assigned combinations processed!"
