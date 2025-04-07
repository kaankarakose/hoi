# Multi-View Prediction Merging Pipeline

This pipeline allows merging object predictions from multiple camera views based on their scores, assuming that the highest score belongs to the same object/mask across views.

## Overview

The pipeline processes predictions from multiple camera views (e.g., `cam_side_l`, `cam_side_r`, `cam_top`) for a specific session and object. For each frame, it collects predictions from all views and merges them based on a chosen strategy.

## Directory Structure

The input data is expected to follow this structure:
```
/base_dir/
├── session_name/
│   ├── camera_view_1/
│   │   ├── object_name/
│   │   │   ├── bbox/
│   │   │   ├── frames/
│   │   │   ├── masks/
│   │   │   │   ├── frame_XXXX/
│   │   │   │   │   ├── all_masks.json
│   │   │   │   │   ├── mask_0.rle
│   │   │   │   │   └── ...
│   │   │   └── scores/
│   │   │       ├── frame_XXXX/
│   │   │       │   ├── score_0.txt
│   │   │       │   └── ...
│   ├── camera_view_2/
│   └── ...
```

The output follows a similar structure under the output directory.

## Files

- `utils.py`: Helper functions for reading/writing scores and masks
- `merge_views.py`: Main implementation of the merging pipeline
- `run_merge.py`: Helper script to process multiple sessions and objects

## Installation

Requirements:
- Python 3.6+
- NumPy
- OpenCV
- pycocotools
- tqdm

You can install the requirements with:
```bash
pip install numpy opencv-python pycocotools tqdm
```

## Usage

### Process a specific session and object

```bash
python merge_views.py --session imi_session1_2 --object AMF1
```

### Custom camera views and merge strategy

```bash
python merge_views.py --session imi_session1_2 --object AMF1 --cameras cam_side_l cam_side_r --strategy union
```

### Process multiple sessions and objects

```bash
python run_merge.py --sessions imi_session1_2 --objects AMF1 AMF2
```

### List available sessions and objects

```bash
python run_merge.py --list-only
```

## Merging Strategies

- `highest`: Use the mask with the highest score (default)
- `average`: Create a weighted average of masks based on scores
- `union`: Take the union of all masks

## Example

```bash
# First make the scripts executable
chmod +x merge_views.py run_merge.py

# Run for a specific session and object with all three camera views
./merge_views.py --session imi_session1_2 --object AMF1 --cameras cam_side_l cam_side_r cam_top

# Use the run_merge script to process multiple objects
./run_merge.py --sessions imi_session1_2 --objects AMF1 AMF2 AMF3 --strategy highest
```

The merged results will be saved to the output directory (default: `/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/multi_view_results`).
