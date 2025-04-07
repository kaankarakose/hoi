export LIGHTING_ITENSITY=0.6 # lighting intensity
export RADIUS=0.3 # distance to camera
python -m src.poses.pyrender $CAD_PATH ./src/poses/predefined_poses/obj_poses_level2.npy $OUTPUT_DIR 0 False $LIGHTING_ITENSITY $RADIUS