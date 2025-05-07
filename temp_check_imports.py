import sys
import os

print("Python sys.path:")
for path in sys.path:
    print(f"  - {path}")

print("\nTrying to import modules...")
try:
    # Adding the project root to the path as is common in many projects
    project_root = os.path.abspath(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to sys.path")
    
    # Try different import paths
    try:
        from src.object_interaction_detection.dataloaders.helper_loader.base_loader import BaseDataLoader
        print("Successfully imported BaseDataLoader from src.object_interaction_detection.dataloaders.helper_loader.base_loader")
    except ImportError as e:
        print(f"Failed to import from src.object_interaction_detection...: {e}")
    
    try:
        from object_interaction_detection.dataloaders.helper_loader.base_loader import BaseDataLoader
        print("Successfully imported BaseDataLoader from object_interaction_detection.dataloaders.helper_loader.base_loader")
    except ImportError as e:
        print(f"Failed to import from object_interaction_detection...: {e}")
    
except Exception as e:
    print(f"Error: {e}")
