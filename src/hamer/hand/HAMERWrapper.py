from pathlib import Path
import torch
import cv2
import numpy as np
import os
import sys

# Add thirdparty paths to system path for proper imports
thirdparty_path = Path('/nas/project_data/B1_Behavior/rush/ados-objects/object_pose_mega/rtdt/hand-object/hand/thirdparty')
if str(thirdparty_path) not in sys.path:
    sys.path.append(str(thirdparty_path))

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

# Import ViTPoseModel directly from thirdparty/hamer directory
vitpose_model_path = thirdparty_path / 'hamer/vitpose_model.py'
import importlib.util
spec = importlib.util.spec_from_file_location("vitpose_model", vitpose_model_path)
vitpose_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vitpose_module)
ViTPoseModel = vitpose_module.ViTPoseModel

class Hand:
    """Class representing a detected hand with useful methods"""
    
    def __init__(self, vertices, cam_t, is_right, bbox=None, keypoints=None, score = None):
        """
        Initialize a Hand object
        
        Args:
            vertices: 3D mesh vertices
            cam_t: Camera translation
            is_right: Boolean indicating if it's a right hand
            bbox: 2D bounding box [x1, y1, x2, y2]
            keypoints: Hand keypoints if available
        """
        self.vertices = vertices
        self.cam_t = cam_t
        self.is_right = is_right
        self.bbox = bbox
        self.keypoints = keypoints
        ## TODO Get score of the hand for each. 
        self.score = score
        self._mask = None
    
    def boundingbox(self):
        """Get the 2D bounding box of the hand"""
        return self.bbox
    
    def mask(self, img_shape=None):
        """
        Get or generate a binary mask for the hand
        
        Args:
            img_shape: If provided, generates mask of this shape
        
        Returns:
            Binary mask of the hand
        """
        if self._mask is not None and img_shape is None:
            return self._mask
        
        # Generate mask based on projected vertices
        # This is a placeholder implementation - would need renderer to do properly
        if img_shape is not None:
            # Create a mask of the right shape
            mask = np.zeros(img_shape[:2], dtype=np.uint8)
            
            # If we have a bbox, fill that region as a simple approximation
            if self.bbox is not None:
                x1, y1, x2, y2 = [int(x) for x in self.bbox]
                mask[y1:y2, x1:x2] = 255
                
            self._mask = mask
        return self._mask

    def get_mesh(self):
        """Return the hand mesh vertices"""
        return self.vertices


class HAMERWrapper:
    """Wrapper class for the HAMER hand model with improved interface"""
    
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, body_detector='vitdet', device=None):
        """
        Initialize the HAMER model and all required components
        
        Args:
            checkpoint: Path to model checkpoint
            body_detector: Type of body detector to use ('vitdet' or 'regnety')
            device: Device to run the model on (None for auto-detection)
        """
        # Set device
        self.device = torch.device('cuda') if torch.cuda.is_available() and device is None else torch.device(device or 'cpu')
        
        # Download and load checkpoints
        download_models(CACHE_DIR_HAMER)
        self.model, self.model_cfg = load_hamer(checkpoint)
        
        # Setup model
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load detector
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        if body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            import hamer
            cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif body_detector == 'regnety':
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        
        # Initialize keypoint detector
        try:
            # Patch the MODEL_DICT with absolute paths before creating the instance
            class ViTPoseModelPatched(ViTPoseModel):
                MODEL_DICT = {
                    'ViTPose+-G (multi-task train, COCO)': {
                        'config': str(thirdparty_path / 'hamer/third-party/ViTPose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py'),
                        'model': str(thirdparty_path / 'hamer/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth'),
                    },
                }
            self.cpm = ViTPoseModelPatched(self.device)
        except Exception as e:
            print(f"Error loading ViTPose model: {e}")
        
        # Setup renderer
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
        
        # Initialize state
        self.left_hands = list()
        self.right_hands = list()
        self.current_image = None
        self.rescale_factor = 2.0  # Default rescale factor
    
    def process_frame(self, img):
        """
        Process a frame to detect and reconstruct hands
        
        Args:
            img: Input image (BGR format from cv2.imread)
            
        Returns:
            True if at least one hand was detected, False otherwise
        """
        self.current_image = img
        self.left_hands = list()
        self.right_hands = list()
        # Detect humans in image
        det_out = self.detector(img)
        img_rgb = img.copy()[:, :, ::-1]  # Convert to RGB for pose model
        
        det_instances = det_out['instances']
        
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.7)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        
        # If no person detected, return False
        if len(pred_bboxes) == 0:
            return False

        # Detect human keypoints for each person
        vitposes_out = self.cpm.predict_pose(
            img_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        ) 
        
        bboxes = []
        is_right = []
        keypoints_list = []
        #print('len',len(vitposes_out))

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:

            left_hand_keyp = vitposes['keypoints'][-42:-21] #TODO check here!
            right_hand_keyp = vitposes['keypoints'][-21:]
            
            # Check left hand
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)
                keypoints_list.append(keyp)
            
            # Check right hand
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)
                keypoints_list.append(keyp)
        
        if len(bboxes) == 0:
            return False
        
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        
        # Run reconstruction on all detected hands - 3D
        dataset = ViTDetDataset(self.model_cfg, img, boxes, right, rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        all_verts = []
        all_cam_t = []
        all_right = []
        all_bboxes = []
        all_keypoints = []
        
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)
            
            # Process camera parameters
            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam'] ## TODO: I have camera parameters!!
            pred_cam[:, 1] = multiplier*pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            
            # Process results for each hand
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right_hand = batch['right'][n].cpu().numpy()
                
                # Flip vertices for correct orientation
                verts[:, 0] = (2*is_right_hand-1)*verts[:, 0]
                cam_t = pred_cam_t_full[n]
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right_hand)
                
                # Store bounding boxes and keypoints using simpler approach
                # Map each detected hand to appropriate box/keypoints
                # Since we know the right/left hand information, use that to match
                is_right_hand_bool = bool(is_right_hand)
                
                # Find a matching box with the same handedness
                matched_box_idx = None
                for idx, (box, is_right_val) in enumerate(zip(boxes, right)):
                    if bool(is_right_val) == is_right_hand_bool:
                        matched_box_idx = idx
                        break
                
                if matched_box_idx is not None and matched_box_idx < len(boxes):
                    all_bboxes.append(boxes[matched_box_idx])
                    
                    if matched_box_idx < len(keypoints_list):
                        all_keypoints.append(keypoints_list[matched_box_idx])
                    else:
                        all_keypoints.append(None)
                else:
                    # If no matching box found, use a default approach
                    if n < len(boxes):
                        all_bboxes.append(boxes[n])
                        if n < len(keypoints_list):
                            all_keypoints.append(keypoints_list[n])
                        else:
                            all_keypoints.append(None)
                    else:
                        print(f"Warning: Could not find matching box for hand {n} (right={is_right_hand_bool})")
                        all_bboxes.append(None)
                        all_keypoints.append(None)
        
        # Create Hand objects
        for i, (verts, cam_t, is_right_hand, bbox, keypoints) in enumerate(
            zip(all_verts, all_cam_t, all_right, all_bboxes, all_keypoints)
        ):
            # Skip if bbox is None
            if bbox is None:
                continue
                
            hand = Hand(verts, cam_t, is_right_hand, bbox, keypoints)

            if is_right_hand:
                self.right_hands.append(hand) ##
            else:
                self.left_hands.append(hand)
 
        return len(all_verts) > 0

    def getRightHands(self):
        """Get the right hand object if detected"""
        return self.right_hands
    
    def getLeftHands(self):
        """Get the left hand object if detected"""
        return self.left_hands
    
    def getHands(self):
        """Get both hands (may be None if not detected)"""
        return {
            'left': self.left_hands,
            'right': self.right_hands
        }
    
    def render(self, image=None, side_view=False, save_path=None):
        """
        Render the detected hands on the image
        
        Args:
            image: Image to render on (uses stored image if None)
            side_view: Whether to include a side view
            save_path: Path to save the rendered image
            
        Returns:
            Rendered image
        """
        if image is None:
            image = self.current_image
        
        if image is None:
            return None
        
        img_rgb = image.copy()[:, :, ::-1]
        
        # Collect all hands to render
        all_verts = []
        all_cam_t = []
        all_right = []
        
        if self.right_hands != []:
            try:
                all_verts.append(self.right_hands[0].vertices)
                all_cam_t.append(self.right_hands[0].cam_t)
                all_right.append(1)
            except:
                print(self.right_hands)
        
        if self.left_hands != []:
            all_verts.append(self.left_hands[0].vertices)
            all_cam_t.append(self.left_hands[0].cam_t)
            all_right.append(0)
        
        if not all_verts:
            return img_rgb
        
        # Render the hands
        img_size = np.array([image.shape[1], image.shape[0]])
        scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * max(img_size)
        
        LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
        misc_args = dict(
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )
        
        # Render RGBA and overlay on image
        cam_view = self.renderer.render_rgba_multiple(
            all_verts, cam_t=all_cam_t, render_res=img_size, is_right=all_right, **misc_args
        )
        
        # Combine with original image
        alpha = cam_view[:, :, 3:4]
        rgb = cam_view[:, :, :3]
        rendered_img = rgb * alpha + img_rgb * (1 - alpha)
        
        if save_path:
            cv2.imwrite(save_path, rendered_img[:, :, ::-1])
        
        return rendered_img


def demo():
    """Demo code to show usage of the wrapper"""
    # Initialize the model
    try:
        hamer = HAMERWrapper()
    except Exception as e:
        print(f"Error initializing HAMERWrapper: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load an image
    img_path = "/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/orginal_frames/imi_session1_2/cam_side_l"
    if not os.path.exists(img_path):
        print(f"Demo image not found: {img_path}")
        # Try to find any image in the example_data directory
        example_data_dir = thirdparty_path / "hamer/example_data"
        image_files = list(example_data_dir.glob("*.jpg")) + list(example_data_dir.glob("*.png"))
        if image_files:
            img_path = str(image_files[0])
            print(f"Using alternative image: {img_path}")
        else:
            print("No test images found in example_data directory")
            return
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
        
    print(f"Image shape: {img.shape}")
    
    # Process the frame
    print("Processing frame...")
    try:
        success = hamer.process_frame(img)
        if not success:
            print("No hands detected")
            return
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Use the interface
    right_hands = hamer.getRightHands()
    left_hands = hamer.getLeftHands()
    
    print("Detected hands:")
    if right_hands:
        print(f"Right hand bounding box: {right_hand.boundingbox()}")
    if left_hands:
        print(f"Left hand bounding box: {left_hand.boundingbox()}")
    
    # Render the result
    print("Rendering result...")
    try:
        demo_out_dir = thirdparty_path / "hamer/demo_out"
        demo_out_dir.mkdir(exist_ok=True)
        save_path = str(demo_out_dir / "result.png")
        rendered = hamer.render(save_path=save_path)
        print(f"Result saved to: {save_path}")
    except Exception as e:
        print(f"Error rendering: {e}")
        import traceback
        traceback.print_exc()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    demo()