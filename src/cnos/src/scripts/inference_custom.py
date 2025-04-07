import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from src.utils.bbox_utils import CropResizePad
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
from src.model.utils import Detections, convert_npz_to_json
from src.model.loss import Similarity
from src.utils.inout import save_json_bop23
import cv2
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

def visualize(rgb, detections, save_path="./tmp/tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # img = (255*img).astype(np.uint8)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    masks = getattr(detections, 'masks')
    object_ids = getattr(detections, 'object_ids')
    
    for mask_idx in range(len(detections)):
        # Convert mask to numpy and threshold to get boolean mask
        mask = masks[mask_idx]
        mask = mask > 0.5  # Convert to boolean mask
        edge = canny(mask.astype(np.uint8) * 255)
        edge = binary_dilation(edge, np.ones((2, 2)))
        obj_id = object_ids[mask_idx].item()
        temp_id = mask_idx  # Using mask_idx as temp_id since we're visualizing detections

        r = int(255*colors[temp_id][0])
        g = int(255*colors[temp_id][1])
        b = int(255*colors[temp_id][2])
        img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
        img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
        img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
        img[edge, :] = 255
    
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat
        
def run_inference(template_dir, rgb_path, num_max_dets, conf_threshold, stability_score_thresh):
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name='run_inference.yaml')
    cfg_segmentor = cfg.model.segmentor_model
    if "fast_sam" in cfg_segmentor._target_:
        logging.info("Using FastSAM, ignore stability_score_thresh!")
    else:
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    metric = Similarity()
    logging.info("Initializing model")
    model = instantiate(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")
        
    
    logging.info("Initializing template")
    template_paths = glob.glob(f"{template_dir}/*.png")
    boxes, templates = [], []
    for path in template_paths:
        image = Image.open(path)
        boxes.append(image.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        templates.append(image)
        
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))
    
    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = CropResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates, boxes=boxes).cuda()
    save_image(templates, f"{template_dir}/cnos_results/templates.png", nrow=7)
    ref_feats = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                )
    logging.info(f"Ref feats: {ref_feats.shape}")
    print('\nLoading reference features...')
    print(f'Reference features shape: {ref_feats.shape}')
    
    print('\nRunning inference...')
    print('Loading and processing input image...')
    # run inference
    rgb = Image.open(rgb_path).convert("RGB")
    print(f'Input image size: {rgb.size}')
    print('\nGenerating masks with SAM...')
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    print(detections.keys())
    # for i, mask in enumerate(detections['masks']):
    #     print(type(mask))
    #     cv2.imwrite(f"/nas/project_data/B1_Behavior/rush/ados-objects/object_pose_mega/rtdt/cnos/cnos/tmp/cnos_results/mask_{i}.png", mask.cpu().numpy()*255.)
    #     print(mask.shape)
    # raise ValueError
    detections = Detections(detections)

    print('\nGenerating descriptors...')
    decriptors = model.descriptor_model.forward(np.array(rgb), detections)
    print(f'Descriptor shape: {decriptors.shape}')
   
    print('\nCalculating similarity scores...')
    # get scores per proposal
    scores = metric(decriptors[:, None, :], ref_feats[None, :, :])
    print(f'Raw scores shape: {scores.shape}')
    score_per_detection = torch.topk(scores, k=10, dim=-1)[0]
    score_per_detection = torch.mean(
        score_per_detection, dim=-1
    )
    print(f'Mean scores shape: {score_per_detection.shape}')
    
    print(f"{num_max_dets=}")
    # get top-k detections
    scores, index = torch.topk(score_per_detection, k=num_max_dets * 3 , dim=-1)
    detections.filter(index)
    
    print(f"{conf_threshold=}")
    # keep only detections with score > conf_threshold
    detections.filter(scores>conf_threshold)
    print(f'{scores=}')
    detections.add_attribute("scores", scores)
    detections.add_attribute("object_ids", torch.zeros_like(scores))
    print(f"object_ids={getattr(detections, 'object_ids')}")
    print('\nConverting detections to numpy and saving...')
    detections.to_numpy()
    save_path = f"{template_dir}/cnos_results/detection"
    print(f'Saving results to: {save_path}')
    detections.save_to_file(0, 0, 0, save_path, "custom", return_results=False)
    
    # First visualize with original detections
    vis_img = visualize(rgb, detections)
    vis_img.save(f"{template_dir}/cnos_results/vis.png")
    
    # Then convert to JSON format
    json_detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path+".json", json_detections)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_dir", nargs="?", help="Path to root directory of the template")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--num_max_dets", nargs="?", default=1, type=int, help="Number of max detections")
    parser.add_argument("--confg_threshold", nargs="?", default=0.52, type=float, help="Confidence threshold")
    parser.add_argument("--stability_score_thresh", nargs="?", default=0.95, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()



    os.makedirs(f"{args.template_dir}/cnos_results", exist_ok=True)
    run_inference(args.template_dir, args.rgb_path, num_max_dets=args.num_max_dets, conf_threshold=args.confg_threshold, stability_score_thresh=args.stability_score_thresh)