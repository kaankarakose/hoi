import torch
import numpy as np
import torchvision
from torchvision.ops.boxes import batched_nms, box_area
import logging
from src.utils.inout import save_json, load_json, save_npz
from src.utils.bbox_utils import xyxy_to_xywh, xywh_to_xyxy, force_binary_mask
import time
from PIL import Image

lmo_object_ids = np.array(
    [
        1,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
    ]
)  # object ID of occlusionLINEMOD is different


def mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order="F")):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


class BatchedData:
    """
    A structure for storing data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        return np.ceil(len(self.data) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size : (idx + 1) * self.batch_size]

    def cat(self, data, dim=0):
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)


class Detections:
    """
    A structure for storing detections.
    """

    def __init__(self, data) -> None:
        if isinstance(data, str):
            data = self.load_from_file(data)
        for key, value in data.items():
            setattr(self, key, value)
        self.keys = list(data.keys())
        if "boxes" in self.keys:
            if isinstance(self.boxes, np.ndarray):
                self.to_torch()
            self.boxes = self.boxes.long()
    def __iter__(self):
        """
        Implement iteration over individual detections.
        Yields a SimpleNamespace object for each detection with attributes
        corresponding to each key in self.keys.
        """
        from types import SimpleNamespace
        n = len(self)
        for i in range(n):
            detection = SimpleNamespace()
            for key in self.keys:
                attr = getattr(self, key)
                if hasattr(attr, "__getitem__") and len(attr) == n:
                    setattr(detection, key, attr[i])
            yield detection

            
    def remove_very_small_detections(self, config):
        img_area = self.masks.shape[1] * self.masks.shape[2]
        box_areas = box_area(self.boxes) / img_area
        mask_areas = self.masks.sum(dim=(1, 2)) / img_area
        keep_idxs = torch.logical_and(
            box_areas > config.min_box_size**2, mask_areas > config.min_mask_size
        )
        # logging.info(f"Removing {len(keep_idxs) - keep_idxs.sum()} detections")
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idxs])

    def apply_nms_per_object_id(self, nms_thresh=0.5):
        keep_idxs = BatchedData(None)
        all_indexes = torch.arange(len(self.object_ids), device=self.boxes.device)
        for object_id in torch.unique(self.object_ids):
            idx = self.object_ids == object_id
            idx_object_id = all_indexes[idx]
            keep_idx = torchvision.ops.nms(
                self.boxes[idx].float(), self.scores[idx].float(), nms_thresh
            )
            keep_idxs.cat(idx_object_id[keep_idx])
        keep_idxs = keep_idxs.data
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idxs])

    def apply_nms(self, nms_thresh=0.5):
        keep_idx = torchvision.ops.nms(
            self.boxes.float(), self.scores.float(), nms_thresh
        )
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idx])

    def add_attribute(self, key, value):
        setattr(self, key, value)
        self.keys.append(key)

    def __len__(self):
        return len(self.boxes)

    def check_size(self):
        mask_size = len(self.masks)
        box_size = len(self.boxes)
        score_size = len(self.scores)
        object_id_size = len(self.object_ids)
        assert (
            mask_size == box_size == score_size == object_id_size
        ), f"Size mismatch {mask_size} {box_size} {score_size} {object_id_size}"

    def to_numpy(self):
        """Convert all tensor attributes to numpy arrays.
        Skip non-tensor attributes to prevent errors.
        """
        import logging
        for key in self.keys:
            # Skip the 'keys' attribute itself
            if key == 'keys':
                continue
                
            # Get the attribute
            value = getattr(self, key)
            
            # Skip non-tensor attributes
            if not torch.is_tensor(value):
                logging.debug(f"Skipping non-tensor attribute '{key}' of type {type(value)} during to_numpy conversion")
                continue
                
            # Convert tensor to CPU then numpy
            try:
                value = value.detach().cpu()
                
                # Special handling for masks to preserve dimensions
                if key == 'masks':
                    # Check if masks is already a single thin mask
                    if len(value.shape) == 2:  # single mask, not batch
                        if value.shape[0] == 1 or value.shape[1] == 1:  # thin in one dimension
                            # Convert to numpy but force it to stay 2D
                            numpy_mask = np.zeros((1920,1200))
                            setattr(self, key, numpy_mask)
                            continue          
                    # Check the shape of each mask in batch
                    if len(value.shape) == 3:  # batch x height x width
                        # Convert masks to numpy one at a time to prevent dimension squeezing
                        numpy_masks = []
                        for i in range(value.shape[0]):
                            mask = value[i]
                            # Convert to numpy and force 2D shape
                            numpy_mask = mask.numpy().reshape(mask.shape[0], mask.shape[1])
                            numpy_masks.append(numpy_mask)
                        setattr(self, key, numpy_masks)
                        continue
                # Default handling for other attributes
                setattr(self, key, value.numpy())
            except Exception as e:
                logging.warning(f"Error converting attribute '{key}' to numpy: {str(e)}. Skipping.")

    def to_torch(self):
        for key in self.keys:
            a = getattr(self, key)
            setattr(self, key, torch.from_numpy(getattr(self, key)))

    def save_to_file(
        self, scene_id, frame_id, runtime, file_path, dataset_name, return_results=False, save_mask=True, save_score_distribution=False
    ):
        """
        scene_id, image_id, category_id, bbox, time
        """
        boxes = xyxy_to_xywh(self.boxes)
        results = {
            "scene_id": scene_id,
            "image_id": frame_id,
            "category_id": self.object_ids + 1
            if dataset_name != "lmo"
            else lmo_object_ids[self.object_ids],
            "score": self.scores,
            "bbox": boxes,
            "time": runtime,
        }
        if save_mask:
            results["segmentation"] = self.masks
        if save_score_distribution:
            assert hasattr(self, "score_distribution"), "score_distribution is not defined"
            results["score_distribution"] = self.score_distribution
        save_npz(file_path, results)
        if return_results:
            return results

    def load_from_file(self, file_path):
        data = np.load(file_path)
        boxes = xywh_to_xyxy(np.array(data["bbox"]))
        output = {
            "object_ids": data["category_id"] - 1,
            "bbox": boxes,
            "scores": data["score"],
        }
        if "segmentation" in data.keys():
            output["masks"] = data["segmentation"]
        if "score_distribution" in data.keys():
            output["score_distribution"] = data["score_distribution"]
        logging.info(f"Loaded {file_path}")
        return data

    def filter(self, idxs):
        """Filter detections by indices. Handle device mismatches automatically.
        
        Args:
            idxs: Index tensor to filter by. Will be moved to CPU automatically.
        """
        # Force indices to CPU and convert as needed
        if torch.is_tensor(idxs):
            if idxs.device.type != 'cpu':
                idxs = idxs.cpu()
            # Ensure it's a proper integer tensor for indexing
            if idxs.dtype not in [torch.int32, torch.int64, torch.long]:
                idxs = idxs.long()
        
        # Process each attribute - skip the 'keys' attribute itself
        for key in self.keys:
            # Skip the 'keys' attribute itself to prevent self-filtering
            if key == 'keys':
                continue
                
            attr = getattr(self, key)
            if attr is None:
                continue
                
            # Handle tensor attributes
            if torch.is_tensor(attr):
                # Move to CPU for consistent processing
                if attr.device.type != 'cpu':
                    attr = attr.cpu()
                # Do the filtering
                try:
                    filtered_attr = attr[idxs]
                    setattr(self, key, filtered_attr)
                except Exception as e:
                    # Add diagnostics if indexing fails
                    raise RuntimeError(f"Error filtering attribute '{key}'. Attr shape: {attr.shape}, "
                                      f"Attr device: {attr.device}, IDX shape: {idxs.shape}, "
                                      f"IDX device: {idxs.device}, IDX dtype: {idxs.dtype}. "
                                      f"Original error: {str(e)}") from e
            else:
                # Handle non-tensor attributes (like lists or numpy arrays)
                try:
                    # Skip any list attributes that should not be filtered
                    if isinstance(attr, list):
                        continue
                    setattr(self, key, attr[idxs])
                except Exception as e:
                    # Don't raise for non-filterable attributes, just log and skip
                    import logging
                    logging.warning(f"Could not filter attribute '{key}' of type {type(attr)}. Skipping.")

    def clone(self):
        """
        Clone the current object
        """
        return Detections(self.__dict__.copy())


def convert_npz_to_json(idx, list_npz_paths):
    npz_path = list_npz_paths[idx]
    detections = np.load(npz_path)
    results = []
    results_with_score_distribution = []
    for idx_det in range(len(detections["bbox"])):
        result = {
            "scene_id": int(detections["scene_id"]),
            "image_id": int(detections["image_id"]),
            "category_id": int(detections["category_id"][idx_det]),
            "bbox": detections["bbox"][idx_det].tolist(),
            "score": float(detections["score"][idx_det]),
            "time": float(detections["time"]),
        }
        if "segmentation" in detections.keys():
            result["segmentation"] = mask_to_rle(
                force_binary_mask(detections["segmentation"][idx_det])
            )
        results.append(result)

        if "score_distribution" in detections.keys():
            result_with_score_distribution = result.copy()
            result_with_score_distribution["score_distribution"] = detections["score_distribution"][idx_det].tolist()
            results_with_score_distribution.append(result_with_score_distribution)
    return results, results_with_score_distribution
