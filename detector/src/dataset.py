import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
import os
import glob
import json
from pycocotools.coco import COCO

from typing import List, Optional, Dict, Any, Tuple



from utils.preprocessing import (
    load_img_rgb,
    load_skeleton,
    process_bbox,
    augmentation,
    transform_input_to_output_space,
)
from config import cfg

'''
skeleton_thab.txt
known dataset_root
'''

def world2cam(world_coord, R, T):
    """
    Transform world coordinates to camera coordinates.
    """
    world_coord_2 = np.concatenate((world_coord, np.ones((len(world_coord), 1))), axis=1)
    nR = np.eye(4)
    nR[:3, :3] = R
    nT = np.eye(4)
    nT[:3, 3:] = T
    cam_coord = (nT @ nR @ world_coord_2.T).T
    return cam_coord[:, :3]

class THABDataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode, prefix, dataset_root, world_scale=10.0):
        assert mode in ["train", "test", "val"], "Mode must be train, test, or val"
        self.mode = mode
        self.img_path = dataset_root
        self.annot_path = dataset_root
        self.joint_num = 21  # single hand keypoints
        self.root_joint_idx = {"right": 0, "left": 21}
        self.joint_type = {
            "right": np.arange(0, self.joint_num),
            "left": np.arange(self.joint_num, self.joint_num * 2),
        }
        self.prefix = prefix
        self.transform = transform
        self.skeleton = load_skeleton(osp.join(self.annot_path, "skeleton_thab.txt"), self.joint_num * 2)
        self.datalist = []

        print("Loading annotations from " + osp.join(self.annot_path))
        db = COCO(osp.join(self.annot_path, self.prefix + self.mode + ".json"))
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann["image_id"]
            img_info = db.loadImgs(image_id)[-1]
            cam_id = str(img_info["camera"]).removeprefix("camera_").removeprefix("camera")
            # Load camera parameters
            cam_file = glob.glob(osp.join(self.annot_path, img_info["file_name"].split("/")[0], "camera_parameter*.json"))
            if len(cam_file) != 1:
                raise ValueError("Expected exactly one camera_parameter*.json file in the training sequence")
            with open(cam_file[0], "r") as fin:
                cam_params = json.load(fin)
            matching = [c for c in cam_params["cameras"] if str(c["camera"]) == cam_id]
            if len(matching) != 1:
                raise ValueError(f"Expected calibration for camera {cam_id} in {cam_file[0]}")
            cam_param = matching[0]
            princpt = np.array([cam_param["camera_matrix"][0][2], cam_param["camera_matrix"][1][2]], dtype=np.float32)
            focal = np.array([cam_param["camera_matrix"][0][0], cam_param["camera_matrix"][1][1]], dtype=np.float32)

            img_kpts = np.array(ann["keypoints"])
            bx_min, by_min = np.min(img_kpts, axis=0)
            bx_max, by_max = np.max(img_kpts, axis=0)
            bbox = [bx_min, by_min, bx_max - bx_min, by_max - by_min]
            hand_presence = ann["hand_type"]

            # Transform 3D keypoints from world to camera coordinate system.
            cam_kpts = world2cam(np.array(ann["keypoints_3d"]) * world_scale,
                                 cam_param["rotation_matrix"],
                                 np.array(cam_param["t_param"]) * world_scale)
            # Adjust keypoints depending on which hand is annotated.
            if hand_presence == "right":
                if len(img_kpts) == self.joint_num:
                    img_kpts = np.vstack((img_kpts, np.zeros_like(img_kpts)))
                elif len(img_kpts) == 2*self.joint_num:
                    img_kpts[21:,:] = 0.0
                else:
                    raise ValueError("Expected 21 or 42 2D joints")
                if len(cam_kpts) == self.joint_num:
                    cam_kpts = np.vstack((cam_kpts, np.zeros_like(cam_kpts)))
                elif len(cam_kpts) == 2*self.joint_num:
                    cam_kpts[21:,:] = 0.0
                else:
                    raise ValueError("Expected 21 or 42 3D joints")
                valid = np.concatenate((np.ones(self.joint_num), np.zeros(self.joint_num)))
            elif hand_presence == "left":
                if len(img_kpts) == self.joint_num:
                    img_kpts = np.vstack((np.zeros_like(img_kpts), img_kpts))
                elif len(img_kpts) == 2*self.joint_num:
                    img_kpts[:21,:] = 0.0
                else:
                    raise ValueError("Expected 21 or 42 2D joints")
                if len(cam_kpts) == self.joint_num:
                    cam_kpts = np.vstack((np.zeros_like(cam_kpts), cam_kpts))
                elif len(cam_kpts) == 2*self.joint_num:
                    cam_kpts[:21,:] = 0.0
                else:
                    raise ValueError("Expected 21 or 42 3D joints")
                valid = np.concatenate((np.zeros(self.joint_num), np.ones(self.joint_num)))
            elif hand_presence == "interacting":
                valid = np.ones(self.joint_num * 2)
            else:
                raise NotImplementedError("Unsupported hand type: " + hand_presence)

            img_width, img_height = cfg.input_img_shape
            bbox = process_bbox(bbox, cfg.bounding_box_scale_factor, (img_width, img_height))
            hand_presence_valid = np.array([1])
            seq_name = img_info["file_name"].split("/")[0]
            frame_idx = img_info["id"]
            capture_id = 0
            abs_depth = {
                "right": cam_kpts[self.root_joint_idx["right"], 2],
                "left": cam_kpts[self.root_joint_idx["left"], 2],
            }
            cam_param_info = {"focal": focal, "princpt": princpt}
            joint = {"cam_coord": cam_kpts, "img_coord": img_kpts, "valid": valid}
            img_file = osp.join(self.annot_path, img_info["file_name"])
            data = {
                "img_path": img_file,
                "seq_name": seq_name,
                "cam_param": cam_param_info,
                "bbox": bbox,
                "joint": joint,
                "hand_presence": hand_presence,
                "hand_presence_valid": hand_presence_valid,
                "abs_depth": abs_depth,
                "file_name": img_info["file_name"],
                "capture": capture_id,
                "cam": img_info["camera"][-1],
                "frame": frame_idx,
            }
            self.datalist.append(data)

    def handtype_str2array(self, hand_presence):
        if hand_presence == "right":
            return np.array([1, 0], dtype=np.float32)
        elif hand_presence == "left":
            return np.array([0, 1], dtype=np.float32)
        elif hand_presence == "interacting":
            return np.array([1, 1], dtype=np.float32)
        else:
            raise ValueError("Unsupported hand type: " + hand_presence)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path = data["img_path"]
        bbox = data["bbox"]
        joint = data["joint"]
        hand_presence = data["hand_presence"]
        hand_presence_valid = data["hand_presence_valid"]

        joint_cam = joint["cam_coord"].copy()
        joint_img = joint["img_coord"].copy()
        joint_valid = joint["valid"].copy()
        hand_presence = self.handtype_str2array(hand_presence)
        # Concatenate 2D image coordinates with the depth from camera keypoints.
        joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None]), axis=1)

        # Load the image and perform augmentation.
        img = load_img_rgb(img_path)
        img, joint_coord, joint_valid, hand_presence, inv_trans = augmentation(
            img, bbox, joint_coord, joint_valid, hand_presence, self.mode, self.joint_type, cfg
        )
        # Compute the relative depth between the two hands.
        rel_root_depth = np.array(
            [joint_coord[self.root_joint_idx["left"], 2] - joint_coord[self.root_joint_idx["right"], 2]],
            dtype=np.float32,
        ).reshape(1)


        root_valid = (
            np.array(
                [joint_valid[self.root_joint_idx["right"]] * joint_valid[self.root_joint_idx["left"]]],
                dtype=np.float32,
            ).reshape(1)
            if hand_presence[0] * hand_presence[1] == 1
            else np.zeros((1), dtype=np.float32)
        )
        # Transform the joint coordinates and validity to output space.
        joint_coord_, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(
            joint_coord, joint_valid, rel_root_depth, root_valid, self.root_joint_idx, self.joint_type
        )

        root_depth_mm_rl = {"right": np.array([joint_coord[self.root_joint_idx['right'],2]]), "left":  np.array([joint_coord[self.root_joint_idx['left'],2]])}

        joint_coord[self.joint_type['right'],2] = joint_coord[self.joint_type['right'],2] - joint_coord[self.root_joint_idx['right'],2]
        joint_coord[self.joint_type['left'],2] = joint_coord[self.joint_type['left'],2] - joint_coord[self.root_joint_idx['left'],2]

        img = self.transform(img.astype(np.float32)) / 255.0

        inputs = {"img": img}
        targets = {
            "joint_coord": joint_coord_,
            "joint_coord_untransformed": joint_coord,
            "rel_root_depth": rel_root_depth,
            "hand_presence": hand_presence,
        }
        meta_info = {
            "joint_valid": joint_valid,
            "root_valid": root_valid,
            "hand_presence_valid": hand_presence_valid,
            "inv_trans": inv_trans,


            "root_depth_mm_rl": root_depth_mm_rl,
            "intrinsics": data["cam_param"],
            "3d_keypoints": joint_cam,
            "2d_keypoints": joint_img,
            "dataset_id": np.array(1)
        }

        return inputs, targets, meta_info


def _read_json(p: str) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def _safe_get(d: dict, keys: List[str], default=None):
    """Try multiple candidate keys in a dict."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def _parse_intrinsics(cam: dict) -> np.ndarray:
    """
    Return 3x3 K. Accepts:
    - 'camera_matrix' / 'K', or
    - constructs from 'focal' (fx, fy) and 'princpt' (cx, cy)
    """
    K = _safe_get(cam, ["camera_matrix", "K"])
    if K is not None:
        K = np.array(K, dtype=np.float32)
        if K.size == 9:
            return K.reshape(3, 3)
        raise ValueError(f"Unexpected intrinsics shape {K.shape}")
    # fallback to focal/princpt
    focal = _safe_get(cam, ["focal"])
    princpt = _safe_get(cam, ["princpt"])
    if focal is not None and princpt is not None:
        fx, fy = float(focal[0]), float(focal[1])
        cx, cy = float(princpt[0]), float(princpt[1])
        return np.array([[fx, 0.0, cx],
                         [0.0, fy, cy],
                         [0.0, 0.0, 1.0]], dtype=np.float32)
    raise KeyError("No intrinsics found: expected 'camera_matrix'/'K' or 'focal'+'princpt'.")


def _parse_rotation(cam: dict) -> np.ndarray:
    """
    Return 3x3 R. Accepts variants:
    - 'rotation_matrix', 'R'
    """
    R = _safe_get(cam, ["rotation_matrix", "R"])
    if R is None:
        # Some files store rotation as a quaternion or missing; default to I
        R = np.eye(3, dtype=np.float32)
    R = np.array(R, dtype=np.float32)
    if R.size == 9:
        return R.reshape(3, 3)
    if R.shape == (3, 3):
        return R
    raise ValueError(f"Unexpected rotation shape {R.shape}")


def _parse_translation(cam: dict) -> np.ndarray:
    """
    Return 3x1 (or (3,)) translation in the same metric as your world coords.
    Accepts variants:
    - 't', 't_param', 'T', 'translation'
    """
    Tt = _safe_get(cam, ["t", "t_param", "T", "translation"], default=[0, 0, 0])
    Tt = np.array(Tt, dtype=np.float32).reshape(3)
    return Tt


def _parse_distortion(cam: dict) -> np.ndarray:
    """
    Return 4x1 (or (4,)) distortion 
    Accepts variants:
    - 'D', 'd'
    """
    Dd = _safe_get(cam, ["D", 'd'], default=[0, 0, 0, 0])
    Dd = np.array(Dd, dtype=np.float32).reshape(4)
    return Dd


def _read_json_array(path):
    with open(path, "r") as f:
        return json.load(f)

def _to_21x3(arr, joint_num=21):
    """Make sure we return exactly (21,3). Handles:
       - extra points (e.g., 24/25 with last==first): drop extras
       - fewer points: pad with zeros
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(f"Bad keypoint shape {a.shape}; expected (N,>=3)")
    # keep only xyz
    a = a[:, :3]

    # slice or pad to joint_num
    if a.shape[0] >= joint_num:
        b = 1
        a = a[b:joint_num+b]
        valid = np.ones(joint_num, dtype=np.float32)
    else:
        valid = np.zeros(joint_num, dtype=np.float32)
        valid[:a.shape[0]] = 1.0
        pad = np.zeros((joint_num - a.shape[0], 3), dtype=np.float32)
        a = np.vstack([a, pad])

    return a, valid

def project_kpts_3d_to_2d(
    kpts_42x3: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    coords: str = "world",
) -> np.ndarray:
    """
    Project 3D keypoints to image pixels using intrinsics K and extrinsics (R, T).

    Args:
        kpts_42x3 : (42, 3) 3D points. If coords='world', they are in world space.
                    If coords='camera', they are already in the camera frame.
        K         : (3, 3) camera intrinsics.
        R         : (3, 3) rotation (world->camera). Ignored if coords='camera'.
        T         : (3,)   translation (world->camera). Ignored if coords='camera'.
        coords    : 'world' or 'camera'.

    Returns:
        kpts_42x2 : (42, 2) pixel coordinates (u, v). Points with Z<=0 get NaNs.
    """
    pts = np.asarray(kpts_42x3, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32).reshape(3)

    if coords == "world":
        # camera coordinates: Xc = R * Xw + T
        pts_cam = (R @ pts.T).T + T[None, :]
    elif coords == "camera":
        pts_cam = pts
    else:
        raise ValueError("coords must be 'world' or 'camera'")

    Z = pts_cam[:, 2:3]  # (N,1)
    # Avoid divide-by-zero; mark invalid where Z<=0
    invalid = Z[:, 0] <= 0

    # Homogeneous projection: [u', v', w']^T = K * [Xc, Yc, Zc]^T
    proj = (pts_cam @ K.T)  # (N,3)
    w = proj[:, 2:3]
    uv = proj[:, :2] / np.maximum(w, 1e-8)

    # Set invalid to NaN (optional behavior)
    uv[invalid] = np.nan
    return uv.astype(np.float32), pts_cam

def load_keypoints_right_left(right_json_path, left_json_path, joint_num=21):
    """
    Returns:
        kpts_42x3: np.ndarray (42,3) stacked as [right(21), left(21)]
        valid_42:  np.ndarray (42,) validity mask (1 if present, else 0)
    Missing files -> zeros and valid=0 for that hand.
    """
    # right
    if right_json_path and os.path.isfile(right_json_path):
        right_raw = _read_json_array(right_json_path)
        right_21x3, right_valid = _to_21x3(right_raw, joint_num)
    else:
        right_21x3 = np.zeros((joint_num, 3), dtype=np.float32)
        right_valid = np.zeros(joint_num, dtype=np.float32)

    # left
    if left_json_path and os.path.isfile(left_json_path):
        left_raw = _read_json_array(left_json_path)
        left_21x3, left_valid = _to_21x3(left_raw, joint_num)
    else:
        left_21x3 = np.zeros((joint_num, 3), dtype=np.float32)
        left_valid = np.zeros(joint_num, dtype=np.float32)

    kpts_42x3 = np.vstack([right_21x3, left_21x3])          # (42,3), right first
    valid_42  = np.concatenate([right_valid, left_valid])    # (42,)
    return kpts_42x3, valid_42

def K_to_cam_param_info(K) -> dict:
    """
    Convert a 3x3 intrinsic matrix K to
    cam_param_info = {"focal": [fx, fy], "princpt": [cx, cy]}.
    Accepts numpy arrays or torch tensors.
    """
    try:
        import torch
        if isinstance(K, torch.Tensor):
            K = K.detach().cpu().numpy()
    except Exception:
        pass

    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    focal   = np.array([fx, fy], dtype=np.float32)
    princpt = np.array([cx, cy], dtype=np.float32)
    return {"focal": focal, "princpt": princpt}

def _flatten_D(D):
    D = np.asarray(D, dtype=np.float32).reshape(-1)
    # Accept 4, 5, or 8+ (Brown model); ignore extras if present
    if D.size >= 4:
        return D[:4].copy()
    raise ValueError(f"Unexpected distortion length {D.size}")

def undistort_fisheye(img_hwc_uint8, K, D, balance=0.0):
    H, W = img_hwc_uint8.shape[:2]
    K = np.asarray(K, np.float32).reshape(3,3)
    D = _flatten_D(D).astype(np.float32)[:4]  # k1,k2,k3,k4 fisheye

    R = np.eye(3, dtype=np.float32)
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (W, H), R, balance=balance, fov_scale=1.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, K_new, (W, H), m1type=cv2.CV_32FC1)
    img_u = cv2.remap(img_hwc_uint8, map1, map2, interpolation=cv2.INTER_LINEAR)
    return img_u, K_new.astype(np.float32), (0,0,W,H)


class ReInterHandDataset(torch.utils.data.Dataset):
    """
    ReInterHand Dataset

    Args:
        root: Re:InterHand download root containing capture directories
        capture_id: e.g. '000001' or a folder name under root
        rig: 'Mugsy' or 'Ego'
        envmap_mode: 'envmap_per_frame' or 'envmap_per_segment'
        cam_names: list of camera names (Mugsy). If None -> all cams from cam_params.json. For Ego, ignored.
        with_forearm: whether to use the *_with_forearm folders in orig_fits
        frame_list_file: 'frame_list.txt' or 'frame_list_orig.txt'
        transform: torchvision transform for images
    """

    def __init__(
        self,
        root: str,
        capture_id: str,
        rig: str = "Mugsy",  # or "Ego"
        envmap_mode: str = "envmap_per_segment",  # or "envmap_per_frame"
        cam_names: Optional[List[str]] = None,  # for Mugsy
        with_forearm: bool = False,
        frame_list_file: str = "frame_list.txt",
        transform: Optional[Any] = None,
        mode: str = "train",
        skeleton_path: Optional[str] = None,
    ):
        super().__init__()
        assert rig in ["Mugsy", "Ego"]
        assert envmap_mode in ["envmap_per_frame", "envmap_per_segment"]
        assert frame_list_file in ["frame_list.txt", "frame_list_orig.txt"]

        self.root = root
        self.cap = capture_id
        self.rig = rig
        self.envmap_mode = envmap_mode
        self.cam_names = cam_names
        self.with_forearm = with_forearm
        self.transform = transform
        self.root_joint_idx = {"right": 0, "left": 21}
        self.mode = mode
        self.joint_num = 21
        self.joint_type = {
            "right": np.arange(0, self.joint_num),
            "left": np.arange(self.joint_num, self.joint_num * 2),
        }

        self.cap_dir = osp.join(root, capture_id)

        # 1) read frame list
        list_path = osp.join(self.cap_dir, frame_list_file)
        if not osp.isfile(list_path):
            raise FileNotFoundError(f"Frame list not found: {list_path}")
        with open(list_path, "r") as f:
            frames = [line.strip().split(" ")[1] for line in f if line.strip()]
        # Normalize to ints when possible:
        def _to_int(s):
            try:
                return int(s)
            except Exception:
                return int(osp.splitext(osp.basename(s))[0])
        self.frames: List[int] = [_to_int(s) for s in frames]

        # 2) camera config
        if self.rig == "Mugsy":
            cam_json = osp.join(self.cap_dir, "Mugsy_cameras", "cam_params.json")
            if not osp.isfile(cam_json):
                raise FileNotFoundError(f"Missing Mugsy cam_params.json at {cam_json}")
            # cam_params.json is a flat dict: { "<cam_id>": {focal, princpt, R, t}, ... }
            self.mugsy_cam: Dict[str, dict] = _read_json(cam_json)

            available_cams = sorted(list(self.mugsy_cam.keys()))
            if self.cam_names is None:
                self.cam_names = available_cams
            else:
                missing = [c for c in self.cam_names if c not in available_cams]
                if missing:
                    print(f"[ReInterHandDataset] WARNING: requested cams not in cam_params.json: {missing}")

        mugsy_imgs_path = osp.join(self.cap_dir, "Mugsy_cameras", envmap_mode, "images")
        # 3) build flat index of samples (frame, cam)
        self.index: List[Dict[str, Any]] = []
        if self.rig == "Mugsy":
            for fr in self.frames:
                for cam in os.listdir(mugsy_imgs_path):#self.cam_names:
                    self.index.append({"frame": fr, "cam": cam})
        else:  # Ego (no named cams)
            for fr in self.frames:
                self.index.append({"frame": fr, "cam": None})

        if skeleton_path is None:
            skeleton_path = osp.join(root, "skeleton_thab.txt")
        self.skeleton = load_skeleton(skeleton_path, self.joint_num * 2)
        
        self.reinterhand_joint_names_to_mano = np.arange(42)

        # small helpers for paths
        self.paths = {
            "mugsy_imgs": mugsy_imgs_path,
            "ego_imgs": osp.join(self.cap_dir, "Ego_cameras", envmap_mode, "images"),
            "orig_right": osp.join(self.cap_dir, "orig_fits", "right"),
            "orig_left": osp.join(self.cap_dir, "orig_fits", "left"),
        }

    def __len__(self) -> int:
        return len(self.index)

    # ---------- camera reading ----------
    def _read_mugsy_cam(self, cam_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cam_name not in self.mugsy_cam:
            raise KeyError(f"Camera {cam_name} not found in cam_params.json")
        cam = self.mugsy_cam[cam_name]
        K = _parse_intrinsics(cam)
        R = _parse_rotation(cam)
        T = _parse_translation(cam)
        return K, R, T
    
    def _read_ego_cam(self, frame: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if frame not in self.frames:
            raise KeyError(f"Camera {frame}.json not found in cam_params")
        cam_json = osp.join(self.cap_dir, "Ego_cameras/envmap_per_segment/cam_params", f"{frame:06d}.json")
        if not osp.isfile(cam_json):
            raise FileNotFoundError(f"Missing Ego cam_params.json at {cam_json}")
        # frame.json is a flat dict
        cam: dict = _read_json(cam_json)
        K = _parse_intrinsics(cam)
        R = _parse_rotation(cam)
        T = _parse_translation(cam)
        D = _parse_distortion(cam)
        return K, R, T, D

    # ---------- optional fits/keypoints ----------
    def _orig_kpt_path(self, hand: str, with_forearm: bool, frame: int) -> str:
        base = self.paths["orig_right"] if hand == "right" else self.paths["orig_left"]
        sub = "Keypoints_with_forearm" if with_forearm else "Keypoints"
        return osp.join(base, sub, f"keypoint-{frame:06d}.json")


    # ---------- image paths ----------
    def _img_path(self, frame: int, cam: Optional[str]) -> str:
        if self.rig == "Mugsy":
            return osp.join(self.paths["mugsy_imgs"], cam, f"{frame:06d}.png")
        else:
            return osp.join(self.paths["ego_imgs"], f"{frame:06d}.png")


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.index[idx]
        frame = sample["frame"]
        cam = sample["cam"]

        # Image
        img_p = self._img_path(frame, cam)
        if not osp.isfile(img_p):
            raise FileNotFoundError(f"Image not found: {img_p}")
        img = load_img_rgb(img_p)

        # Camera params
        if self.rig == "Mugsy":
            K, R, T = self._read_mugsy_cam(cam)
        else:
            K_d, R, T, D = self._read_ego_cam(frame)
            img_d = img.copy()
            img, K, _ssds = undistort_fisheye(img_d, K_d, D)
        keypoint_paths = [self._orig_kpt_path(hand, self.with_forearm, frame)
                          for hand in ("right", "left")]
        keypoint_paths = [path if osp.isfile(path) else None for path in keypoint_paths]
        kpts_42x3_reinterhand, valid_42_reinterhand = load_keypoints_right_left(*keypoint_paths)

        kpts_42x3 = kpts_42x3_reinterhand[self.reinterhand_joint_names_to_mano, :]
        valid_42 = valid_42_reinterhand[self.reinterhand_joint_names_to_mano]

        img_height, img_width, img_c = img.shape
        kpts_42x2, kpts_42x3_cam = project_kpts_3d_to_2d(kpts_42x3, K, R, T)
        bx_min, by_min = np.min(kpts_42x2, axis=0)
        bx_max, by_max = np.max(kpts_42x2, axis=0)
        bbox_xywh = [bx_min, by_min, bx_max - bx_min, by_max - by_min]
        bbox = process_bbox(bbox_xywh, cfg.bounding_box_scale_factor, (img_width, img_height))

        joint_coord = np.concatenate((kpts_42x2, kpts_42x3_cam[:, 2, None]), axis=1)
        joint_valid = valid_42
        hand_presence = np.array([1, 1], dtype=np.float32)

        img, joint_coord, joint_valid, hand_presence, inv_trans = augmentation(
            img, bbox, joint_coord, joint_valid, hand_presence, self.mode, self.joint_type, cfg, extra_rotation=180
        )
        # Compute the relative depth between the two hands.
        rel_root_depth = np.array(
            [joint_coord[self.root_joint_idx["left"], 2] - joint_coord[self.root_joint_idx["right"], 2]],
            dtype=np.float32,
        ).reshape(1)


        root_valid = (
            np.array(
                [joint_valid[self.root_joint_idx["right"]] * joint_valid[self.root_joint_idx["left"]]],
                dtype=np.float32,
            ).reshape(1)
            if hand_presence[0] * hand_presence[1] == 1
            else np.zeros((1), dtype=np.float32)
        )
        # Transform the joint coordinates and validity to output space.
        joint_coord_, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(
            joint_coord, joint_valid, rel_root_depth, root_valid, self.root_joint_idx, self.joint_type
        )

        root_depth_mm_rl = {"right": np.array([joint_coord[self.root_joint_idx['right'],2]]), "left":  np.array([joint_coord[self.root_joint_idx['left'],2]])}

        joint_coord[self.joint_type['right'],2] = joint_coord[self.joint_type['right'],2] - joint_coord[self.root_joint_idx['right'],2]
        joint_coord[self.joint_type['left'],2] = joint_coord[self.joint_type['left'],2] - joint_coord[self.root_joint_idx['left'],2]

        img = self.transform(img.astype(np.float32)) / 255.0

        inputs = {"img": img}
        targets = {
            "joint_coord": joint_coord_,
            "joint_coord_untransformed": joint_coord,
            "rel_root_depth": rel_root_depth,
            "hand_presence": hand_presence,
        }
        meta_info = {
            "joint_valid": joint_valid,
            "root_valid": root_valid,
            "hand_presence_valid": valid_42,
            "inv_trans": inv_trans,


            "root_depth_mm_rl": root_depth_mm_rl,
            "intrinsics": K_to_cam_param_info(K),
            "3d_keypoints": kpts_42x3_cam,
            "2d_keypoints": kpts_42x2,
            "dataset_id": np.array(0) if self.rig == "Mugsy" else np.array(2)
        }

        return inputs, targets, meta_info
