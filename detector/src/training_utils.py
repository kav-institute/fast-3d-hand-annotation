"""Collation helpers for mixed hand-pose training datasets."""
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

def _to_tensor(x):
    if isinstance(x, np.ndarray):
        # use float32 by default; change to x.dtype if you prefer exact
        return torch.from_numpy(x).float()
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], np.ndarray):
        # common case: list of ndarrays
        try:
            return torch.from_numpy(np.stack(x)).float()
        except Exception:
            return [torch.from_numpy(v).float() for v in x]
    return x


def _map_tensors(d):
    # map recursively only for simple dicts: values are tensors/ndarray/scalars
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _map_tensors(v)
        elif v is None:
            out[k] = None
        else:
            out[k] = _to_tensor(v)
    return out


def _safe_stack(values):
    """Stack only if all are tensors with identical shapes/dtypes."""
    vals = [_to_tensor(v) for v in values]
    if len(vals) == 0:
        return vals
    # None present => cannot stack
    if any(v is None for v in vals):
        return values
    # all tensors & same shape
    if all(torch.is_tensor(v) for v in vals):
        shapes = [tuple(v.shape) for v in vals]
        dtypes = [v.dtype for v in vals]
        if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
            return torch.stack(vals, dim=0)
    return values


def collate_meta_recursive(meta_list, stack_paths=None):
    """
    Recursively collate a list of meta dicts.

    - For each (possibly nested) key, produce either:
        * a list of values (one per sample), or
        * a stacked tensor (if path is whitelisted in stack_paths and stack is possible).
    - Keys missing in some samples will get value None in that position.

    Args:
        meta_list: list[dict]
        stack_paths: iterable of tuple paths to attempt stacking, e.g.
                     {("inv_trans",), ("intrinsics","focal"), ("intrinsics","princpt")}
                     If None -> never stack; everything becomes lists.
    """
    if stack_paths is None:
        stack_paths = set()
    else:
        stack_paths = set(tuple(p) for p in stack_paths)

    def _recurse(objs, path=()):
        # If every element at this level is a dict -> recurse per key
        if all(isinstance(o, dict) for o in objs if o is not None):
            # union of keys across dicts
            keys = set()
            for o in objs:
                if isinstance(o, dict):
                    keys.update(o.keys())
            out = {}
            for k in keys:
                child_vals = [(o.get(k) if isinstance(o, dict) else None) for o in objs]
                out[k] = _recurse(child_vals, path + (k,))
            return out
        else:
            # Leaf level: try stack if path is whitelisted, else return list
            if path in stack_paths:
                return _safe_stack(objs)
            return objs  # list across batch

    return _recurse(list(meta_list), ())


def mixed_triplet_collate(batch):
    """
    batch: list of (inputs, targets, meta_info)
    - inputs/targets -> default_collate after numpy->tensor conversion
    - meta_info      -> keep as a list of dicts (converted to tensors where possible),
                        but DO NOT stack (shapes may vary)
    """

    stack_paths = {
        ("joint_valid",),
        ("root_valid",),
        ("hand_presence_valid",),
        ("inv_trans",),
        ("root_depth_mm_rl", "right"),
        ("root_depth_mm_rl", "left"),
        ("intrinsics", "focal"),
        ("intrinsics", "princpt"),
        ("3d_keypoints",),
        ("2d_keypoints",),
        ("dataset_id",),
    }

    inputs_list, targets_list, meta_list = zip(*batch)

    inputs_list  = [_map_tensors(x) for x in inputs_list]
    targets_list = [_map_tensors(x) for x in targets_list]
    meta_list    = [_map_tensors(x) for x in meta_list]  # convert values, keep list

    batched_inputs  = default_collate(inputs_list)
    batched_targets = default_collate(targets_list)
    batched_metas = collate_meta_recursive(meta_list, stack_paths)

    return batched_inputs, batched_targets, batched_metas

