"""Shared model dimensions and data-augmentation settings.

Training, device and output options are configured by train.py arguments.
"""


class Config:
    input_img_shape = (256, 256)
    output_hm_shape = (64, 64, 64)
    bbox_3d_size = 400
    bbox_3d_size_root = 400
    output_root_hm_shape = 64
    bounding_box_scale_factor = 1.5
    trans_factor = 0.15
    scale_factor = 0.25
    rot_factor = 45
    color_factor = 0.2
    mask_img_percent = 0.3


cfg = Config()
