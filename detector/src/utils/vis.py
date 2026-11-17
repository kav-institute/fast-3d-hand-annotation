import numpy as np
from functools import lru_cache
from PIL import Image, ImageDraw


DEFAULT_KP_RGB = (230, 230, 0)
KEYPOINT_SUFFIX_RGB = (
    ('thumb_null', (255, 0, 0)),
    ('thumb3', (255, 51, 51)),
    ('thumb2', (255, 102, 102)),
    ('thumb1', (255, 153, 153)),
    ('thumb0', (255, 204, 204)),
    ('index_null', (0, 255, 0)),
    ('index3', (51, 255, 51)),
    ('index2', (102, 255, 102)),
    ('index1', (153, 255, 153)),
    ('middle_null', (255, 128, 0)),
    ('middle3', (255, 153, 51)),
    ('middle2', (255, 178, 102)),
    ('middle1', (255, 204, 153)),
    ('ring_null', (0, 128, 255)),
    ('ring3', (51, 153, 255)),
    ('ring2', (102, 178, 255)),
    ('ring1', (153, 204, 255)),
    ('pinky_null', (255, 0, 255)),
    ('pinky3', (255, 51, 255)),
    ('pinky2', (255, 102, 255)),
    ('pinky1', (255, 153, 255)),
)


@lru_cache(maxsize=16)
def _get_keypoint_rgb_from_names(joint_names):
    rgb_dict = {}
    for joint_name in joint_names:
        rgb = DEFAULT_KP_RGB
        for suffix, color in KEYPOINT_SUFFIX_RGB:
            if joint_name.endswith(suffix):
                rgb = color
                break
        rgb_dict[joint_name] = rgb
    return rgb_dict


def get_keypoint_rgb(skeleton):
    joint_names = tuple(joint['name'] for joint in skeleton)
    return _get_keypoint_rgb_from_names(joint_names)


def vis_keypoints_np(img, kps, score, skeleton, score_thr=0.4, line_width=3, circle_rad=3):
    rgb_dict = get_keypoint_rgb(skeleton)
    valid = np.asarray(score) > score_thr
    joint_names = [joint['name'] for joint in skeleton]
    parent_ids = [joint['parent_id'] for joint in skeleton]

    _img = Image.fromarray(img)
    draw = ImageDraw.Draw(_img)

    for i, pid in enumerate(parent_ids):
        if pid == -1 or not valid[i] or not valid[pid]:
            continue
        parent_joint_name = joint_names[pid]
        draw.line(
            [(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])],
            fill=rgb_dict[parent_joint_name],
            width=line_width,
        )

    for i, joint_name in enumerate(joint_names):
        if not valid[i]:
            continue
        x, y = kps[i][0], kps[i][1]
        draw.ellipse(
            (x - circle_rad, y - circle_rad, x + circle_rad, y + circle_rad),
            fill=rgb_dict[joint_name],
        )
    return _img
