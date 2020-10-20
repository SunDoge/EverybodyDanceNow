from pathlib import Path
from typing import Any
import torch
import json

from torch.tensor import Tensor
import copy
from typed_args import TypedArgs, add_argument
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np


@dataclass
class Args(TypedArgs):
    input_file: str = add_argument(help='输入的pt文件')
    output_dir: Path = add_argument(help='输出的文件夹，一般命名为keypoints')

# 每个视频51个点， 0 ~ 9， 12， 手20个点


_JSON_TEMPLATE = {
    'version': 1.2,
    'people': [{
        'pose_keypoints_2d': [0.0 for _ in range(25 * 3)],
        'face_keypoints_2d': [],
        'hand_left_keypoints_2d': [0.0 for _ in range(21 * 3)],
        'hand_right_keypoints_2d': [0.0 for _ in range(21 * 3)],
    }]
}

_MAP_BODY_INDEX = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    12,
]

_WIDTH = 1920
_HEIGHT = 1080

_LEFT_HAND_FIRST_POINT = 7
_RIGHT_HAND_FIRST_POINT = 4


def pt_to_json(pt_path: str):
    # [T, V, C], keypoints V=51, C=2
    x = torch.load(pt_path, map_location='cpu')
    # x = (x + 1) / 2
    x = x + 0.5
    # x[..., 0] *= _HEIGHT
    # x[..., 1] *= _HEIGHT
    x[..., 0] *= _HEIGHT
    x[..., 1] *= _HEIGHT

    x = x.numpy().astype(float)

    return [frame_to_json(frame) for frame in x]


def frame_to_json(frame: Tensor):
    """
    [V, C]
    """
    tpl: Any = copy.deepcopy(_JSON_TEMPLATE)

    body = frame[:11]
    left_hand = frame[11:31]
    right_hand = frame[31:51]

    pose_keypoints_2d = tpl['people'][0]['pose_keypoints_2d']
    hand_left_keypoints_2d = tpl['people'][0]['hand_left_keypoints_2d']
    hand_right_keypoints_2d = tpl['people'][0]['hand_right_keypoints_2d']

    for i, json_i in enumerate(_MAP_BODY_INDEX):
        x, y = body[i].tolist()
        c = 0.9
        pose_keypoints_2d[json_i * 3] = x
        pose_keypoints_2d[json_i * 3 + 1] = y
        pose_keypoints_2d[json_i * 3 + 2] = c

    # 设置第一个点的座标
    hand_left_keypoints_2d[0] = body[_LEFT_HAND_FIRST_POINT, 0]
    hand_left_keypoints_2d[1] = body[_LEFT_HAND_FIRST_POINT, 1].item()
    hand_left_keypoints_2d[2] = 0.9

    hand_right_keypoints_2d[0] = body[_RIGHT_HAND_FIRST_POINT, 0]
    hand_right_keypoints_2d[1] = body[_RIGHT_HAND_FIRST_POINT, 1]
    hand_right_keypoints_2d[2] = 0.9

    for i in range(20):
        json_i = i + 1
        hand_left_keypoints_2d[json_i * 3] = left_hand[i, 0]
        hand_left_keypoints_2d[json_i * 3 + 1] = left_hand[i, 1]
        hand_left_keypoints_2d[json_i * 3 + 2] = 0.9

        hand_right_keypoints_2d[json_i * 3] = right_hand[i, 0]
        hand_right_keypoints_2d[json_i * 3 + 1] = right_hand[i, 1]
        hand_right_keypoints_2d[json_i * 3 + 2] = 0.9

    print(tpl)
    return tpl


def main():
    args = Args.from_args()

    data = pt_to_json(args.input_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, x in enumerate(tqdm(data)):
        with open(args.output_dir / f'frame{i:06d}_keypoints.json', 'w') as f:
            json.dump(x, f)


if __name__ == "__main__":
    main()
