"""
首先统计中心点的平均位置
然后统计中心点到屁股的平均长度
最后统计两肩的平均长度
"""

from os import PathLike
import re
from typing import Tuple
import numpy as np
from .renderopenpose import readkeypointsfile
from typed_args import TypedArgs, add_argument
from dataclasses import dataclass
from pathlib import Path
import yaml
from tqdm import trange
import os


class Body25:
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    MidHip = 8
    RHip = 9
    RKnee = 10
    RAnkle = 11
    LHip = 12
    LKnee = 13
    LAnkle = 14
    REye = 15
    LEye = 16
    REar = 17
    LEar = 18
    LBigToe = 19
    LSmallToe = 20
    LHeel = 21
    RBigToe = 22
    RSmallToe = 23
    RHeel = 24
    Background = 25


class Body23:
    Nose = 0
    RShoulder = 1
    RElbow = 2
    RWrist = 3
    LShoulder = 4
    LElbow = 5
    LWrist = 6
    RHip = 7
    RKnee = 8
    RAnkle = 9
    LHip = 10
    LKnee = 11
    LAnkle = 12
    REye = 13
    LEye = 14
    REar = 15
    LEar = 16
    LBigToe = 17
    LSmallToe = 18
    LHeel = 19
    RBigToe = 20
    RSmallToe = 21
    RHeel = 22
    Background = 23


@dataclass
class Head:
    """
    相对鼻子的距离
    """
    l_eye_dx: float = 0.
    l_eye_dy: float = 0.
    r_eye_dx: float = 0.
    r_eye_dy: float = 0.
    l_ear_dx: float = 0.
    l_ear_dy: float = 0.
    r_ear_dx: float = 0.
    r_ear_dy: float = 0.


@dataclass
class Args(TypedArgs):
    ref: Path = add_argument('-r', '--ref')
    # num_refs: int = add_argument('--num-refs', default=100)
    input: Path = add_argument('-i', '--input')
    output: Path = add_argument('-o', '--output')
    fake_head: bool = add_argument('--fake-head', action='store_true')


def get_refs(input_dir: Path):
    posepts_list = []
    facepts_list = []
    r_handpts_list = []
    l_handpts_list = []

    num_files = len(list(input_dir.iterdir()))

    num_refs = num_files // 4

    for i in range(num_refs):
        key_name = str(input_dir / f'frame{i:06d}')
        # print(key_name)
        posepts = readkeypointsfile(key_name + "_pose")
        facepts = readkeypointsfile(key_name + "_face")
        r_handpts = readkeypointsfile(key_name + "_hand_right")
        l_handpts = readkeypointsfile(key_name + "_hand_left")

        posepts_list.append(posepts)
        facepts_list.append(facepts)
        r_handpts_list.append(r_handpts)
        l_handpts_list.append(l_handpts)

    posepts_np = np.array(posepts_list)
    facepts_np = np.array(facepts_list)
    r_handpts_np = np.array(r_handpts_list)
    l_handpts_np = np.array(l_handpts_list)

    return posepts_np, facepts_np, r_handpts_np, l_handpts_np


def get_neck_to_mid_hip_dis_mean(posepts_np: np.ndarray) -> float:
    """
    [T, 25, 3]
    获取中心点到屁股的距离
    """
    dy_mean = np.mean(
        np.abs(posepts_np[:, Body25.Neck, 1] - posepts_np[:, Body25.MidHip, 1])
    )
    return dy_mean


def get_shoulders_dis_mean(posepts_np: np.ndarray) -> float:
    """
    [T, 25, 3]
    获取两肩距离
    """
    dx_mean = np.mean(
        np.abs(posepts_np[:, Body25.RShoulder, 0] -
               posepts_np[:, Body25.LShoulder, 0])
    )
    return dx_mean


def get_neck_position(posepts_np: np.ndarray) -> Tuple[float, float]:
    neck_x_mean = np.mean(posepts_np[:, Body25.Neck, 0])
    neck_y_mean = np.mean(posepts_np[:, Body25.Neck, 1])
    return neck_x_mean, neck_y_mean


def get_head_nose_relative_mean(posepts_np: np.ndarray) -> Head:

    def _get_dx_dy_mean(part: int) -> Tuple[float, float]:
        dx_mean = np.mean(posepts_np[:, part, 0] -
                          posepts_np[:, Body25.Nose, 0])
        dy_mean = np.mean(posepts_np[:, part, 1] -
                          posepts_np[:, Body25.Nose, 1])
        return dx_mean, dy_mean

    head = Head()
    head.l_eye_dx, head.l_eye_dy = _get_dx_dy_mean(Body25.LEye)
    head.r_eye_dx, head.r_eye_dy = _get_dx_dy_mean(Body25.REye)
    head.l_ear_dx, head.l_ear_dy = _get_dx_dy_mean(Body25.LEar)
    head.r_ear_dx, head.r_ear_dy = _get_dx_dy_mean(Body25.REar)

    return head


def read_posepts_np_from_yml(filename: str) -> np.ndarray:
    data = read_opencv_yml(filename)
    first_key = next(iter(data.keys()))
    return np.array(data[first_key]['data']).reshape(-1, 3)


def make_fake_head(posepts: np.ndarray, head: Head):
    def _update(part: int, dx: float, dy: float):
        posepts[part, 0] = dx + posepts[Body25.Nose, 0]
        posepts[part, 1] = dy + posepts[Body25.Nose, 1]
        posepts[part, 2] = posepts[Body25.Nose, 2]  # 给一个假的score

    _update(Body25.LEye, head.l_eye_dx, head.l_eye_dy)
    _update(Body25.REye, head.r_eye_dx, head.r_eye_dy)
    _update(Body25.LEar, head.l_ear_dx, head.l_ear_dy)
    _update(Body25.REar, head.r_ear_dx, head.r_ear_dy)


def read_opencv_yml(filename: str) -> dict:
    with open(filename, 'r') as f:
        f.readline()
        content = f.read()
        content = content.replace('!!opencv-matrix', '')
        data = yaml.safe_load(content)

    return data


def replace_yml(filename: str, pts: np.ndarray, output_dir: Path):
    data = read_opencv_yml(filename)
    basename = os.path.basename(filename)
    first_key = next(iter(data.keys()))
    data[first_key]['data'] = pts.flatten().tolist()

    output_file = output_dir / basename
    with open(output_file, 'w') as f:
        yaml.safe_dump(data, f)


def main():
    args = Args.from_args()
    posepts_np, facepts_np, r_handpts_np, l_handpts_np = get_refs(
        args.ref)

    posepts_np = posepts_np.reshape(-1, 25, 3)

    neck_x_mean, neck_y_mean = get_neck_position(posepts_np)
    neck_to_mid_hip_dis_mean = get_neck_to_mid_hip_dis_mean(posepts_np)
    shoulders_dis_mean = get_shoulders_dis_mean(posepts_np)

    head = get_head_nose_relative_mean(posepts_np)

    num_input_files = len(list(args.input.iterdir()))
    args.output.mkdir(parents=True, exist_ok=True)

    num_frames = num_input_files // 4
    for i in trange(num_frames):
        key_name = str(args.input / f'frame{i:06d}')
        posepts = read_posepts_np_from_yml(key_name + '_pose.yml')
        facepts = read_posepts_np_from_yml(key_name + "_face.yml")
        r_handpts = read_posepts_np_from_yml(key_name + "_hand_right.yml")
        l_handpts = read_posepts_np_from_yml(key_name + "_hand_left.yml")
        neck_x, neck_y = get_neck_position(
            posepts[None]
        )
        neck_to_mid_hip_dis = get_neck_to_mid_hip_dis_mean(
            posepts[None]
        )
        shoulders_dis = get_shoulders_dis_mean(
            posepts[None]
        )

        dx = neck_x_mean - neck_x
        dy = neck_y_mean - neck_y
        scale_x = shoulders_dis_mean / shoulders_dis
        scale_y = neck_to_mid_hip_dis_mean / neck_to_mid_hip_dis

        def _update(data: np.ndarray):
            data[:, 0] = data[:, 0] * scale_x + dx
            data[:, 1] = data[:, 1] * scale_y + dy

        _update(posepts)
        _update(facepts)
        _update(r_handpts)
        _update(l_handpts)

        if args.fake_head:
            make_fake_head(posepts, head)

        replace_yml(key_name + '_pose.yml', posepts, args.output)
        replace_yml(key_name + '_face.yml', facepts, args.output)
        replace_yml(key_name + '_hand_right.yml', r_handpts, args.output)
        replace_yml(key_name + '_hand_left.yml', l_handpts, args.output)
        # import ipdb
        # ipdb.set_trace()


if __name__ == "__main__":
    main()
