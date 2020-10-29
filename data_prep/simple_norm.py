"""
首先统计中心点的平均位置
然后统计中心点到屁股的平均长度
最后统计两肩的平均长度
"""

import numpy as np
from .renderopenpose import readkeypointsfile
from typed_args import TypedArgs, add_argument
from dataclasses import dataclass


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
class Args(TypedArgs):
    pass


def main():
    pass


if __name__ == "__main__":
    main()
