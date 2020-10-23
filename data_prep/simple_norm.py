"""
首先统计中心点的平均位置
然后统计中心点到屁股的平均长度
最后统计两肩的平均长度
"""

import numpy as np
from .renderopenpose import readkeypointsfile
from typed_args import TypedArgs, add_argument
from dataclasses import dataclass


@dataclass
class Args(TypedArgs):
    pass


def main():
    pass


if __name__ == "__main__":
    main()
