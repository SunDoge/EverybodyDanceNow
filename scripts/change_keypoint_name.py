"""
将openpose输出的keypoint文件名改成符合格式的文件名
python -m scripts.change_keypoint_name ~/Code/python/playground/urmp-vid01/vn/keypoints -r VidSep_2_vc_01_Jupiter_000000
"""

from pathlib import Path
from typed_args import TypedArgs, add_argument
from dataclasses import dataclass
import os
from tqdm import tqdm


@dataclass
class Args(TypedArgs):
    input_dir: Path = add_argument()
    # replace: str = add_argument('-r', '--replace')


def main():
    args = Args.from_args()

    example_file = next(args.input_dir.glob('*_face.yml'))
    replace_name = example_file.name[:-15]
    print('Replace name:', replace_name)

    # exit()

    for filename in tqdm(os.listdir(args.input_dir)):
        new_file: Path = args.input_dir / filename.replace(
            replace_name, 'frame'
        )
        (args.input_dir / filename).replace(new_file)


if __name__ == "__main__":
    main()
