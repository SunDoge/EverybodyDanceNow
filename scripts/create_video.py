"""
读取生成图片和生成骨架图
生成两个视频
再把两个视频拼起来
"""

from typing import List
from typed_args import TypedArgs, add_argument
from dataclasses import dataclass
import subprocess
from pathlib import Path
from shlex import quote


PATTERNS = {
    'frames': 'frame%06d_synthesized_image.png',
    'pose': 'frame%06d.png'
}


@dataclass
class Args(TypedArgs):

    pose: Path = add_argument('-p', '--pose', help='path to skeleton images')
    frames: Path = add_argument('-f', '--frames', help='path to frame images')
    fps: float = add_argument('--fps', default=29.97)
    scale: str = add_argument('--scale', default='256:256')
    output: Path = add_argument(help='output dir')


def generate_video(
    in_dir: Path,
    image_type: str,
    out_dir: Path,
    fps: float = 29.97,
    scale: str = '256:256',
    crop: str = 'in_w/2:in_h:0:0'
):
    out_path = out_dir / f'{image_type}.mp4'
    cmd = [
        'ffmpeg',
        '-r', str(fps),
        '-i', quote(str(in_dir / PATTERNS[image_type])),
        # '-filter:v', f'crop={crop}'
        '-c:v', 'mpeg4',
        '-vf', f'scale={scale}',
        str(out_path)
    ]
    subprocess.run(cmd)
    return out_path


def concat_video(video1: str, video2: str, out_dir: Path):
    out_path = out_dir / 'output.mp4'
    cmd = [
        'ffmpeg',
        '-i', video1,
        '-i', video2,
        '-c:v', 'mpeg4',
        '-filter_complex', 'hstack',
        str(out_path)
    ]

    subprocess.run(cmd)
    return out_path


def main():
    args = Args.from_args()

    args.output.mkdir(exist_ok=True)

    video1 = generate_video(args.pose, 'pose', args.output,
                            fps=args.fps, scale=args.scale)
    print('generate video1:', video1)
    video2 = generate_video(args.frames, 'frames',
                            args.output, fps=args.fps, scale=args.scale)
    print('generate video2:', video2)

    video_hstack = concat_video(str(video1), str(video2), args.output)
    print('save final video to:', video_hstack)


if __name__ == "__main__":
    main()
