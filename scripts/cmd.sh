OPENPOSE_BIN=/openpose/build/examples/openpose/openpose.bin
VIDEO=/home/huangdeng/Datasets/URMP_video_sep/vc/VidSep_2_vc_01_Jupiter.mp4

# do not open display window
$OPENPOSE_BIN \
--face \
--hand \
--video $VIDEO \
--display 0 \
--render_pose 0 \
--write_keypoint_format yml \
--write_keypoint keypoints/ 

# 分帧
ffmpeg -i $VIDEO original_frames/frame%06d.png

ROOT=/home/huangdeng/Code/python/playground/urmp-vid01/vn
python -m data_prep.graph_train \
--keypoints_dir $ROOT/keypoints \
--frames_dir $ROOT/original_frames \
--save_dir $ROOT/savefolder \
--spread 0 2000 1 \
--map_25_to_23 \
--facetexts
