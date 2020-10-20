set -x

KEYPOINT=temp/keypoints
FRAMES=$HOME/Code/python/playground/urmp-vid01/vn/original_frames

python -m data_prep.graph_avesmooth \
--keypoints_dir $KEYPOINT \
--frames_dir $FRAMES \
--save_dir temp/savefolder \
--spread 0 130 1 \
--facetexts