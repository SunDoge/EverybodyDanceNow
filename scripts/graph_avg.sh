set -x

# PT_NAME=Sep_1_vn_44_K515_-04%3A18-04%3A24
# PT_NAME=Sep_1_vn_44_K515_-02%3A54-03%3A00
# PT_NAME=Sep_1_vn_44_K515_-02%3A30-02%3A36
# PT_NAME=Sep_1_vn_26_King_-01%3A18-01%3A24
PT_NAME=Sep_1_vn_25_Pirates_-00%3A42-00%3A48

KEYPOINT=temp/${PT_NAME}/keypoints
FRAMES=$HOME/Code/python/playground/urmp-vid01/vn/original_frames

python -m data_prep.graph_avesmooth \
    --keypoints_dir $KEYPOINT \
    --frames_dir $FRAMES \
    --save_dir temp/${PT_NAME}/savefolder \
    --spread 0 130 1 \
    --facetexts
