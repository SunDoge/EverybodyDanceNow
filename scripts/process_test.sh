# 处理test的keypoints
# TARGET=/home/huangdeng/Code/python/playground/urmp-vid01/vn
# SOURCE=/home/huangdeng/Code/python/playground/urmp-vid01/vn
# python -m data_prep.graph_posenorm \
#     --target_keypoints $TARGET/keypoints \
#     --source_keypoints $SOURCE/keypoints \
#     --target_shape 1080 1920 3 \
#     --source_shape 1080 1920 3 \
#     --source_frames $SOURCE/original_frames \
#     --results $TARGET/savefolder \
#     --target_spread 0 1800 \
#     --source_spread 0 1800 \
#     --calculate_scale_translation \
#     --map_25_to_23

FRAMES=$HOME/Code/python/playground/urmp-vid01/vn/original_frames

TARGET=/home/huangdeng/Code/python/playground/urmp-vid01/vn
SOURCE=temp
python -m data_prep.graph_posenorm \
    --target_keypoints $TARGET/keypoints \
    --source_keypoints $SOURCE/keypoints \
    --target_shape 1080 1920 3 \
    --source_shape 1080 1920 3 \
    --source_frames $SOURCE/original_frames \
    --results $SOURCE/savefolder \
    --target_spread 0 130 \
    --source_spread 0 130 \
    --calculate_scale_translation \
    --map_25_to_23
