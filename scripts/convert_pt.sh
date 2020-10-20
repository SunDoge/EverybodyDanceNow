set -x

# PT_NAME=Sep_1_vn_44_K515_-04%3A18-04%3A24
# PT_NAME=Sep_1_vn_44_K515_-02%3A54-03%3A00
# PT_NAME=Sep_1_vn_44_K515_-02%3A30-02%3A36
# PT_NAME=Sep_1_vn_26_King_-01%3A18-01%3A24
PT_NAME=Sep_1_vn_25_Pirates_-00%3A42-00%3A48

ARGS=(python -m scripts.pt_to_yml)
ARGS+=(/home/xuhaoming/Projects/motion-to-sound-master/exps-urmp/1018/vn-correct-startEmbedding-1layer/gen_fix_center/pose/examples/${PT_NAME}.pt)
ARGS+=(temp/${PT_NAME}/keypoints/)

${ARGS[@]}
