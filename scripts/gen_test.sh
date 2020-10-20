# 生成测试图片

set -x

# PT_NAME=Sep_1_vn_44_K515_-04%3A18-04%3A24
# PT_NAME=Sep_1_vn_44_K515_-02%3A54-03%3A00
# PT_NAME=Sep_1_vn_44_K515_-02%3A30-02%3A36
# PT_NAME=Sep_1_vn_26_King_-01%3A18-01%3A24
PT_NAME=Sep_1_vn_25_Pirates_-00%3A42-00%3A48

# ROOT=/home/huangdeng/Code/python/playground/urmp-vid11/vc/savefolder
ROOT=temp/${PT_NAME}/savefolder
RESULT=temp/${PT_NAME}/result

ARGS=(python test_fullts.py)
ARGS+=(--name vn_global)
ARGS+=(--dataroot $ROOT)
ARGS+=(--checkpoints_dir exps/001-vn-vid01)
# ARGS+=(--results_dir exps/001-vc-vid11-result)
ARGS+=(--results_dir $RESULT)
ARGS+=(--loadSize 512)
ARGS+=(--no_instance)
ARGS+=(--how_many 2000)
ARGS+=(--label_nc 6)
# ARGS+=(--flow)

${ARGS[@]}
