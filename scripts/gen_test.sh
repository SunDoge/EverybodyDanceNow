# 生成测试图片

set -x

# ROOT=/home/huangdeng/Code/python/playground/urmp-vid11/vc/savefolder
ROOT=temp/savefolder
RESULT=temp/result

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
