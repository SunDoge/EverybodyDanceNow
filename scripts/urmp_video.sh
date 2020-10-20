set -x
# PT_NAME=Sep_1_vn_44_K515
# PT_NAME=Sep_1_vn_44_K515_-02%3A54-03%3A00
# PT_NAME=Sep_1_vn_44_K515_-02%3A30-02%3A36
# PT_NAME=Sep_1_vn_26_King
PT_NAME=Sep_1_vn_25_Pirates

ARGS=(python -m scripts.create_video)
ARGS+=(-p "temp/${PT_NAME}/savefolder/test_label")
ARGS+=(-f "temp/${PT_NAME}/result/vn_global/test_latest/images")
ARGS+=(temp/${PT_NAME}/output)

${ARGS[@]}
