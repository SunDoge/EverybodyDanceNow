# 把声音加入到视频中

set -x

# VIDEO=exps/000-vc-result/vc-videos/output.mp4
# AUDIO=/home/xiaohongdong/URMP/01_Jupiter_vn_vc/AuSep_2_vc_01_Jupiter.wav
# OUTPUT=exps/000-vc-result/vc-videos/output-audio.mp4

# PT_NAME=Sep_1_vn_44_K515_-04%3A18-04%3A24
# PT_NAME=Sep_1_vn_44_K515_-02%3A54-03%3A00
# PT_NAME=Sep_1_vn_44_K515_-02%3A30-02%3A36
# PT_NAME=Sep_1_vn_26_King_-01%3A18-01%3A24
# PT_NAME=Sep_1_vn_25_Pirates_-00%3A42-00%3A48

# PT_NAME=Sep_1_vn_44_K515
# PT_NAME=Sep_1_vn_44_K515_-02%3A54-03%3A00
# PT_NAME=Sep_1_vn_44_K515_-02%3A30-02%3A36
PT_NAME=Sep_1_vn_26_King
# PT_NAME=Sep_1_vn_25_Pirates
SS="00:01:18"
T=6
VIDEO=temp/${PT_NAME}/output/output.mp4
AUDIO=/home/xiaohongdong/URMP/26_King_vn_vn_va_vc/AuSep_1_vn_26_King.wav
OUTPUT=temp/${PT_NAME}/output/output-audio.mp4

ffmpeg -i $VIDEO -ss $SS -t $T -i $AUDIO -c:v copy -map 0:v:0 -map 1:a:0 -max_muxing_queue_size 1024 $OUTPUT