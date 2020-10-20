# 把声音加入到视频中

set -x

VIDEO=exps/000-vc-result/vc-videos/output.mp4
AUDIO=/home/xiaohongdong/URMP/01_Jupiter_vn_vc/AuSep_2_vc_01_Jupiter.wav
OUTPUT=exps/000-vc-result/vc-videos/output-audio.mp4
ffmpeg -i $VIDEO -i $AUDIO -map 0:v:0 -map 1:a:0 $OUTPUT