ROOT=/home/huangdeng/Code/python/playground/urmp-vid11/vc/savefolder
python test_fullts.py \
    --name vc_global \
    --dataroot $ROOT \
    --checkpoints_dir exps/000-vc \
    --results_dir exps/000-vc-vid11-result-flow \
    --loadSize 512 \
    --no_instance \
    --how_many 2000 \
    --label_nc 6 \
    --flow
