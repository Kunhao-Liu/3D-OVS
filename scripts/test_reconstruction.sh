expname=table
CUDA_VISIBLE_DEVICES=$1 python train.py \
--config configs/reconstruction.txt \
--ckpt log/$expname/$expname.th \
--render_only 1 \
--render_train 1