CUDA_VISIBLE_DEVICES=$3 python train_seg.py \
--config $2 \
--ckpt $1 \
--render_only 1 \
--render_seg_train 1 \
--render_seg_test 1 \
--render_seg_path 1 \
--render_seg_depth 0 \
--render_feature 0 \
--render_select 0 \
--downsample_test 8 \
# --reference_text 'Hatsune Miku cartoon statue'