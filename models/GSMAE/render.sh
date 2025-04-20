
CUDA_VISIBLE_DEVICES=0 python render_mv.py \
     -c config/render_mv.json \
     -s ./WD-Objects/gsmae \
     -m ./WD-Objects/gsmae \
     -vid 3 \
     -cid 0 \
     --reg_alpha \
     -w \
     --static_cam \
     --is_render \
     --mv_num 100 \
     --radius_set 1.5 \
     --elevation_set 20.0 \
     # --gs_ply /data/ckpt/aizixiang/Gaussian-MAE/experiments/pretrain_enc_full_group_xyz_1k/pretrain/gaussian_mae_enc_full_group_xyz_1k/save_ply/1a640c8dffc5d01b8fd30d65663cfd42_ep_0300_full_rebuild_gaussian.ply