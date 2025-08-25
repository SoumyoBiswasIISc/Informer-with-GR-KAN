#!/bin/bash
# Script to fine-tune GR-KAN Informer from 4 best Informer checkpoints
# Runs 2 jobs in parallel (GPU 0 and GPU 1), then the next 2.

# Base command (common args)
BASE_CMD="python main_informer.py \
  --model informer \
  --data custom \
  --root_path ./data/ \
  --data_path integrated_dataset.csv \
  --features MS \
  --target Ground_Truth \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 1 \
  --enc_in 8 \
  --dec_in 1 \
  --c_out 1 \
  --cols NDVI EVI NIR-V CSIF Rainfall Tmax Tmin Solr_mean \
  --train_epochs 20 \
  --learning_rate 0.0001 \
  --patience 5 \
  --des rice_yield_grkan \
  --itr 1 \
  --attn full"

# Checkpoints (best runs from each config)
CKPTS=(
"/user1/res/cvpr/soumyo.b_r/Informer2020/checkpoints/informer_custom_ftMS_sl12_ll6_pl1_dm512_nh8_el2_dl1_df1024_atprob_fc5_ebtimeF_dtTrue_mxTrue_rice_yield_kharif_0/checkpoint.pth"
"/user1/res/cvpr/soumyo.b_r/Informer2020/checkpoints/informer_custom_ftMS_sl12_ll6_pl1_dm256_nh8_el2_dl1_df1024_atprob_fc5_ebtimeF_dtTrue_mxTrue_rice_yield_kharif_1/checkpoint.pth"
"/user1/res/cvpr/soumyo.b_r/Informer2020/checkpoints/informer_custom_ftMS_sl12_ll6_pl1_dm256_nh8_el2_dl1_df1024_atprob_fc3_ebtimeF_dtTrue_mxTrue_rice_yield_kharif_1/checkpoint.pth"
"/user1/res/cvpr/soumyo.b_r/Informer2020/checkpoints/informer_custom_ftMS_sl12_ll6_pl1_dm512_nh8_el2_dl1_df1024_atprob_fc3_ebtimeF_dtTrue_mxTrue_rice_yield_kharif_0/checkpoint.pth"
)

# Run first 2 jobs in parallel
CUDA_VISIBLE_DEVICES=0 $BASE_CMD --pretrain_path ${CKPTS[0]} > grkan_run1.log 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=1 $BASE_CMD --pretrain_path ${CKPTS[1]} > grkan_run2.log 2>&1 &
PID2=$!

# Wait for them to finish
wait $PID1 $PID2

# Run next 2 jobs in parallel
CUDA_VISIBLE_DEVICES=0 $BASE_CMD --pretrain_path ${CKPTS[2]} > grkan_run3.log 2>&1 &
PID3=$!
CUDA_VISIBLE_DEVICES=1 $BASE_CMD --pretrain_path ${CKPTS[3]} > grkan_run4.log 2>&1 &
PID4=$!

# Wait for them to finish
wait $PID3 $PID4

echo "All GR-KAN fine-tuning jobs completed."
