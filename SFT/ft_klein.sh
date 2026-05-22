#!/bin/bash
NAME="ETCHR_SFT"
Data_Path="YOUR_DATA_PATH"
Output_Path="output"
mkdir -p "log"
export DIFFSYNTH_SKIP_DOWNLOAD=True

accelerate launch \
  --num_processes=8 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  DiffSynth-Studio/examples/flux2/model_training/train.py \
  --dataset_base_path "${Data_Path}" \
  --dataset_metadata_path "data/sft_data.csv" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_paths 'Your_Model_Path' \
  --tokenizer_path "Your_Tokenizer_Path"\
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${Output_Path}" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out,single_transformer_blocks.20.attn.to_out,single_transformer_blocks.21.attn.to_out,single_transformer_blocks.22.attn.to_out,single_transformer_blocks.23.attn.to_out" \
  --lora_rank 768 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 >"log/${NAME}.txt" 2>&1