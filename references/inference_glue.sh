export TASK_NAME=mrpc

OUT_PATH=/home/nfs_data/zhanggh/tmp/$TASK_NAME/

CUDA_VISIBLE_DEVICES=0 python run_glue.py \
  --model_name_or_path $OUT_PATH \
  --task_name $TASK_NAME \
  --cache_dir /home/nfs_data/zhanggh/.cache \
  --do_check \
  --output_attentions \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --output_dir $OUT_PATH\
  --overwrite_output_dir