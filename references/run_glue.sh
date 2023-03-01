export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --cache_dir /home/nfs_data/zhanggh/.cache \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir /home/nfs_data/zhanggh/tmp/$TASK_NAME/ \
  --overwrite_output_dir