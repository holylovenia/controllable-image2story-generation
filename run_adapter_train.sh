CUDA_VISIBLE_DEVICES=2 python adapter_train.py \
   --model_size="medium" \
   --load_checkpoint_adapter="" \
   --genre="Fiction" --adapter_id=1 \
   --match_up_to_n_genres=3 \
   --max_seq_len=256 \
   --dataset_path="/home/bryan/datasets/bookcorpusopen/bookcorpusopen_chunked.arrow" \
   --preprocessing_num_workers=8 --bookcorpusopen_story_column_name=chunk \
   --per_device_train_batch_size=1 --per_device_eval_batch_size=1 \
   --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
   --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
   --fp16 --fp16_backend=amp \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=100 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 \
   --do_train=True --do_eval=True \
   --overwrite_output_dir=True \