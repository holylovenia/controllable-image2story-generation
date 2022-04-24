# CLIP embeddings
CUDA_VISIBLE_DEVICES=1 python decoder_train.py \
   --model_size="medium" \
   --load_checkpoint_adapter="" \
   --cache_dir_name="./cache/adapter_finetuning" \
   --dataset_path="/home/bryan/datasets/bookcorpusopen/bookcorpusopen_chunked.arrow"\
   --preprocessing_num_workers=8 --bookcorpusopen_story_column_name=chunk \
   --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
   --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
   --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
   --fp16 --fp16_backend=amp \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=100 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --gradient_checkpointing=True \
   --do_train=True --do_eval=True \
   --overwrite_output_dir=True \