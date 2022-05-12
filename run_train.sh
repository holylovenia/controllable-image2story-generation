# Generic
# CUDA_VISIBLE_DEVICES=1 python train.py \
#    --model_name_or_path="save/decoder_finetuning/gpt2/checkpoint-1751175" \
#    --cache_dir_path="./cache/openai/clip-vit-base-patch32" \
#    --preprocessing_num_workers=8 --image_column_name=image_id --text_column_name=caption \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#    --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
#    --fp16 --fp16_backend=amp \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
#    --gradient_checkpointing=True \
#    --do_train=True \
#    --preprocessing_only=False --overwrite_output_dir=True \

# Romance
# CUDA_VISIBLE_DEVICES=0 python train.py \
#    --model_name_or_path="save/with-adapters/GPT2small_adapterid0_genreRomance_matched3_sampleNone_maxseqlen512_bs8_lr5e-05_2.0epoch_wd0.0_ws0/checkpoint-15000" \
#    --cache_dir_path="./cache/openai/clip-vit-base-patch32" \
#    --preprocessing_num_workers=8 --image_column_name=image_id --text_column_name=caption \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#    --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
#    --fp16 --fp16_backend=amp \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
#    --gradient_checkpointing=True \
#    --do_train=True \
#    --preprocessing_only=False --overwrite_output_dir=True \

# # Action
# CUDA_VISIBLE_DEVICES=1 python train.py \
#    --model_name_or_path="save/with-adapters/GPT2small_adapterid0_genreAction_matched3_sampleNone_maxseqlen512_bs8_lr5e-05_10.0epoch_wd0.0_ws0/checkpoint-15000" \
#    --cache_dir_path="./cache/openai/clip-vit-base-patch32" \
#    --preprocessing_num_workers=8 --image_column_name=image_id --text_column_name=caption \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#    --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
#    --fp16 --fp16_backend=amp \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
#    --gradient_checkpointing=True \
#    --do_train=True \
#    --preprocessing_only=False --overwrite_output_dir=True \

# # Thriller
# CUDA_VISIBLE_DEVICES=4 python train.py \
#    --model_name_or_path="save/with-adapters/GPT2small_adapterid0_genreThriller_matched3_sampleNone_maxseqlen512_bs8_lr0.001_10.0epoch_wd0.0_ws0/checkpoint-30000" \
#    --cache_dir_path="./cache/openai/clip-vit-base-patch32" \
#    --preprocessing_num_workers=8 --image_column_name=image_id --text_column_name=caption \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#    --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
#    --fp16 --fp16_backend=amp \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=10 --save_total_limit=3 --load_best_model_at_end \
#    --gradient_checkpointing=True \
#    --do_train=True \
#    --preprocessing_only=False --overwrite_output_dir=True \

# # Historical
# CUDA_VISIBLE_DEVICES=6 python train.py \
#    --model_name_or_path="save/with-adapters/GPT2small_adapterid0_genreHistorical_matched3_sampleNone_maxseqlen512_bs8_lr0.001_10.0epoch_wd0.0_ws0/checkpoint-10000" \
#    --cache_dir_path="./cache/openai/clip-vit-base-patch32" \
#    --preprocessing_num_workers=8 --image_column_name=image_id --text_column_name=caption \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#    --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
#    --fp16 --fp16_backend=amp \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=10 --save_total_limit=3 --load_best_model_at_end \
#    --gradient_checkpointing=True \
#    --do_train=True \
#    --preprocessing_only=False --overwrite_output_dir=True \

# Mystery
CUDA_VISIBLE_DEVICES=6 python train.py \
   --model_name_or_path="save/with-adapters/GPT2small_adapterid0_genreMystery_matched3_sampleNone_maxseqlen512_bs8_lr0.001_10.0epoch_wd0.0_ws0/checkpoint-5000" \
   --cache_dir_path="./cache/openai/clip-vit-base-patch32" \
   --preprocessing_num_workers=8 --image_column_name=image_id --text_column_name=caption \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
   --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
   --fp16 --fp16_backend=amp \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=10 --save_total_limit=3 --load_best_model_at_end \
   --gradient_checkpointing=True \
   --do_train=True \
   --preprocessing_only=False --overwrite_output_dir=True \