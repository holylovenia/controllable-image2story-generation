CUDA_VISIBLE_DEVICES=2 python adapter_train.py \
   --model_size="medium" \
   --load_checkpoint_adapter="" \
   --genre="Thriller" --adapter_id=0 \
   --match_up_to_n_genres=3 \
   --max_seq_len=512 \
   --dataset_path="/home/bryan/datasets/bookcorpusopen/bookcorpusopen_chunked.arrow" \
   --preprocessing_num_workers=24 --bookcorpusopen_story_column_name=chunk \
   --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
   --dataloader_num_workers=24 --dataloader_pin_memory --group_by_length \
   --seed=14045 --num_train_epochs=3 --learning_rate=1e-3 \
   --fp16 --fp16_backend=amp \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=steps --eval_steps=5000 --eval_accumulation_steps=100 \
   --save_strategy=steps --save_steps=5000 --save_total_limit=3 \
   --do_train=True --do_eval=True \
   --overwrite_output_dir=True \
   --output_dir='./save/' \
   --early_stopping_patience=5 > thriller_medium.log


CUDA_VISIBLE_DEVICES=2 python adapter_train.py \
   --model_size="medium" \
   --load_checkpoint_adapter="" \
   --genre="Science Fiction" --adapter_id=0 \
   --match_up_to_n_genres=3 \
   --max_seq_len=512 \
   --dataset_path="/home/bryan/datasets/bookcorpusopen/bookcorpusopen_chunked.arrow" \
   --preprocessing_num_workers=24 --bookcorpusopen_story_column_name=chunk \
   --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
   --dataloader_num_workers=24 --dataloader_pin_memory --group_by_length \
   --seed=14045 --num_train_epochs=3 --learning_rate=1e-3 \
   --fp16 --fp16_backend=amp \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=steps --eval_steps=5000 --eval_accumulation_steps=100 \
   --save_strategy=steps --save_steps=5000 --save_total_limit=3 \
   --do_train=True --do_eval=True \
   --overwrite_output_dir=True \
   --output_dir='./save/' \
   --early_stopping_patience=5 > sciencefiction_medium.log
   

CUDA_VISIBLE_DEVICES=2 python adapter_train.py \
   --model_size="medium" \
   --load_checkpoint_adapter="" \
   --genre="Historical" --adapter_id=0 \
   --match_up_to_n_genres=3 \
   --max_seq_len=512 \
   --dataset_path="/home/bryan/datasets/bookcorpusopen/bookcorpusopen_chunked.arrow" \
   --preprocessing_num_workers=24 --bookcorpusopen_story_column_name=chunk \
   --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
   --dataloader_num_workers=24 --dataloader_pin_memory --group_by_length \
   --seed=14045 --num_train_epochs=3 --learning_rate=1e-3 \
   --fp16 --fp16_backend=amp \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=steps --eval_steps=5000 --eval_accumulation_steps=100 \
   --save_strategy=steps --save_steps=5000 --save_total_limit=3 \
   --do_train=True --do_eval=True \
   --overwrite_output_dir=True \
   --output_dir='./save/' \
   --early_stopping_patience=5 > Historical_medium.log
   
   
CUDA_VISIBLE_DEVICES=2 python adapter_train.py \
   --model_size="medium" \
   --load_checkpoint_adapter="" \
   --genre="Adventure" --adapter_id=0 \
   --match_up_to_n_genres=3 \
   --max_seq_len=512 \
   --dataset_path="/home/bryan/datasets/bookcorpusopen/bookcorpusopen_chunked.arrow" \
   --preprocessing_num_workers=24 --bookcorpusopen_story_column_name=chunk \
   --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
   --dataloader_num_workers=24 --dataloader_pin_memory --group_by_length \
   --seed=14045 --num_train_epochs=3 --learning_rate=1e-3 \
   --fp16 --fp16_backend=amp \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=steps --eval_steps=5000 --eval_accumulation_steps=100 \
   --save_strategy=steps --save_steps=5000 --save_total_limit=3 \
   --do_train=True --do_eval=True \
   --overwrite_output_dir=True \
   --output_dir='./save/' \
   --early_stopping_patience=5 > Adventure_medium.log
   

CUDA_VISIBLE_DEVICES=2 python adapter_train.py \
   --model_size="medium" \
   --load_checkpoint_adapter="" \
   --genre="Romance" --adapter_id=0 \
   --match_up_to_n_genres=3 \
   --max_seq_len=512 \
   --dataset_path="/home/bryan/datasets/bookcorpusopen/bookcorpusopen_chunked.arrow" \
   --preprocessing_num_workers=24 --bookcorpusopen_story_column_name=chunk \
   --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
   --dataloader_num_workers=24 --dataloader_pin_memory --group_by_length \
   --seed=14045 --num_train_epochs=3 --learning_rate=1e-3 \
   --fp16 --fp16_backend=amp \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=steps --eval_steps=5000 --eval_accumulation_steps=100 \
   --save_strategy=steps --save_steps=5000 --save_total_limit=3 \
   --do_train=True --do_eval=True \
   --overwrite_output_dir=True \
   --output_dir='./save/' \
   --early_stopping_patience=5 > Romance_medium.log