# Controllable Image to Story Generation
Every picture tells a story: Controllable neural story generation from images

### PPCM
How to:
1. Make sure all requirements are installed, or install it via: `pip install -r requirements.txt`
2. Get your copy of bookcorpusopen_chunked dataset via __copying__ to __`your/own/path`__, either from:
    - `/home/bryan/datasets/bookcorpusopen/bookcorpusopen_chunked.arrow`
    - `/home/holy/datasets/bookcorpusopen/bookcorpusopen_chunked.arrow`
2. Inside this repo, do `git clone https://github.com/andreamad8/PPCM.git` and `cd PPCM`
3. Download the PPCM models, run `./download_data.sh`
4. Move the PPCM models to our repo, run:
    1. `cd ..`
    2. `mkdir ppcm_models/dialoGPT`
    3. `mv PPCM/models/dialoGPT/* ppcm_models/dialoGPT/`
4. In the `run_adapter_train.sh` change the dataset_path to __`your/own/path`__
5. Run `bash run_adapter_train.sh` to train the adapter with the designated book genres
6. Post training, load the model using, e.g.:
    ```python
    import os
    from transformers import GPT2Tokenizer, TrainingArguments
    from utils.helper import load_model_recursive
    from ppcm_models.pytorch_pretrained_bert.modeling_adapter import GPT2LMHeadModel, GPT2Config
    
    model_args.model_path = f'ppcm_models/dialoGPT/small/'
    config = GPT2Config.from_json_file(os.path.join(model_args.model_path, 'config.json'))
    tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_path)
    
    path = './save/model_run_name/pytorch_model.bin'
    model = load_model_recursive(GPT2LMHeadModel(config), path, model_args, verbose=True)
    
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(inputs['input_ids'], task_id=[0])
    # or
    outputs = model.transformer(inputs['input_ids'], task_id=0)
    ```