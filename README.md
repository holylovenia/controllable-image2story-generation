# Controllable Image to Story Generation
Every picture tells a story: Controllable neural story generation from images

### PPCM
How to:
1. Make sure all requirements are installed, or install it via: `pip install -r requirements.txt`
2. Get your copy of bookcorpusopen_chunked dataset via __copying__ to __`your/own/path`__, either from:
    - `/home/bryan/datasets/bookcorpusopen/bookcorpusopen_chunked.arrow`
    - `/home/holy/datasets/bookcorpusopen/bookcorpusopen_chunked.arrow`
2. Run `git clone https://github.com/andreamad8/PPCM.git` and `cd PPCM`
3. Download the PPCM models, run `./download_data.sh`
4. Move the PPCM models to our repo, run:
    1. `cd ..`
    2. `mkdir ppcm_models/dialoGPT`
    3. `mv PPCM/models/dialoGPT/* ppcm_models/dialoGPT/`
4. In the `run_adapter_train.sh` change the dataset_path to __`your/own/path`__
5. Run `bash run_adapter_train.sh` to train the adapter with the designated book genres
6. Post training, predict and get the outputs using, e.g.:
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
7. Or generate using, e.g.:
    ```python
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import GPT2Tokenizer, TrainingArguments
    from utils.helper import load_model_recursive
    from ppcm_models.pytorch_pretrained_bert.modeling_adapter import GPT2LMHeadModel, GPT2Config
    
    model_args.model_path = f'ppcm_models/dialoGPT/small/'
    config = GPT2Config.from_json_file(os.path.join(model_args.model_path, 'config.json'))
    tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_path)
    
    model_run_names = ['GPT2small_adapterid0_genreAction_matched3_sampleNone_maxseqlen512_bs8_lr5e-05_10.0epoch_wd0.0_ws0',
                       'GPT2small_adapterid0_genreAction_matched3_sampleNone_maxseqlen512_bs8_lr5e-05_2.0epoch_wd0.0_ws0',
                       'GPT2small_adapterid0_genreMystery_matched3_sampleNone_maxseqlen512_bs8_lr0.0005_5.0epoch_wd0.0_ws0',
                       'GPT2small_adapterid0_genreRomance_matched3_sampleNone_maxseqlen512_bs8_lr5e-05_2.0epoch_wd0.0_ws0']

    for i, model_run_name in enumerate(model_run_names):

        path = './save/'+model_run_name+'/pytorch_model.bin'
        model = load_model_recursive(GPT2LMHeadModel(config), path, model_args, verbose=True)

        length = 100
        text = "Hello, my dog is cute"
        generated = tokenizer.encode(text)
        context = torch.tensor([generated])

        generation = model.generate(inputs=context,
                                   num_beams=3, 
                                   length_penalty=3, 
                                   early_stopping=1, 
                                   num_beam_groups=3, 
                                   do_sample=False, 
                                   num_return_sequences=2, 
                                   bos_token_id=50256,
                                   eos_token_id=50256,
                                   pad_token_id=50256,
                                   output_scores=True,
                                   output_attentions=True,
                                   output_hidden_states=True,
                                   return_dict_in_generate=True,
                                   repetition_penalty=1.1,
                                   min_length = 0,
                                   max_length = length,
                                   no_repeat_ngram_size=2,
                                   encoder_no_repeat_ngram_size=False,
                                   bad_words_ids=[[100]], # tokenizer.decode(100)
                                   diversity_penalty=0.2,
                                   forced_bos_token_id=50256,
                                   forced_eos_token_id=50256,
                                   remove_invalid_values=True,
                                   exponential_decay_length_penalty=[1.0, 1.2])

        print(tokenizer.decode(generation[0][0]), '\n')
    ```