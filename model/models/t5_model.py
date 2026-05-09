import gc
import json
import os
import pandas as pd
import time
import torch

from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pathlib import Path
from peft import (
    LoraConfig, 
    PeftConfig,
    PeftModel,
    TaskType, 
    get_peft_model
)
from tqdm import tqdm
from transformers import (AutoTokenizer, T5ForConditionalGeneration, BitsAndBytesConfig,)
from trl import SFTConfig, SFTTrainer

from utils import lang_codes, model_paths

torch.manual_seed(6)

def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

def print_header(txt: str): 
    print('*'*50, flush=True)
    print(txt, flush=True)
    print('*'*50, flush=True)

class T5L2WritingAssistant(): 
    def __init__(self, 
                 model: str, 
                 l1: str, 
                 l2: str, 
                 out_dir: str,
                 size: str = 'base', 
                 #finetune can probably just be an int
                 finetune: bool = False,
                 quantization: str = '4bit',): 
        print_header('Initializing')
        # self.model_name = model.name.lower()
        # self.size = size.name.lower()
        self.model_name = model.lower()
        self.size = size.lower()
        self.l1 = l1
        self.l2 = l2
        self.out_dir = out_dir
        self.info = {**locals()}
        self.info.pop('self')
        # self.out_dir = '/data/shelbyho/capstone_outputs/41981_t5_small_fr_en'
        if self.model_name == 't5': 
            # valid_sizes = ['small', 'base', 'large', 'xl', 'xxl']
            # if size.name.lower() not in valid_sizes: 
            #     raise ValueError(f'Invalid size. Size must be in {valid_sizes}')
            model_path = f'{model_paths[self.model_name]}-{self.size}'
        elif self.model_name == 'madlad': 
            model_path = model_paths[self.model_name]
        else: 
            raise ValueError(f'Model not supported. Currently supports T5 and Madlad')
        self.model_path = model_path
        self.model = None

        if quantization == '4bit':
            self.q_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.d_type = torch.bfloat16
        elif quantization == '8bit': 
            self.q_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            self.d_type = 'auto'
        else: 
            self.q_config = None
            self.d_type = 'auto'
        
        if not finetune: 
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_path,
                device_map='auto',
                quantization_config=self.q_config,
                use_cache=True,
                dtype=self.d_type,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"

    def preprocess(self, example): 
        if self.model_name == 't5': 
            prompt = f'translate from {lang_codes[example['l1']]} to {lang_codes[example['l2']]}: {example['input']}',
            prompt = prompt[0]
        else: 
            prompt = f'<2{example['l2']}> {example['input']}'

        return {
            'prompt': prompt,
            'completion': example['ref']
        }

    def finetune(self, ds): 
        print_header('Training')
        ds['train'] = ds['train'].map(self.preprocess, remove_columns=['input', 'ref', 'alts', 'l1', 'l2'])
        ds['val'] = ds['val'].map(self.preprocess, remove_columns=['input', 'ref', 'alts', 'l1', 'l2'])

        model = T5ForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.d_type,
            quantization_config=self.q_config,
            use_cache=True,
            trust_remote_code=True,
        )

        local_model = Path(f'{self.out_dir}/models/final')
        if local_model.exists(): 
            os.chdir(self.out_dir)
            model_dirs = [name for name in os.listdir('models')
                          if os.path.isdir(os.path.join('models', name))]
            ckpt_dir = f'models/{sorted(model_dirs)[0]}'
            peft_config = PeftConfig.from_pretrained(ckpt_dir)
            peft_model = PeftModel.from_pretrained(model, ckpt_dir, is_trainable=True)
        else: 
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=64,
                lora_alpha=64,
                lora_dropout=0.1,
                target_modules=['q', 'k', 'v', 'o']
            )

            peft_model = get_peft_model(model, lora_config)

        args = SFTConfig(
            output_dir=f'{self.out_dir}/models',
            learning_rate=6.3e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            fp16=False,
            bf16=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            report_to='none',
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            push_to_hub=False,
            save_total_limit=1,
        )

        trainer = SFTTrainer(
            model=peft_model,
            args=args,
            train_dataset=ds['train'],
            eval_dataset=ds['val'],
            processing_class=self.tokenizer,
        )

        trainer.train()

        peft_model = peft_model.merge_and_unload()
        peft_model.save_pretrained(save_directory=f'{self.out_dir}/models/final', push_to_hub=False)
        del peft_model
        del trainer
        flush()

    def evaluate(self, test_ds): 
        # Since the number of sentences is relatively small, we *start* with doing inference
        # one at a time. 
        # TODO: implement batch inference
        test_ds = test_ds.map(self.preprocess, remove_columns=['ref'])

        smoothing = SmoothingFunction()
        output_data = {
            "Input": [],
            "Generated": [], 
            "Gold": [], 
            "BLEU": [], 
            "Time": [], 
        }
        lang_pair_indices = []

        if self.model is None: 
            os.chdir(self.out_dir)
            self.model = T5ForConditionalGeneration.from_pretrained(
                f'models/final',
                dtype=self.d_type,
                device_map='auto',
                use_cache=True,
                trust_remote_code=True,
            )

        # Generate text and calculate BLEU
        with torch.inference_mode():
            for sent in tqdm(test_ds): 
                lang_pair_indices.append(f"{sent['l1']}-{sent['l2']}")
                input_ids = self.tokenizer(sent['prompt'], return_tensors='pt').input_ids.to('cuda')
                t = time.time()
                gen_out = self.model.generate(input_ids, max_new_tokens=100, do_sample=False)
                output_data['Time'].append(time.time() - t)
                out_text = self.tokenizer.decode(gen_out[0], skip_special_tokens=True)

                output_data['Input'].append(sent['input'])
                output_data['Generated'].append(out_text)
                output_data['Gold'].append(sent['alts'])

                bleu = sentence_bleu(
                    [alt.split(' ') for alt in sent['alts']], 
                    out_text.split(' '), 
                    smoothing_function = smoothing.method1,
                )
                output_data['BLEU'].append(bleu)
        
        # Save output
        os.mkdir(f'{self.out_dir}/results')
        all_data = pd.DataFrame(data=output_data, index=lang_pair_indices)
        for pair in set(lang_pair_indices): 
            data = all_data.loc[pair].copy()
            data.sort_values(by=['BLEU'], inplace=True)
            averages = data[['BLEU', 'Time']].mean()
            results = {
                'Model Info': self.info,
                'Example Prompt': repr(test_ds[0]['prompt']),
                'Avg BLEU': averages.loc['BLEU'],
                'Avg Generation Time (s)': averages.loc['Time'], 
                'Outputs': data.to_dict("records"),
            }

            output_file = f'{self.out_dir}/results/{pair}_results.json'
            with open(output_file, 'w', encoding='utf8') as f: 
                json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__": 
    test = T5L2WritingAssistant(
        model='t5',
        l1='fr', 
        l2='en',
        out_dir='test/dir',
        size='small',
        finetune=True,
        quantization='4bit',
    )
    print(test.info)