import gc
import json
import os
import pandas as pd
import time
import torch

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
from transformers import (AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,)
from trl import SFTConfig, SFTTrainer

from utils import model_paths, lang_codes

torch.manual_seed(6)

def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

def print_header(txt: str): 
    print('*'*50, flush=True)
    print(txt, flush=True)
    print('*'*50, flush=True)

class GemmaL2WritingAssistant(): 
    def __init__(self, 
                 l1: str, 
                 l2: str, 
                 out_dir: str,
                 finetune: bool = False,
                 quantization: str = '4bit',): 
        print_header('Initializing')
        self.l1 = l1
        self.l2 = l2
        self.out_dir = out_dir
        self.info = {**locals()}
        self.info['model'] = 'gemma'
        self.info.pop('self')
        self.model_path = model_paths['gemma']
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
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                device_map='auto',
                quantization_config=self.q_config,
                # use_cache=True,
                dtype=self.d_type,
            )
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
        # self.processor.chat_template = open("L2WritingAssistant/models/gemma_chat_templates/l2wa.jinja").read()
        self.processor.padding_side = "left"

    def sft_preprocess(self, example): 
        source_lang = lang_codes[example['l1']]
        target_lang = lang_codes[example['l2']]
        return {
            'prompt': f"""user\nYou are a professional {source_lang} ({example['l1']}) to {target_lang} ({example['l2']}) translator. 
Your goal is to accurately convey the meaning and nuances of the original {source_lang} text while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.
Produce only the {target_lang} translation, without any additional explanations or commentary. Please translate the following {source_lang} text into {target_lang}:\n\n{example['input']}

model\n
""", 
            'completion': example['ref']
        }

    def preprocess(self, example): 
        return {
            'prompt': {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": example['l1'],
                        "target_lang_code": example['l2'],
                        "text": example['input'],
                    }
                ],
            },
        }

    def finetune(self, ds): 
        print_header('Training')
        ds['train'] = ds['train'].map(self.sft_preprocess, remove_columns=['input', 'ref', 'alts', 'l1', 'l2'])
        ds['val'] = ds['val'].map(self.sft_preprocess, remove_columns=['input', 'ref', 'alts', 'l1', 'l2'])

        model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=self.d_type,
            quantization_config=self.q_config,
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
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
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
            processing_class=self.processor,
        )

        trainer.train()

        peft_model = peft_model.merge_and_unload()
        peft_model.save_pretrained(save_directory=f'{self.out_dir}/models/final', push_to_hub=False)
        self.processor.save_pretrained(save_directory=f'{self.out_dir}/processor', push_to_hub=False)
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
            self.model = AutoModelForImageTextToText.from_pretrained(f'models/final',
                                                                    dtype=self.d_type,
                                                                    device_map='auto',
                                                                    trust_remote_code=True,
                                                                    )

        # Generate text and calculate BLEU
        # TODO: Batch generation by applying chat template to test_ds['prompt']
        with torch.inference_mode():
            for sent in tqdm(test_ds): 
                lang_pair_indices.append(f"{sent['l1']}-{sent['l2']}")
                input_ids = self.processor.apply_chat_template(
                    [sent['prompt']], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
                ).to(self.model.device, dtype=torch.bfloat16)
                input_len = len(input_ids['input_ids'][0])
                t = time.time()
                generated = self.model.generate(**input_ids, max_new_tokens=100, do_sample=False)
                output_data['Time'].append(time.time() - t)
                out_text = self.processor.decode(generated[0][input_len:], skip_special_tokens=True)

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
        full_text = self.processor.decode(generated[0], skip_special_tokens=True)
        for pair in set(lang_pair_indices): 
            data = all_data.loc[pair].copy()
            data.sort_values(by=['BLEU'], inplace=True)
            averages = data[['BLEU', 'Time']].mean()
            results = {
                'Model Info': self.info,
                'Example Prompt': full_text.split('model')[0] + 'model',
                'Avg BLEU': averages.loc['BLEU'],
                'Avg Generation Time (s)': averages.loc['Time'], 
                'Outputs': data.to_dict("records"),
            }

            output_file = f'{self.out_dir}/results/{pair}_results.json'
            with open(output_file, 'w', encoding='utf8') as f: 
                json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__": 
    test = GemmaL2WritingAssistant(
        model='t5',
        l1='fr', 
        l2='en',
        out_dir='test/dir',
        size='small',
        finetune=True,
        quantization='4bit',
    )
    print(test.info)