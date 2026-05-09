from process_data import make_semeval_dataset, make_nllb_dataset
from models.t5_model import T5L2WritingAssistant
from models.gemma_model import GemmaL2WritingAssistant
from utils import valid_pairs

import click
import enum 

class Model(enum.Enum): 
    t5 = enum.auto()
    madlad = enum.auto()
    gemma = enum.auto()

class T5Size(enum.Enum): 
    small = enum.auto()
    base = enum.auto()
    large = enum.auto()
    xl = enum.auto()
    xxl = enum.auto()
    
@click.command()
# Perhaps change to singular 'pair' arg
@click.argument('l1')
@click.argument('l2')
@click.argument('output-dir')
@click.option('--model', type=click.STRING, default='t5', help='Model to use')
@click.option('--size', type=click.STRING, default='base', help='Size of T5 model to use')
# @click.option('--model', type=click.Choice(Model, case_sensitive=False), default='t5', help='Model to use.')
# @click.option('--size', type=click.Choice(T5Size, case_sensitive=False), default='base', help='Size of T5 model to use.')
@click.option('--finetune', type=click.IntRange(0, 3), default=0)
@click.option('--split', type=click.IntRange(0, 100, clamp=True), default=10)
@click.option('--quantize', default='4bit', help='Precision to quantize to. Must be either ["4bit", "8bit", "full"]')
@click.option('--eval-all', type=click.BOOL, default=False)
def main(l1, l2, output_dir, size, model, finetune, split, quantize, eval_all): 
    if (l1, l2) not in valid_pairs and (l1, l2) != ('all', 'all'): 
        raise ValueError(f'{l1}-{l2} not supported.')

    if model != 'gemma': 
        model = T5L2WritingAssistant(
            model=model,
            l1=l1,
            l2=l2,
            out_dir=output_dir,
            size=size,
            finetune=bool(finetune),
            quantization=quantize,
        )
    else: 
        model = GemmaL2WritingAssistant(
            l1=l1,
            l2=l2, 
            out_dir=output_dir,
            finetune=bool(finetune),
            quantization=quantize,
        )
    ds = make_semeval_dataset(l1=l1, l2=l2, do_split=True)
    if eval_all: 
        test_ds = make_semeval_dataset(l1='all', l2='all', do_split=True)['test']
    else: 
        test_ds = ds['test']
    if finetune == 0: 
        model.evaluate(test_ds)
    elif finetune == 1: 
        model.finetune(ds)
        model.evaluate(test_ds)
    elif finetune == 2: 
        train_val_ds = make_nllb_dataset(l1=l1, l2=l2, split=split)
        model.finetune(train_val_ds)
        model.evaluate(test_ds)
    elif finetune == 3: 
        train_val_ds = make_nllb_dataset(l1=l1, l2=l2, split=split)
        model.finetune(train_val_ds)
        model.finetune(ds)
        model.evaluate(test_ds)

if __name__ == '__main__': 
    main()