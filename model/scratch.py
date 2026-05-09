from process_data import make_semeval_dataset, make_nllb_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(example):
    smoothing = SmoothingFunction()

    bleu = sentence_bleu(
        [alt.split(' ') for alt in example['alts']],
        example['input'].split(' '),
        smoothing_function = smoothing.method1,
    ) 

    return {'BLEU': bleu}

def main(): 
    print('Avg BLEU for')
    for pair in [('fr', 'en'), ('nl', 'en'), ('en', 'es'), ('en', 'de')]: 
        ds = make_semeval_dataset(l1=pair[0], l2=pair[1], do_split=True)['test']
        ds = ds.map(calculate_bleu)
        # print(ds)
        # print(ds[0])
        df = ds.to_pandas()
        averages = df[['BLEU']].mean()
        print(f'\t{pair[0]}-{pair[1]}: {averages.loc['BLEU']}')


if __name__ == "__main__": 
    main()