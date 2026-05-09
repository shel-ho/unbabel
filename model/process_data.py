import torch

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from lxml import etree
from typing import List

from utils import flores_lang_id, valid_pairs

torch.manual_seed(6)


class SentencePair:
    def __init__(
        self,
        input: str,
        ref: str,
        alts: List[str] = None,
    ):
        self.input = input
        self.ref = ref
        self.alts = alts

    def __str__(self):
        return f"Input: {self.input}\nRef: {self.ref}\nAlts: {self.alts}\n"


def get_right_text(subnode) -> str:
    """
    Extracts text to the right of the L1/translated L2 fragment

    :param subnode: XML subnode containing L1/translated L2 fragment
    :return: text to the right of the L1/translated L2 fragment
    :rtype: str
    """
    # Handles cases when fragment ends the sentence
    if not subnode[0].tail:
        right = ""
    else:
        right = subnode[0].tail.strip()
    return right


def get_left_text(subnode) -> str:
    # Handles cases when L1 text starts the sentence
    if not subnode.text:
        left = ""
    else:
        left = subnode.text.strip()
    return left


# def load_data(l1: str, l2: str, train: bool = False):
#     sent_pairs = []
#     alt_sent_pairs = []
#     fn = f"./data/{l1}-{l2}.gold.untokenised.xml"
#     with open(fn, "rb") as corpus:
#         parser = etree.iterparse(corpus)
#         for _, node in parser:
#             input = ""
#             ref = ""
#             alts = []
#             if node.tag == "s":
#                 for subnode in node:
#                     left = get_left_text(subnode)

#                     if subnode.tag == "input":
#                         input = " ".join(
#                             [left, subnode[0].text.strip(), get_right_text(subnode)]
#                         )
#                     if subnode.tag == "ref":
#                         ref = " ".join(
#                             [left, subnode[0].text.strip(), get_right_text(subnode)]
#                         )
#                         alts.append(ref)
#                         if train:
#                             alt_sent_pairs.append(SentencePair(input, ref))
#                         if len(subnode[0]) > 0:
#                             for subsubnode in subnode[0]:
#                                 alt = " ".join(
#                                     [
#                                         left,
#                                         subsubnode.text.strip(),
#                                         get_right_text(subnode),
#                                     ]
#                                 )
#                                 if train:
#                                     alt_sent_pairs.append(SentencePair(input, alt))
#                                 alts.append(alt)
#                 pair = SentencePair(input, ref, alts)
#                 sent_pairs.append(pair)
#     return sent_pairs, alt_sent_pairs


def data_gen(l1: str, l2: str, split: bool):
    """
    Generator that parses xml file and yields dicts to make a Dataset object.
    Each alternative translation is an individual datapoint

    :param l1: L1 language
    :type l1: str
    :param l2: L2 language
    :type l2: str
    """
    fn = f"./data/{l1}-{l2}.gold.untokenised.xml"
    with open(fn, "rb") as corpus:
        parser = etree.iterparse(corpus)
        for _, node in parser:
            input = ""
            ref = ""
            golds = []
            if node.tag == "s":
                for subnode in node:
                    left = get_left_text(subnode)
                    if subnode.tag == "input":
                        input = " ".join(
                            [left, subnode[0].text.strip(), get_right_text(subnode)]
                        )
                    if subnode.tag == "ref":
                        ref = " ".join(
                            [left, subnode[0].text.strip(), get_right_text(subnode)]
                        )
                        golds.append(ref)
                        if len(subnode[0]) > 0:
                            for subsubnode in subnode[0]:
                                alt = " ".join(
                                    [
                                        left,
                                        subsubnode.text.strip(),
                                        get_right_text(subnode),
                                    ]
                                )
                                golds.append(alt)
                if split:
                    for gold in golds:
                        yield {"input": input, "ref": gold, "alts": golds}
                else:
                    yield {"input": input, "ref": ref, "alts": golds}


def langs_map(example, l1, l2):
    return {
        "input": example["input"],
        "ref": example["ref"],
        "alts": example["alts"],
        "l1": l1,
        "l2": l2,
    }


def make_semeval_dataset(l1: str, l2: str, do_split: bool):
    # We only need to check l1 because it isn't possible to get all with anything else 
    # bc of validation in main
    if l1 == "all":
        dsets = [make_semeval_dataset(l1, l2, do_split) for l1, l2 in valid_pairs]
        return DatasetDict(
            {
                split: concatenate_datasets([dset[split] for dset in dsets])
                for split in dsets[0]
            }
        ).shuffle(seed=6)
    else:
        ds = Dataset.from_generator(
            data_gen, gen_kwargs={"l1": l1, "l2": l2, "split": do_split}
        )
        ds = ds.map(langs_map, fn_kwargs={"l1": l1, "l2": l2})
        ds = ds.shuffle(seed=6)
        if not do_split:
            return ds
        train_test_ds = ds.train_test_split(test_size=0.2, shuffle=False)
        test_val_ds = train_test_ds["test"].train_test_split(
            test_size=0.5, shuffle=False
        )
        return DatasetDict(
            {
                "train": train_test_ds["train"],
                "val": test_val_ds["train"],
                "test": test_val_ds["test"],
            }
        )


def make_nllb_dataset(l1: str, l2: str, split: int):
    """
    Makes dataset from subset of NLLB dataset. Because we are fine-tuning, we do not need millions of
    datapoints. So we consider a full training dataset to be 100k sentences and set aside 1k sentences
    for validation.
    """
    if l1 == "all":
        dsets = [make_nllb_dataset(l1, l2, split) for l1, l2 in valid_pairs]
        return DatasetDict(
            {
                split: concatenate_datasets([dset[split] for dset in dsets])
                for split in dsets[0]
            }
        ).shuffle(seed=6)
    else:
        source_lang = flores_lang_id[l1]
        target_lang = flores_lang_id[l2]
        # Load dataset expects a certain ordering for each language pair
        try:
            iterable_ds = load_dataset(
                "allenai/nllb",
                f"{source_lang}-{target_lang}",
                streaming=True,
                trust_remote_code=True,
            )
        except:
            iterable_ds = load_dataset(
                "allenai/nllb",
                f"{target_lang}-{source_lang}",
                streaming=True,
                trust_remote_code=True,
            )
        iterable_ds = iterable_ds.map(
            translation_map,
            fn_kwargs={"source_lang": source_lang, 
                       "target_lang": target_lang,
                       "l1": l1,
                       "l2": l2},
            remove_columns=["translation", "laser_score"],
        )
        iterable_ds = iterable_ds["train"].take((split + 1) * 1000)
        list_ds = list(iterable_ds)
        return DatasetDict(
            {
                "train": Dataset.from_list(list_ds[:-1000]),
                "val": Dataset.from_list(list_ds[-1000:]),
            }
        )


def translation_map(example, source_lang, target_lang, l1, l2):
    return {
        "input": example["translation"][source_lang],
        "ref": example["translation"][target_lang],
        "alts": [example["translation"][target_lang]],
        "l1": l1,
        "l2": l2,
    }


if __name__ == "__main__":
    ds = make_nllb_dataset(l1="nl", l2="en", split=1)
    print(ds["train"][0])
