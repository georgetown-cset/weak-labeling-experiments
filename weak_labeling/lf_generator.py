import re
from typing import List

import pandas as pd
from snorkel.labeling import LabelingFunction, LFAnalysis, PandasLFApplier
from textblob import TextBlob
from tqdm import tqdm

ABSTAIN = -1
RELEVANT = 1
IRRELEVANT = 0


def data_import():
    df_keywords = pd.read_csv("../data/keyword_categories.csv", prefix=None)
    df_train = pd.read_csv("../data/training_data.csv", prefix=None)\
        .drop_duplicates(subset="paperid", keep="last")\
        .sample(frac=1, random_state=123)\
        .reset_index(drop=True)
    dict_keywords = df_keywords.to_dict(orient="record")
    return [df_train, df_keywords, dict_keywords]


def keyword_lookup(data_point, keywords: list, label: int):
    if any(word in f"{data_point.papertitle.lower()} {data_point.abstract.lower()}" for word in keywords):
        return label
    return ABSTAIN


def make_keyword_lf(keywords: list, label: int = RELEVANT, lf_name: str = None):
    f_name = lf_name if lf_name is not None else f"keyword_{keywords[0]}"
    return LabelingFunction(
        name=f_name,
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label))


def correct_likely_phrase_column(phrases: str):
    all_likely_phrases = []
    for i in phrases:
        word_list = i.split(sep=",")
        all_likely_phrases.extend(word_list)
    wl = [TextBlob(w.strip().lower()).correct() for w in tqdm(set(all_likely_phrases))]
    all_phrases = map(lambda w: ''.join(w), wl)
    return set(all_phrases)


def generate_lfs(word_list: set, use_categories: bool = False, keywords_dict: List[dict] = None, wipo_id: str = None):
    lfs = []
    if(use_categories and keywords_dict is not None):
        for keyword in keywords_dict:
            wordlist = keyword["LikelyPhrases"].split(sep=",")
            # TODO: correct spelling errors first
            # wordlist = [TextBlob(w.strip().lower()).correct() for w in set(wl)]
            # wordlist = map(lambda w: ''.join(w), wordlist)
            lf = make_keyword_lf(keywords=wordlist, lf_name=keyword["Id"])
            lfs.append(lf)
    else:
        lfs = [make_keyword_lf(keywords=[word]) for word in word_list]
    return lfs


if __name__ == "__main__":
    df_train, df_keywords, keywords_dict = data_import()
    likely_phrases = df_keywords.LikelyPhrases
    word_list_set = correct_likely_phrase_column(likely_phrases)
    lfs = generate_lfs(word_list=word_list_set, use_categories=True, keywords_dict=keywords_dict)

    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)

    lf_analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()

    print(lf_analysis.sort_values('Coverage'))
