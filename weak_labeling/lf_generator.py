import re
from datetime import datetime
from multiprocessing import Pool
from typing import List

import pandas as pd
from snorkel.labeling import LabelingFunction, LFAnalysis, PandasLFApplier
from snorkel.labeling.model import LabelModel
from textblob import TextBlob
from tqdm import tqdm

ABSTAIN = -1
RELEVANT = 1
IRRELEVANT = 0


def data_import():
    df_keywords = pd.read_csv("../data/keyword_categories_custom.csv", prefix=None)
    df_train = pd.read_csv("../data/wipo_training_data.csv", prefix=None)\
        .drop_duplicates(subset="paperid", keep="last")\
        .sample(frac=1, random_state=123)\
        .reset_index(drop=True)
    dict_keywords = df_keywords.to_dict(orient="record")
    return [df_train, df_keywords, dict_keywords]


def keyword_lookup(data_point, keywords: list, label: int):
    current_data_point = f"{data_point.papertitle.lower()} {data_point.abstract.lower()}"
    # if any(word in current_data_point for word in keywords):
    #     return label
    # else:
    for word in keywords:
        if (len(word.split(sep=" ")) > 2):
            word_regex = word.strip().replace(" ", ".*")
            return label if re.search(fr"(?i){word_regex}", current_data_point) else ABSTAIN
        elif any(word in current_data_point for word in keywords):
            return label
        else:
            return ABSTAIN
    return ABSTAIN


def make_keyword_lf(keywords: list, label: int = RELEVANT, lf_name: str = None):
    labeling_function_name = f"keyword_{re.sub(' ', '_', keywords[0].strip())}"
    return LabelingFunction(
        name=labeling_function_name,
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label))


def create_wordlist_set(phrases: str):
    all_likely_phrases = []
    for i in phrases:
        word_list = i.split(sep=", ")
        all_likely_phrases.extend(word_list)
    # wl = [TextBlob(w.strip().lower()).correct() for w in tqdm(set(all_likely_phrases))]
    # all_phrases = map(lambda w: ''.join(w), wl)
    return set(all_likely_phrases)


def generate_lfs(word_list: set, use_categories: bool = True, keywords_dict: List[dict] = None, wipo_id: str = None):
    wipo_lfs = {}
    if(use_categories and keywords_dict is not None):
        for i, keyword in enumerate(keywords_dict):
            label = RELEVANT  # if keyword["Id"] == wipo_id else 0
            wordlist = keyword["ImprovedLikelyPhrases"].split(sep=", ")
            negative_wordlist = list(set(keyword["LikelyPhrases"].split(sep=", ")) - set(wordlist))
            positive_keyword_lfs = [make_keyword_lf(keywords=[wl], label=RELEVANT) for wl in wordlist]
            negative_keyword_lfs = [make_keyword_lf(keywords=[wl], label=IRRELEVANT) for wl in negative_wordlist]
            wipo_lfs[keyword["Id"]] = [*positive_keyword_lfs, *negative_keyword_lfs]
    else:
        lfs = [make_keyword_lf(keywords=[word]) for word in tqdm(word_list)]
        # return lfs
    return wipo_lfs


def train_model(df_train: pd.DataFrame, application_area_lfs: list, output_file_name: str = "output"):
    file_name_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    applier = PandasLFApplier(lfs=application_area_lfs)
    L_train = applier.apply(df=df_train)
    model = LabelModel(cardinality=2, verbose=True)
    model.fit(L_train=L_train, n_epochs=800, log_freq=100)
    model_predict = model.predict(L=L_train, return_probs=True, tie_break_policy="abstain")
    lf_analysis = LFAnalysis(L=L_train, lfs=application_area_lfs).lf_summary()
    with open(f"out/{output_file_name}_{file_name_timestamp}.csv", "w") as outfile:
        lf_analysis = lf_analysis.sort_values("Coverage")
        lf_analysis.to_csv(outfile, encoding="utf-8", index=True)


if __name__ == "__main__":
    df_train, df_keywords, keywords_dict = data_import()
    likely_phrases = df_keywords.ImprovedLikelyPhrases
    word_list_set = create_wordlist_set(likely_phrases)
    lfs = generate_lfs(word_list=word_list_set, use_categories=True, keywords_dict=keywords_dict)

    pool = Pool()
    for key, application_area_lfs in lfs.items():
        pool.starmap(train_model, [(df_train, application_area_lfs, key)])

    pool.close()
    pool.join()
