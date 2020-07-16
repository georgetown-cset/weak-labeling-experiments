import argparse
import json
import re
from datetime import datetime
from multiprocessing import Pool
from typing import List

import pandas as pd
from snorkel.labeling import (LabelingFunction, LFAnalysis, PandasLFApplier,
                              filter_unlabeled_dataframe)
from snorkel.labeling.model import LabelModel

ABSTAIN = -1
RELEVANT = 1
IRRELEVANT = 0


def data_import(keywords_category_path: str = "../data/keyword_categories_custom.csv", wipo_training_data_path: str = "../data/wipo_training_data.csv"):
    df_keywords = pd.read_csv(keywords_category_path, prefix=None)
    df_train = pd.read_csv(wipo_training_data_path, prefix=None)\
        .drop_duplicates(subset="paperid", keep="last")\
        .sample(frac=1, random_state=123)\
        .reset_index(drop=True)
    dict_keywords = df_keywords.to_dict(orient="record")
    return [df_train, df_keywords, dict_keywords]


def keyword_lookup(data_point, keywords: list, label: int):
    current_data_point = f"{data_point.papertitle.lower()} {data_point.abstract.lower()}"
    for word in keywords:
        if (len(word.split(sep=" ")) > 2):
            w = word.strip().split(sep=" ")
            word_regex = rf"({w[0]})(\W*\w*){0,2}{w[1]}"
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


def create_wordlist_set(phrases: list):
    all_likely_phrases = []
    for phrase in phrases:
        word_list = phrase.split(sep=", ")
        all_likely_phrases.extend(word_list)
    return set(all_likely_phrases)


def generate_lfs(word_list: set, keywords_dict: List[dict] = None, wipo_id: str = None):
    lfs = {}
    if(wipo_id):
        keywords_dict = list(filter(lambda kd: kd["Id"] == wipo_id, keywords_dict))

    for i, keyword in enumerate(keywords_dict):
        wordlist = keyword["ImprovedLikelyPhrases"].split(sep=", ")
        negative_wordlist = list(word_list - set(wordlist))
        positive_keyword_lfs = [make_keyword_lf(keywords=[wl], label=RELEVANT) for wl in wordlist]
        negative_keyword_lfs = [make_keyword_lf(keywords=[wl], label=IRRELEVANT) for wl in negative_wordlist]
        lfs[keyword["Id"]] = [*positive_keyword_lfs, *negative_keyword_lfs]
    return lfs


def train_model(df_train: pd.DataFrame, application_area_lfs: list, analysis_path: str = "output", label_output_path: str = "labels.jsonl", save_model_path: str = None):
    file_name_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    applier = PandasLFApplier(lfs=application_area_lfs)
    L_train = applier.apply(df=df_train)

    model = LabelModel(cardinality=2, verbose=True)
    model.fit(L_train=L_train, n_epochs=800, log_freq=100)
    if(save_model_path is not None):
        model.save(save_model_path)

    int_labels, prob_labels = model.predict(L=L_train, return_probs=True, tie_break_policy="abstain")
    probs_df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
        X=df_train, y=prob_labels, L=L_train)

    int_df_train_filtered, int_train_filtered = filter_unlabeled_dataframe(
        X=df_train, y=int_labels, L=L_train)
    # write out both labels. In the probability outputs, p_rel is the second probability listed
    assert list(probs_df_train_filtered["paperid"]) == list(int_df_train_filtered["paperid"])
    with open(f"{label_output_path}", mode="w") as out:
        for idx, paper_id in enumerate(probs_df_train_filtered["paperid"]):
            out.write(json.dumps({
                "id": paper_id,
                # cast to int and float to get rid of nonserializable numpy types
                "is_rel": int(int_train_filtered[idx]),
                "p_rel": float(probs_train_filtered[idx][1])
            })+"\n")

    # output LF analysis to csv file sorted by coverage
    lf_analysis = LFAnalysis(L=L_train, lfs=application_area_lfs).lf_summary()
    with open(f"../output/{analysis_path}_{file_name_timestamp}.csv", "w") as outfile:
        lf_analysis = lf_analysis.sort_values("Coverage")
        lf_analysis.to_csv(outfile, encoding="utf-8", index=True)


if __name__ == "__main__":
    df_train, df_keywords, keywords_dict = data_import()
    likely_phrases = df_keywords.ImprovedLikelyPhrases
    word_list_set = create_wordlist_set(likely_phrases)
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--wipo_id", help="wipo id for lf generation. will generate lfs for all wipo application areas if left blank")
    parser.add_argument("--label_out", help="path to file where integer outputs should be written")
    parser.add_argument("--save_model", help="path to location for saving the trained model")
    parser.add_argument("--analysis_path", help="path to file where model analysis summary should be written")

    # parse arguments
    args = parser.parse_args()

    lfs = generate_lfs(word_list=word_list_set, keywords_dict=keywords_dict, wipo_id=args.wipo_id)
    pool = Pool()
    pool.starmap(train_model, [(
        df_train,
        application_area_lfs,
        args.analysis_path or key,
        args.label_out,
        args.save_model
    ) for key, application_area_lfs, in lfs.items()])

    pool.close()
    pool.join()
