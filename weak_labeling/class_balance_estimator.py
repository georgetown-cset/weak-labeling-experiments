from lf_generator import data_import, create_wordlist_set
from tqdm import tqdm
import re
import pandas as pd
from typing import List
import json


def estimate_class_balance(word_list_set: set, kw_dict: List[dict], train_data: pd.DataFrame):
    # This is but a proof of concept. I'm sure there are better ways of doing this
    train_data = train_data[1000:]
    paper_titles = " ".join(train_data.papertitle)
    paper_abstracts = " ".join(train_data.abstract)
    # Is there a better way of doing this? Not exactly sure this could be done in BQ
    data = f"{paper_titles} {paper_abstracts}"
    corpus_size = train_data.shape[0]

    match_values = {}
    for i in tqdm(word_list_set):
        # TODO: use the proper regular expression. For now we just use the keywords as is
        matches = len(re.findall(fr"(?i){i}", data))
        match_values[i] = float(matches/corpus_size)

    wipo_class_balance_avg = {}
    for i, keyword in enumerate(kw_dict):
        wordlist = keyword["ImprovedLikelyPhrases"].split(sep=", ")
        average_freq = (sum([match_values.get(key) for key in wordlist]))/len(wordlist)
        wipo_class_balance_avg[keyword["Id"]] = average_freq

    with open('result.json', 'w') as out:
        json.dump(wipo_class_balance_avg, out)


def main():
    df_train, df_keywords, keywords_dict = data_import()
    likely_phrases = df_keywords.ImprovedLikelyPhrases
    word_list_set = create_wordlist_set(likely_phrases)
    estimate_class_balance(word_list_set, keywords_dict, df_train)


if __name__ == "__main__":
    main()
