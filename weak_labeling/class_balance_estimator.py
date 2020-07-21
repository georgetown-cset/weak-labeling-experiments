import csv
import json
import re
import statistics

from google.cloud import bigquery


def format_keyword(kw: str):
    words = kw.strip().split()
    return r"(?i)"+r"\S*\s+".join(words)


def get_cat_id_to_keywords(kw_path: str):
    cat_id_to_keywords = {}
    for line in csv.DictReader(open(kw_path)):
        cat_id_to_keywords[line["Id"]] = [format_keyword(k) for k in line["ImprovedLikelyPhrases"].split(",")]
    return cat_id_to_keywords


def estimate_class_balance():
    client = bigquery.Client(project="***REMOVED***")
    cat_to_keywords = get_cat_id_to_keywords("../data/keyword_categories_custom.csv")
    print(cat_to_keywords)
    cat_counts, kw_counts = {}, {}
    for cat, kws in cat_to_keywords.items():
        cat_counts[cat] = 0
        for kw in kws:
            kw_counts[kw] = 0
    corpus_size = 0

    for record in client.list_rows("cn498_sandbox.mag_subset_for_snorkel_eval"):
        if corpus_size % 1000 == 0:
            print("on "+str(corpus_size))
        corpus_size += 1
        text = record["papertitle"]+" "+record["abstract"]
        for cat in cat_to_keywords:
            has_match = False
            for kw in cat_to_keywords[cat]:
                if re.search(kw, text):
                    kw_counts[kw] += 1
                    has_match = True
            if has_match:
                cat_counts[cat] += 1

    # write out the frequency with which *any* of a category's kws match
    with open('cat_total_freq.json', 'w') as out:
        cat_to_total_freq = {cat: hits/corpus_size for cat, hits in cat_counts.items()}
        json.dump(cat_to_total_freq, out)

    # write out the median frequency of match over all of a category's kws
    with open('cat_median_freq.json', 'w') as out:
        cat_to_median_freq = {}
        for cat in cat_to_keywords:
            rel_kw_freqs = [kw_counts[kw]/corpus_size for kw in cat_to_keywords[cat]]
            cat_to_median_freq[cat] = statistics.median(rel_kw_freqs)
        json.dump(cat_to_median_freq, out)


if __name__ == "__main__":
    estimate_class_balance()
