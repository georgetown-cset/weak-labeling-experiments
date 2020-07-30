import csv
import json
import os
import re
import statistics

from multiprocessing import Pool


def format_keyword(kw: str):
    words = kw.strip().split()
    return r"(?i)"+r"\S*\s+".join(words)


def get_cat_id_to_keywords(kw_path: str):
    cat_id_to_keywords = {}
    for line in csv.DictReader(open(kw_path)):
        cat_id_to_keywords[line["Id"]] = [format_keyword(k) for k in line["ImprovedLikelyPhrases"].split(",")]
    return cat_id_to_keywords


def get_frequencies(fi: str, cat_to_keywords: dict):
    cat_counts, kw_counts = {}, {}
    for cat, kws in cat_to_keywords.items():
        cat_counts[cat] = 0
        for kw in kws:
            kw_counts[kw] = 0
    corpus_size = 0

    for line in open(fi):
        record = json.loads(line)
        corpus_size += 1
        text = record["papertitle"]+" "+record["abstract"]
        cat_to_has_match = {cat: False for cat in cat_counts}
        for kw in kw_counts:
            if re.search(kw, text):
                kw_counts[kw] += 1
                for cat in cat_to_keywords:
                    if kw in cat_to_keywords[cat]:
                        cat_to_has_match[cat] = True
        for cat in cat_to_has_match:
            if cat_to_has_match[cat]:
                cat_counts[cat] += 1
    return {
        "corpus_size": corpus_size,
        "cat_counts": cat_counts,
        "kw_counts": kw_counts
    }


def estimate_class_balance(input_dir: str):
    cat_to_keywords = get_cat_id_to_keywords("../data/keyword_categories_custom.csv")

    with Pool() as pool:
        freq_info = pool.starmap(get_frequencies, [(os.path.join(input_dir, fi), cat_to_keywords) for fi in os.listdir(input_dir)])
   
    # all the elements of freq_info will have the same keys in their kw_counts and cat_counts, so we can just
    # take the first element to get the key sets
    one_elt = freq_info[0]
    corpus_size = sum([e["corpus_size"] for e in freq_info])
    print(f"corpus size was {corpus_size}")
    kw_counts = {kw: sum([e["kw_counts"][kw] for e in freq_info]) for kw in one_elt["kw_counts"]}
    cat_counts = {cat: sum([e["cat_counts"][cat] for e in freq_info]) for cat in one_elt["cat_counts"]}

    # write out the frequency with which *any* of a category"s kws match
    with open("cat_total_freq.json", "w") as out:
        cat_to_total_freq = {cat: hits/corpus_size for cat, hits in cat_counts.items()}
        json.dump(cat_to_total_freq, out)

    # write out the median frequency of match over all of a category"s kws
    with open("cat_median_freq.json", "w") as out:
        cat_to_median_freq = {}
        for cat in cat_to_keywords:
            rel_kw_freqs = [kw_counts[kw]/corpus_size for kw in cat_to_keywords[cat]]
            cat_to_median_freq[cat] = statistics.median(rel_kw_freqs)
        json.dump(cat_to_median_freq, out)

    # write out the average frequency of match over all of a category"s kws
    with open("cat_average_freq.json", "w") as out:
        cat_to_average_freq = {}
        for cat in cat_to_keywords:
            rel_kw_freqs = [kw_counts[kw]/corpus_size for kw in cat_to_keywords[cat]]
            cat_to_average_freq[cat] = sum(rel_kw_freqs)/len(rel_kw_freqs)
        json.dump(cat_to_average_freq, out)



if __name__ == "__main__":
    estimate_class_balance("mag_subset_for_snorkel_eval0721")
