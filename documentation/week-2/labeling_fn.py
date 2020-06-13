from snorkel.labeling import LabelingFunction
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling import labeling_function
from snorkel.labeling.model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter

from sklearn.model_selection import train_test_split
from textblob import TextBlob

import re
import pandas as pd
import numpy as np

# import data
df = pd.read_csv("../week1-data/merged_data.csv",
                 prefix=None).sample(frac=1, random_state=123).reset_index(drop=True)
df.columns = map(str.lower, df.columns)

# TODO write a function that properly handles missing abstracts. Talk to Jennifer about
# potential options before writting it
df.fillna(value="NA", inplace=True)

# I'm sure there is a better way of splitting the data, but I'll use this for now
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=123, stratify=df.relevancy)

# set the label values
RELEVANT = 1
IRRELEVANT = 0
ABSTAIN = -1

# extract the relevancy vector for ease of use later
Y_test = df_test.relevancy.values

def encode_labels(label):
    return RELEVANT if label == "relevant" else IRRELEVANT

Y_test = np.asarray(list(map(encode_labels, Y_test)), dtype=np.intc)

# Simple LFs
@labeling_function()
def vehicle_detection(data_point):
    # return a label of relevant if "vehicle detection" in abstract
    if data_point.abstract is (None or "NA"):
        return RELEVANT if r"vehicle detection" in data_point.papertitle.lower() else ABSTAIN
    return RELEVANT if re.search(r"vehicle.*detect", data_point.abstract.lower(), flags=re.I) else ABSTAIN


@labeling_function()
def human_detection(data_point):
    # human detection
    return RELEVANT if r"human detection" in data_point.abstract.lower() else ABSTAIN


# Keyword-based LFs
def keyword_lookup(data_point, keywords, label):
    if any(word in data_point.abstract.lower() for word in keywords):
        return label
    return ABSTAIN


def make_keyword_lf(keywords, label=RELEVANT):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),)


keyword_recognition = make_keyword_lf(
    keywords=["recognition", "vehicle recognition", "vehicle identification"], label=RELEVANT)
keyword_driving_system = make_keyword_lf(
    keywords=["driving system"])
keyword_autonomous_vehicle = make_keyword_lf(
    keywords=["autonomous vehicle"])

# Heuristic functions
@labeling_function()
def heuristics_lookup(data_point):
    # Short or missing abstract with not relevant keywords in paper title
    return IRRELEVANT if ((len(data_point.abstract.lower()) < 10) and r"detection" not in data_point.papertitle.lower()) else ABSTAIN


@labeling_function()
def heuristics_lang(data_point):
    # Abstact is not written in English*
    # This LF will need some review, since it could have the right keywords regardless of if
    # it was written in English or not
    if (len(data_point.abstract.lower()) != "na" and data_point.abstract.lower()):
        blob = TextBlob(data_point.abstract.lower())
        return IRRELEVANT if ((blob.detect_language() is None) or (blob.detect_language() != "en")) else ABSTAIN
    else:
        ABSTAIN


# Apply LFs
lfns = [
    vehicle_detection,
    human_detection,
    keyword_recognition,
    keyword_driving_system,
    keyword_autonomous_vehicle,
    heuristics_lookup
    # heuristics_lang
]

applier = PandasLFApplier(lfs=lfns)
L_train = applier.apply(df=df_train)
L_test = applier.apply(df=df_test)

# coverage information
lf_analysis_train = LFAnalysis(L=L_train, lfs=lfns).lf_summary()
print(lf_analysis_train)


# Build noise aware majority model
majority_model = MajorityLabelVoter()
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
majority_acc = majority_model.score(
    L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")
label_model_acc = label_model.score(
    L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
