import pandas_gbq as pd_gbq
import pandas as pd
import numpy as np
import re
import os
import data_import
from snorkel.labeling import LabelingFunction
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling import labeling_function
from snorkel.labeling.model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.analysis import get_label_buckets


# import test data
df_test = pd.read_csv("../week1-data/merged_data.csv", prefix=None)\
    .drop_duplicates(subset="PaperId", keep="last")\
    .sample(frac=1, random_state=120)\
    .reset_index(drop=True)
df_test.columns = map(str.lower, df_test.columns)

# import training data
df_train = data_import\
    .get_data(local=True)\
    .sample(frac=1, random_state=120)\
    .drop_duplicates(subset="paperid", keep="last")\
    .reset_index(drop=True)

# drop duplicates in the training and testing sets
cond_drop_duplicates = df_train['paperid'].isin(df_test['paperid'])
df_train.drop(df_train[cond_drop_duplicates].index, inplace=True)

# remove all abstracts from the test set with no abstract
df_test = df_test[df_test.abstract.notnull()]

# set the label values
RELEVANT = 1
IRRELEVANT = 0
ABSTAIN = -1


def encode_labels(label):
    return RELEVANT if label == "relevant" else IRRELEVANT


# extract the relevancy vector for ease of use later
Y_test = np.asarray(
    list(map(encode_labels, df_test["relevancy"].tolist())), dtype=np.intc)


# Keyword-based LFs
def keyword_lookup(data_point, keywords, label):
    if any(word in f"{data_point.papertitle.lower()} {data_point.abstract.lower()}" for word in keywords):
        return label
    return ABSTAIN


def make_keyword_lf(keywords, label=RELEVANT):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),)


# Make keywords
keyword_vehicle = make_keyword_lf(keywords=["vehicle"])
keyword_vehicle_detection = make_keyword_lf(
    keywords=["vehicle detection", "vehicle detector"])
keyword_driver_identification = make_keyword_lf(
    keywords=["driver identification", "driver identifier"])
keyword_human_detection = make_keyword_lf(
    keywords=["human detection", "human detector"])
keyword_license_info = make_keyword_lf(
    keywords=["license plate", "license number"])
keyword_recognition = make_keyword_lf(
    keywords=["recognition", "vehicle recognition", "vehicle identification"])
keyword_driving_system = make_keyword_lf(
    keywords=["driving system"])
keyword_autonomous_vehicle = make_keyword_lf(
    keywords=["autonomous vehicle"], label=IRRELEVANT)
keyword_driverless_vehicle = make_keyword_lf(
    keywords=["driverless cars", "driverless vehicle", "unmanned vehicle"], label=IRRELEVANT)
keyword_lidar = make_keyword_lf(
    keywords=["lidar", "laser detection"], label=IRRELEVANT)
keyword_radar = make_keyword_lf(
    keywords=["radar", "vehicle radar"], label=IRRELEVANT)
keyword_computer_vision = make_keyword_lf(
    keywords=["computer vision", "opencv"], label=IRRELEVANT
)

# Heuristic functions
@labeling_function()
def heuristics_lookup(data_point):
    # Short or missing abstract
    return IRRELEVANT if (len(data_point.abstract.lower()) < 10) else ABSTAIN


# Apply LFs
lfns = [
    keyword_vehicle_detection,
    keyword_human_detection,
    keyword_driver_identification,
    keyword_recognition,
    keyword_driving_system,
    keyword_autonomous_vehicle,
    keyword_driverless_vehicle,
    keyword_lidar,
    keyword_radar,
    keyword_computer_vision,
    heuristics_lookup
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
label_model.fit(L_train=L_train, n_epochs=800, log_freq=100, seed=120)

# predicted labels
Y_pred = label_model.predict(L_test, tie_break_policy="abstain")

majority_acc = majority_model.score(
    L=L_test, Y=Y_test, tie_break_policy="abstain", metrics=["f1", "accuracy"])
label_model_acc = label_model.score(
    L=L_test, Y=Y_test, tie_break_policy="abstain", metrics=["f1", "accuracy"])


print(f"{'Majority Vote Accuracy:':<25} {majority_acc['accuracy'] * 100:.1f}%")
print(f"{'Majority Vote F1 Score:':<25} {majority_acc['f1'] * 100:.1f}%")

print(
    f"\n{'Label Model Accuracy:':<25} {label_model_acc['accuracy'] * 100:.1f}%")
print(f"{'Label Model F1 Score:':<25} {label_model_acc['f1'] * 100:.1f}%")

# how many documents abstained
probs_train = majority_model.predict_proba(L=L_train)
df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)

print(f"\nABSTAIN: {len(df_train_filtered)}")

#print(df_test[df_test[df_test['relevancy'] == "irrelevant"], 'labels'])
buckets = get_label_buckets(Y_test, Y_pred)

true_positives, false_positives, true_negatives, false_negatives = (
    buckets.get((1, 1)), buckets.get((1, 0)), buckets.get((0, 0)), buckets.get((0, 1)))
print("\n")
print(f"true positives: {len(true_positives)}")
print(f"false positives: {false_positives}")
print(f"false negatives: {len(false_negatives)}")
print(f"true_negatives: {len(true_negatives)}")

print("\n")

print(f"abstained positives {len(buckets[(1, -1)])}")
print(f"abstained negatives {len(buckets[(0, -1)])}")
