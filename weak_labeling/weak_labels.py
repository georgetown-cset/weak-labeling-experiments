# import dependencies
import argparse
import datetime
import json
from typing import List
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from snorkel.analysis import get_label_buckets, metric_score
from snorkel.labeling import (LabelingFunction, LFAnalysis, PandasLFApplier,
                              filter_unlabeled_dataframe, labeling_function)
from snorkel.labeling.model import LabelModel, MajorityLabelVoter

from data_import import get_training_data

'''
This script is used to get training data from Big Query and train a Snorkel Label Model using the data.
'''
# set the label values
RELEVANT = 1
IRRELEVANT = 0
ABSTAIN = -1


def prepare_data(gold_label_path: str, training_data_path: str = "../data/training_data.csv", local: bool = True) -> List[pd.DataFrame]:
    """Prepare the training data as well as the test data (gold set) to generate the label model

    :param gold_label_path: Path to the gold labels
    :type gold_label_path: str
    :param training_data_path: Path to the training data used for the model, defaults to "../data/training_data.csv"
    :type training_data_path: str, optional
    :param local: Determines where the training data should be retrieved from. If true, it retrieves the data from
        the local disk. If false, the data will be obtained from Big Query. You will need access to the Big Query Instance
        to query the data. Defaults to True
    :type local: bool, optional
    :return: A list containing the training dataset and the gold labels as panda dataframes
    :rtype: List[pd.DataFrame]
    """
    df_test = pd.read_csv(gold_label_path, prefix=None)\
        .drop_duplicates(subset="PaperId", keep="last")\
        .sample(frac=1, random_state=123)\
        .reset_index(drop=True)

    df_test.columns = map(str.lower, df_test.columns)

    # remove all data points from the test set with that do not have an abstract
    df_test = df_test[df_test.abstract.notnull()]

    # import training data using helper function
    df_train = get_training_data(training_data_path=training_data_path, from_local_disk=local)\
        .sample(frac=1, random_state=123)\
        .drop_duplicates(subset="paperid", keep="last")\
        .reset_index(drop=True)

    # drop duplicate data points in the training and testing sets
    cond_drop_duplicates = df_train['paperid'].isin(df_test['paperid'])
    df_train.drop(df_train[cond_drop_duplicates].index, inplace=True)
    return [df_train, df_test]


def encode_labels(label: str):
    """Encode the labels from the gold labels to integers

    :param label: The handlabled label of the datapoint 
    :type label: str
    :return: An encoded label 
    :rtype: int
    """
    return RELEVANT if label == "relevant" else IRRELEVANT


def keyword_lookup(data_point: str, keywords: list, label: int):
    """The keyword lookup function is the base of the labeling function. It takes a list of keywords,
        and checks that list of keywords against a data point. If the any of the keywords are in the
        data point it returns the label passed into the function. Otherwise, it abstains (returns -1)

    :param data_point: The data point is the paper title and paper abstract concatenated together
    :type data_point: str
    :param keywords: A list of keywords related to the application area
    :type keywords: list
    :param label: A numerical label to assign to the data point if any of the keywords match
    :type label: int
    :return: Return the label if any of the keywords match or abstain if non matched
    :rtype: [type]
    """
    if any(word in f"{data_point.papertitle.lower()} {data_point.abstract.lower()}" for word in keywords):
        return label
    return ABSTAIN


def make_keyword_lf(keywords: list, label: int = RELEVANT):
    """Generate labeling functions from keywords related to the application area

    :param keywords: A list of keywords related to the application area
    :type keywords: list
    :param label: The label that should be assigned to each labeling function, defaults to RELEVANT
    :type label: int, optional
    :return: Returns a labeling function which implements the `keyword_lookup` function
    :rtype: LabelingFunction
    """
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label))


def LF_applier(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Create the labling functions and apply those labeling functions on the data points

    :param df_train: The training dataset
    :type df_train: pd.DataFrame
    :param df_test: The gold labels
    :type df_test: pd.DataFrame
    :return: Return the matrix of labels emitted by the labeling functions
    :rtype: List[np.ndarray, np.ndarray, List[LabelingFunction]]
    """
    # Make keywords
    keyword_vehicle_detection = make_keyword_lf(keywords=["vehicle detection", "vehicle detector"])
    keyword_driver_identification = make_keyword_lf(keywords=["driver identification", "driver identifier"])
    keyword_human_detection = make_keyword_lf(keywords=["human detection", "human detector"])
    keyword_license_info = make_keyword_lf(keywords=["license plate", "license number"])
    keyword_vehicle_recognition = make_keyword_lf(keywords=["vehicle recognition", "vehicle identification"])
    keyword_driving_system = make_keyword_lf(keywords=["driving system"])
    keyword_autonomous_vehicle = make_keyword_lf(keywords=["autonomous vehicle"], label=IRRELEVANT)
    keyword_driverless_vehicle = make_keyword_lf(keywords=["driverless cars", "driverless vehicle", "unmanned vehicle"], label=IRRELEVANT)
    keyword_lidar = make_keyword_lf(keywords=["lidar", "laser detection"], label=IRRELEVANT)
    keyword_radar = make_keyword_lf(keywords=["radar", "vehicle radar"], label=IRRELEVANT)
    keyword_computer_vision = make_keyword_lf(keywords=["computer vision", "opencv"], label=IRRELEVANT)

    # Apply LFs
    lfns = [
        keyword_vehicle_detection,
        keyword_human_detection,
        keyword_driver_identification,
        keyword_vehicle_recognition,
        keyword_driving_system,
        keyword_autonomous_vehicle,
        keyword_driverless_vehicle,
        keyword_lidar,
        keyword_radar,
        keyword_computer_vision
    ]

    applier = PandasLFApplier(lfs=lfns)
    apply_train_time_start = time()
    L_train = applier.apply(df=df_train)
    apply_train_time_end = time()
    print(f"LF Application Time: {apply_train_time_end - apply_train_time_start} seconds")
    L_test = applier.apply(df=df_test)
    return [L_train, L_test, lfns]


def model_analysis(label_model: LabelModel, training_set: pd.DataFrame, L_train: np.ndarray, L_test: np.ndarray, Y_test: np.ndarray, lfs: list, output_file="output") -> None:
    # TODO: consider using **kwargs instead of this painful list of arguments
    """Output analysis for the label model to a file

    :param label_model: The current label model which we want to output analysis for
    :type label_model: LabelModel
    :param training_set: A dataframe containing the training dataset
    :type training_set: pd.DataFrame
    :param L_train: The matrix of labels generated by the labeling functions on the training data
    :type L_train: np.ndarray
    :param L_test: The matrix of labels generated bt the labeling functions on the testing data
    :type L_test: np.ndarray
    :param Y_test: Gold labels associated with data points in L_test
    :type Y_test: np.ndarray
    :param lfs: List of labeling functions
    :type lfs: list
    :param output_file: A path where the output file should be writtent to, defaults to `PROJECT_ROOT/output`
    :type output_file: str, optional
    """
    Y_train = label_model.predict_proba(L=L_train)
    Y_pred = label_model.predict(L=L_test, tie_break_policy="abstain")
    lf_analysis_train = LFAnalysis(L=L_train, lfs=lfs).lf_summary()

    # TODO: Write this df to a output file. Ask Jennifer about how to handle this
    print(lf_analysis_train)

    # build majority label voter model
    majority_model = MajorityLabelVoter()
    majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="abstain", metrics=["f1", "accuracy"])
    label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="abstain", metrics=["f1", "accuracy"])

    # get precision and recall scores
    p_score = precision_score(y_true=Y_test, y_pred=Y_pred, average='weighted')
    r_score = recall_score(y_true=Y_test, y_pred=Y_pred, average='weighted', labels=np.unique(Y_pred))

    # how many documents abstained
    probs_train = majority_model.predict_proba(L=L_train)
    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(X=training_set, y=probs_train, L=L_train)

    # get number of false positives
    buckets = get_label_buckets(Y_test, Y_pred)
    true_positives, false_positives, true_negatives, false_negatives = (
        buckets.get((1, 1)), buckets.get((1, 0)), buckets.get((0, 0)), buckets.get((0, 1)))
    # write analysis to file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with open(f"{'../output/logs/'}{output_file}_run_{timestamp}.txt", "w") as output_file:
        output_file.write(f"{'Majority Vote Accuracy:':<25} {majority_acc['accuracy'] * 100:.2f}%")
        output_file.write(f"\n{'Majority Vote F1 Score:':<25} {majority_acc['f1'] * 100:.2f}%")
        output_file.write(f"\n{'Label Model Accuracy:':<25} {label_model_acc['accuracy'] * 100:.2f}%")
        output_file.write(f"\n{'Label Model F1 Score:':<25} {label_model_acc['f1'] * 100:.2f}%")
        output_file.write(f"\n{'Precision Score:':<25} {p_score * 100:.2f}%")
        output_file.write(f"\n{'Recall Score:':<25} {r_score * 100:.2f}%")
        output_file.write(f"\n{'Abstained Data Points:':<25} {len(df_train_filtered)}")
        output_file.write(f"\n{'True Positives:':<25} {len(true_positives) if true_positives is not None else 0}")
        output_file.write(f"\n{'False Positives:':<25} {len(false_positives) if false_positives is not None else 0}")
        output_file.write(f"\n{'False Negatives:':<25} {len(false_negatives) if false_negatives is not None else 0}")
        output_file.write(f"\n{'True Negatives:':<25} {len(true_negatives) if true_negatives is not None else 0}")
        output_file.write(f"\n{'Abstained Positives:':<25} {len(buckets[(1, -1)])}")
        output_file.write(f"\n{'Abstained Negatives:':<25} {len(buckets[(0, -1)])}")


def train_model(training_data: pd.DataFrame, testing_data: pd.DataFrame, L_train: np.ndarray, save_model=True) -> LabelModel:
    """Train a label model using the label matrix generated by the labeling functions

    :param training_data: Dataframe of training data
    :type training_data: pd.DataFrame
    :param testing_data: Dataframe of testing data
    :type testing_data: pd.DataFrame
    :param L_train: The matrix of labels generated by the labeling functions on the training data
    :type L_train: np.ndarray
    :param save_model: Set this to `True` to save the model to disk, defaults to `True`
    :type save_model: bool, optional
    :return: A label model
    :rtype: LabelModel
    """
    # Build noise aware majority model
    model = LabelModel(cardinality=2, verbose=True)
    model.fit(L_train=L_train, n_epochs=800, log_freq=100)  # , class_balance=[0.673, 0.327])
    if(save_model):
        model.save("../output/model_export/saved_label_model.pkl")
    return model


def main(output_path: str, training_data: str, gold_labels: str, label_output_path: str) -> None:
    df_train, df_test = prepare_data(gold_label_path=gold_labels)
    L_train, L_test, lfns = LF_applier(df_train, df_test)
    Y_test = np.asarray(list(map(encode_labels, df_test["relevancy"].tolist())), dtype=np.intc)

    # Build noise aware majority model
    begin_train_time = time()
    label_model = train_model(training_data=df_train, testing_data=df_test, L_train=L_train, save_model=True)
    end_train_time = time()

    print(f"Training time: {end_train_time - begin_train_time}")
    model_analysis(label_model=label_model, training_set=df_train, L_train=L_train, L_test=L_test, Y_test=Y_test, lfs=lfns, output_file=output_path)
    # get both integer and probability labels for data, filtering out unlabeled data points: https://www.snorkel.org/use-cases/01-spam-tutorial#filtering-out-unlabeled-data-points
    int_labels, prob_labels = label_model.predict(L=L_train, return_probs=True)
    probs_df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
        X=df_train, y=prob_labels, L=L_train
    )
    int_df_train_filtered, int_train_filtered = filter_unlabeled_dataframe(
        X=df_train, y=int_labels, L=L_train
    )
    # write out both labels. In the probability outputs, p_rel is the second probability listed
    assert list(probs_df_train_filtered["paperid"]) == list(int_df_train_filtered["paperid"])
    with open(label_output_path, mode="w") as out:
        for idx, paper_id in enumerate(probs_df_train_filtered["paperid"]):
            out.write(json.dumps({
                "id": paper_id,
                # cast to int and float to get rid of nonserializable numpy types
                "is_rel": int(int_train_filtered[idx]),
                "p_rel": float(probs_train_filtered[idx][1])
            })+"\n")


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output file for model analytics")
    parser.add_argument("--data", help="path to training dataset")
    parser.add_argument("--golds", help="path to gold labels")
    parser.add_argument("--label_out", help="path to file where integer outputs should be written", default="labels.jsonl")

    # parse arguments
    args = parser.parse_args()
    golds = args.golds if args.golds is not None else "../data/testing_data.csv"
    output_file_name = args.output if args.output is not None else "output"

    main(output_path=output_file_name, training_data=args.data, gold_labels=golds, label_output_path=args.label_out)
