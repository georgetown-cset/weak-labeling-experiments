import argparse
import json
import re
from datetime import datetime
from multiprocessing import Pool
from typing import List
import os
import pandas as pd
from snorkel.labeling import (LabelingFunction, LFAnalysis, PandasLFApplier, filter_unlabeled_dataframe)
from snorkel.labeling.model import LabelModel

class LFGenerator:
    """LF Generator generates labeling functions and a label model for an application area
        given keywords and training data
    """
    def __init__(self):
        self.RELEVANT = 1
        self.IRRELEVANT = 0
        self.ABSTAIN = -1
        self.PROJECT_ROOT = os.path.abspath(os.pardir)

        # import training data and keywords
        data = self.data_import()
        df_keywords = data.get("keywords")

        self.df_train = data.get("training_data")
        self.keywords_dict = df_keywords.to_dict(orient="record")
        self.likely_phrases = df_keywords.ImprovedLikelyPhrases
        self.word_list_set = self.create_wordlist_set(self.likely_phrases)
        super().__init__()

    def data_import(self, wipo_training_data_path: str = None, keywords_category_path: str = None):
        """Import the training data from a csv file, along with the keywords used in generating the positive and negative labeling functions

        :param wipo_training_data_path: Path to the training dataset, defaults to `PROJECT_ROOT/data/wipo_training_data.csv`
        :type wipo_training_data_path: str, optional
        :param keywords_category_path: Path to the keywords file, defaults to `PROJECT_ROOT/data/keyword_categories_custom.csv`
        :type keywords_category_path: str, optional
        :return: Returns a dictionary containing the two dataframes: the training dataset and the keywords dataframe
        :rtype: Dict[pd.Dataframe]
        """
        keywords_category_path = f"{self.PROJECT_ROOT}/data/keyword_categories_custom.csv" if keywords_category_path is None else keywords_category_path
        wipo_training_data_path = f"{self.PROJECT_ROOT}/data/wipo_training_data.csv" if wipo_training_data_path is None else wipo_training_data_path
        df_keywords = pd.read_csv(keywords_category_path, prefix=None)
        df_train = pd.read_csv(wipo_training_data_path, prefix=None)\
            .drop_duplicates(subset="paperid", keep="last")\
            .reset_index(drop=True)
        return {
            "training_data": df_train,
            "keywords": df_keywords
        }

    def keyword_lookup(self, data_point, keywords: list, label: int):
        """Given a datapoint and a list of keywords, lookup which of those 
            keywords are in the datapoint and return a label or abstain.
            This function is at the core of the current labeling functions implementation

        :param data_point: The paper title and paper abstract concatenated together
        :type data_point: str
        :param keywords: Keywords related to the application area
        :type keywords: list
        :param label: A label to assign to the labeling function. Either positive or negative
        :type label: int
        :return: Returns a label for the current datapoint if any keyword matches else, it abstains (returns -1)
        :rtype: int
        """
        current_data_point = f"{data_point.papertitle.lower()} {data_point.abstract.lower()}"
        for word in keywords:
            if (len(word.split(sep=" ")) > 2):
                w = word.strip().split(sep=" ")
                word_regex = rf"({w[0]})(\W*\w*){0,2}{w[1]}"
                return label if re.search(fr"(?i){word_regex}", current_data_point) else self.ABSTAIN
            elif any(word in current_data_point for word in keywords):
                return label
            else:
                return self.ABSTAIN
        return self.ABSTAIN

    def make_keyword_lf(self, keywords: list, label: int, lf_name: str = None):
        """Generate a labeling function from a keyword

        :param keywords: A list of keywords which will be used to generate the labeling functions
        :type keywords: list
        :param label: The label to assign to the labeling function
        :type label: int
        :param lf_name: A unique name for the labling function
        :type lf_name: str, optional
        :return: returns a labeling function which implements `keyword_lookup`
        :rtype: LabelingFunction
        """
        labeling_function_name = f"keyword_{re.sub(' ', '_', keywords[0].strip())}"
        return LabelingFunction(
            name=labeling_function_name,
            f=self.keyword_lookup,
            resources=dict(keywords=keywords, label=label))

    def create_wordlist_set(self, phrases: list):
        """Create a unique set of keywords from all the keywords from various application areas

        :param phrases: A list of keywords for a particular application area
        :type phrases: list
        :return: A unique set of keywords for all application areas
        :rtype: Set[List]
        """
        all_likely_phrases = []
        for phrase in phrases:
            word_list = phrase.split(sep=", ")
            all_likely_phrases.extend(word_list)
        return set(all_likely_phrases)

    def generate_lfs(self, word_list: set, keywords_dict: List[dict] = None, wipo_id: str = None):
        """Programmatically generate a dictionary of labeling functions for each WIPO application area

        :param word_list: A unique set of all the keywords across various application areas
        :type word_list: set
        :param keywords_dict: A dictionary of keywords for the specific application area, defaults to None
        :type keywords_dict: List[dict], optional
        :param wipo_id: The unique id of the application area, defaults to None
        :type wipo_id: str, optional
        :return: Returns a dictionary with the `wipo_id` as the key and a list of positive and negative
            labeling functions as the values
        :rtype: [type]
        """
        lfs = {}
        if(wipo_id):
            keywords_dict = list(
                filter(lambda kd: kd["Id"] == wipo_id, keywords_dict))

        for i, keyword in enumerate(keywords_dict):
            wordlist = keyword["ImprovedLikelyPhrases"].split(sep=", ")
            negative_wordlist = list(word_list - set(wordlist))
            positive_keyword_lfs = [self.make_keyword_lf(keywords=[wl], label=self.RELEVANT) for wl in wordlist]
            negative_keyword_lfs = [self.make_keyword_lf(keywords=[wl], label=self.IRRELEVANT) for wl in negative_wordlist]
            lfs[keyword["Id"]] = [*positive_keyword_lfs, *negative_keyword_lfs]
        return lfs

    def train_model(self, df_train: pd.DataFrame, application_area_lfs: list, analysis_path: str = "output", label_output_path: str = "labels.jsonl", save_model_path: str = None):
        """Using our labeling functions, we can train a probabilistic model which is able to generate weak labels for our data points

        :param df_train: The training data for the model
        :type df_train: pd.DataFrame
        :param application_area_lfs: A list of labeling functions to use in training the Label Model
        :type application_area_lfs: list
        :param analysis_path: Folder path where the model output should be stored, defaults to `PROJECT_ROOT/output`
        :type analysis_path: str, optional
        :param label_output_path: Path to file where probabilistic labels generated by the model should be stored, defaults to "labels.jsonl"
        :type label_output_path: str, optional
        :param save_model_path: A path to where the Label Model should be save at. If no path is provided, the model is not saved
        :type save_model_path: str, optional
        """
        file_name_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        applier = PandasLFApplier(lfs=application_area_lfs)
        L_train = applier.apply(df=df_train)

        model = LabelModel(cardinality=2, verbose=True)
        model.fit(L_train=L_train, n_epochs=800, log_freq=100)
        if(save_model_path is not None):
            model.save(save_model_path)

        int_labels, prob_labels = model.predict(L=L_train, return_probs=True, tie_break_policy="abstain")
        probs_df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(X=df_train, y=prob_labels, L=L_train)

        int_df_train_filtered, int_train_filtered = filter_unlabeled_dataframe(X=df_train, y=int_labels, L=L_train)
        # write out both labels. In the probability outputs, p_rel is the second probability listed
        assert list(probs_df_train_filtered["paperid"]) == list(
            int_df_train_filtered["paperid"])
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
        with open(f"{self.PROJECT_ROOT}/output/{analysis_path}_{file_name_timestamp}.csv", "w") as outfile:
            lf_analysis = lf_analysis.sort_values("Coverage")
            lf_analysis.to_csv(outfile, encoding="utf-8", index=True)


if __name__ == "__main__":
    # create labeling function generator
    LFG = LFGenerator()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--wipo_id", help="wipo id for lf generation. will generate lfs for all wipo application areas if left blank")
    parser.add_argument("--label_out", help="path to file where integer outputs should be written")
    parser.add_argument("--save_model", help="path to location for saving the trained model")
    parser.add_argument("--analysis_path", help="path to file where model analysis summary should be written")
    args = parser.parse_args()

    lfs = LFG.generate_lfs(word_list=LFG.word_list_set,
                           keywords_dict=LFG.keywords_dict, wipo_id=args.wipo_id)
    pool = Pool()
    pool.starmap(LFG.train_model, [(
        LFG.df_train,
        application_area_lfs,
        args.analysis_path or key,
        args.label_out,
        args.save_model
    ) for key, application_area_lfs, in lfs.items()])

    pool.close()
    pool.join()
