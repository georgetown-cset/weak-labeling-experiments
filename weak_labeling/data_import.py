import pandas as pd
import os
import csv
import re
from os import path
import numpy as np
from google.cloud import bigquery

project_ID = "***REMOVED***"
client = bigquery.Client(project=project_ID)


def build_query(query_regex: str, limit: int):
    return f"""
            select paper.PaperId, paper.PaperTitle, paper.Abstract, RAND() as rand
                from gcp_cset_mag.PapersWithAbstracts paper
                left join gcp_cset_mag.PaperFieldsOfStudy paper_fos on paper.PaperId = paper_fos.PaperId
                left join gcp_cset_mag.FieldsOfStudy fieldOfStudy on paper_fos.FieldOfStudyId = fieldOfStudy.FieldOfStudyId
                left join `***REMOVED***.cn498_sandbox.mag_lid` lang_id on cast(paper.PaperId as string) = lang_id.id
            where regexp_contains(NormalizedName, {query_regex})
            and (DocType != "Dataset") and (DocType != "Patent") and (paper.Abstract is not null) and (paper.Abstract != "") and (lower(lang_id.title_lid) = "english") and (lower(lang_id.abstract_lid) = "english")
            order by rand desc limit {limit}
            """


def get_training_data(training_data_path: str, from_local_disk: bool, save_to_local: bool = True):

    if(from_local_disk and path.exists(training_data_path)):
        data = pd.read_csv(training_data_path, prefix=None).sample(
            frac=1, random_state=123).reset_index(drop=True)
        return data
    else:
        regex_relevant = r"\"(?i)vehicle.*identification$|driving.*automation|connected vehicle|driver.*assistance\""
        regex_irrelevant = r"\"(?i)vehicle.*identification$|driving.*automation|connected vehicle|driver.*assistance\""
        regex_confusable = r"\"(?i)computer vision\""
        sql_likely_relevant = build_query(regex_relevant, 5000)
        sql_likely_confusable = build_query(regex_confusable, 1000)
        sql_likely_irrelevant = build_query(regex_irrelevant, 1000)

        likely_relevant = pd.read_gbq(
            query=sql_likely_relevant, project_id=project_ID, dialect="standard")
        likely_irrelevant = pd.read_gbq(
            query=sql_likely_irrelevant, project_id=project_ID, dialect="standard")
        likely_confusable = pd.read_gbq(
            query=sql_likely_confusable, project_id=project_ID, dialect="standard")
        df = pd.concat([likely_relevant, likely_confusable, likely_irrelevant])
        df.columns = map(str.lower, df.columns)

        if(save_to_local):
            df.to_csv("data.csv", encoding="utf-8", index=False,
                      quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        return df


def regex_from_keywords(words: list):
    word_list = map(lambda w: w.strip().replace(" ", ".*"), words)
    reg_exp = f"r\"(?i){'|'.join(word_list)}\""
    return str(reg_exp)


def wipo_training_data(keywords_path: str = "../data/keyword_categories_custom.csv", training_data_path: str = "../data/wipo_training_data.csv", train_size: int = 10):
    keywords_categories = pd.read_csv(keywords_path, prefix=None)
    with open(file=training_data_path, mode="a+") as training_data_dump:
        regex_list = []
        for words in keywords_categories.ImprovedLikelyPhrases:
            words_reg_exp = words.split(sep=", ")
            regex_list.extend(words_reg_exp)
        regex_list = list(set(regex_list))
        reg_exp = regex_from_keywords(words=regex_list)
        data_query = build_query(query_regex=reg_exp, limit=train_size)

        # get the data from bigquery and save to a csv file locally
        df_training_data = pd.read_gbq(query=data_query, project_id=project_ID, dialect="standard")
        df_training_data.to_csv(training_data_dump, mode="a",  encoding="utf-8", header=os.path.exists(training_data_path), index=False,
                                quotechar='"', quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
    # get_training_data(training_data_path="../data/training_data.csv", from_local_disk=True, save_to_local=True)
    # train_size = n * 54, where n is the number of data points per application field (n = 10)
    wipo_training_data(train_size=530)
