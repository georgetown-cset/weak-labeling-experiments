import pandas as pd
import os
import csv
from os import path
import numpy as np
from google.cloud import bigquery

project_ID = "***REMOVED***"
client = bigquery.Client(project=project_ID)


def get_training_data(training_data_path: str, from_local_disk: bool, save_to_local: bool = True):

    if(from_local_disk and path.exists(training_data_path)):
        data = pd.read_csv(training_data_path, prefix=None).sample(
            frac=1, random_state=123).reset_index(drop=True)
        return data
    else:
        sql_likely_relevant = """
            select paper.PaperId, paper.PaperTitle, paper.Abstract, RAND() as rand
                from gcp_cset_mag.PapersWithAbstracts paper
                left join gcp_cset_mag.PaperFieldsOfStudy paper_fos on paper.PaperId = paper_fos.PaperId
                left join gcp_cset_mag.FieldsOfStudy fieldOfStudy on paper_fos.FieldOfStudyId = fieldOfStudy.FieldOfStudyId
                left join `gcp-gu-cset-analysis`.cn498_sandbox.mag_lid lang_id on cast(paper.PaperId as string) = lang_id.id
            where regexp_contains(NormalizedName, r"(?i)vehicle.*identification$|driving.*automation|connected vehicle|driver.*assistance")
            and (DocType != "Dataset") and (DocType != "Patent") and (paper.Abstract is not null) and (paper.Abstract != "") and (lower(lang_id.title_lid) = "english") and (lower(lang_id.abstract_lid) = "english")
            order by rand desc limit 5000
            """
        sql_likely_confusable = """
        select paper.PaperId, paper.PaperTitle, paper.Abstract, RAND() as rand
            from gcp_cset_mag.PapersWithAbstracts paper
            left join gcp_cset_mag.PaperFieldsOfStudy paper_fos on paper.PaperId = paper_fos.PaperId
            left join gcp_cset_mag.FieldsOfStudy fieldOfStudy on paper_fos.FieldOfStudyId = fieldOfStudy.FieldOfStudyId
            left join `gcp-gu-cset-analysis`.cn498_sandbox.mag_lid lang_id on cast(paper.PaperId as string) = lang_id.id
        where regexp_contains(fieldOfStudy.NormalizedName, r"(?i)computer vision")
        and (DocType != "Dataset") and (DocType != "Patent") and (paper.Abstract is not null) and (paper.Abstract != "") and (lower(lang_id.title_lid) = "english") and (lower(lang_id.abstract_lid) = "english")
        order by rand desc limit 1000
        """
        sql_likely_irrelevant = """
        select paper.PaperId, paper.PaperTitle, paper.Abstract, RAND() as rand
            from gcp_cset_mag.PapersWithAbstracts paper
            left join gcp_cset_mag.PaperFieldsOfStudy paper_fos on paper.PaperId = paper_fos.PaperId
            left join gcp_cset_mag.FieldsOfStudy fieldOfStudy on paper_fos.FieldOfStudyId = fieldOfStudy.FieldOfStudyId
            left join `gcp-gu-cset-analysis`.cn498_sandbox.mag_lid lang_id on cast(paper.PaperId as string) = lang_id.id
        where not regexp_contains(NormalizedName, r"(?i)vehicle.*identification$|driving.*automation|connected vehicle|driver.*assistance")
        and (DocType != "Dataset") and (DocType != "Patent") and (paper.Abstract is not null) and (paper.Abstract != "") and (lower(lang_id.title_lid) = "english") and (lower(lang_id.abstract_lid) = "english")
        order by rand desc limit 1000
        """

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


if __name__ == "__main__":
    get_training_data(training_data_path="../data/training_data.csv", from_local_disk=True, save_to_local=True)
