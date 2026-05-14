# pylint: disable=C0103, C0114
"""This is not a notebook. Run as a script.

Allows to use LLM to assign SOC codes.

To execute, run:
    `python notebooks/assign_soc_code_2026_03.py `
"""

import asyncio
import json
import math
import os

import dotenv
import pandas as pd
from occupational_classification.data_access.soc_data_access import (
    _combine_soc_index_job_title as combine_job_title,
)

from occupational_classification_utils.llm.llm import ClassificationLLM

### Constants ###
knowledge_bucket = dotenv.get_key(".env", "KNOWLEDGE_BUCKET")

output_folder = "notebooks/soc_data"

# file_name = "ashe_in_soc_index"
file_name = "ashe_correct_spelling"

input_file_name = "_2026_04_20"
# input_file_name = "_clean"

output_file_name = "_llm_soc_codes_index_2026_04_30_attempt8"

JOB_TITLE_COLUMN = "corrected_spelling"
# JOB_TITLE_COLUMN = "documents"
CODE_COLUMN = "label"

BATCH_SIZE = 10

### Initiate llm connection ###
c_llm = ClassificationLLM("gemini-2.5-flash", verbose=False)

### Access data ###
try:
    data = pd.read_csv(f"{output_folder}/{file_name}{input_file_name}.csv")
    print("Database loaded from local.")
except FileNotFoundError:
    print("KNOWLEDGE_BUCKET not found in .env file. Please set it.")
    data = pd.read_csv(f"{knowledge_bucket}{file_name}{input_file_name}.csv")
    print("Database loaded from storage.")

try:
    with open(
        f"{output_folder}/{file_name}{output_file_name}.json", encoding="utf-8"
    ) as file:
        recent_batch_id = json.load(file)["completed_batches"]
except FileNotFoundError:
    recent_batch_id = 0

print(
    f"STARTING FROM {recent_batch_id} batch (row {recent_batch_id * BATCH_SIZE} out of {len(data)})."  # pylint: disable=C0301
)


### Read the data ###
def load_soc_index(filepath: str) -> pd.DataFrame:
    """Load SOC index.
    Provides a list of over 32,000 titles associated with employment.

    Args:
        filepath (str): A path to the file containing SOC Index.

    Returns:
        pd.DataFrame: A DataFrame with transformed job titles.
    """
    # pylint: disable=R0801
    soc_index_df = pd.read_excel(
        filepath,
        sheet_name="SOC2020 coding index",
        usecols=["SOC_2020", "INDEXOCC-natural_word_order", "ADD", "IND"],
        dtype=str,
    )

    soc_index_df.columns = [col.lower() for col in soc_index_df.columns]

    soc_index_df = soc_index_df.rename(
        columns={"indexocc-natural_word_order": "natural_word", "soc_2020": "code"}
    )

    soc_index_df = soc_index_df[soc_index_df["code"] != "}}}}"]
    soc_index_df = soc_index_df.dropna(subset=["code", "natural_word"])
    soc_index_df["title"] = soc_index_df.apply(combine_job_title, axis=1)
    soc_index_df = soc_index_df[["code", "title"]]
    soc_index_df["title"] = soc_index_df["title"].str.capitalize()

    return soc_index_df


def load_soc_framework(filepath: str) -> pd.DataFrame:
    """Load SOC structure.

    Provides structure with all levels and names of the SOC 2020.

    Args:
        filepath (str): A path to the file containing SOC Structure.

    Returns:
        pd.DataFrame: A DataFrame containing group code, group title,
        group description, typical entry routes and associated qualifications,
        and list of tasks.
    """
    # pylint: disable=R0801
    soc_df = pd.read_excel(
        filepath,
        sheet_name="SOC2020 framework",
        usecols=[
            "SOC2020 Unit Group",
            "SOC2020 Group Title",
        ],
        dtype=str,
    )
    soc_df.columns = [
        col.lower().replace(" ", "_").replace("__", "_").replace("\n", "")
        for col in soc_df.columns
    ]
    soc_df = soc_df.rename(
        columns={"soc2020_unit_group": "code", "soc2020_group_title": "title"}
    )

    for col in soc_df.columns:
        soc_df[col] = soc_df[col].str.strip()

    return soc_df


### Create a dictionary for short list ###
s_list = load_soc_framework(
    f"{knowledge_bucket}soc2020volume2thecodingindexexcel03122025.xlsx"
)
# s_list = load_soc_index(
#     f"{knowledge_bucket}soc2020volume2thecodingindexexcel03122025.xlsx"
# )

s_list = s_list[s_list["code"].notna()]

if isinstance(s_list, pd.DataFrame):
    s_list = s_list.to_dict(orient="records")

data = data.drop_duplicates(subset=[JOB_TITLE_COLUMN, CODE_COLUMN], keep="last")


async def run_soc_code(jt: str):
    """Makes call to LLM to decide whether is codable to SOC.

    Args:
        jt (str): Job title to be categorised.

    Returns:
        codable (bool)
    """
    response = await c_llm.get_soc_code(job_title=jt)
    return response


async def batching(job_titles_column: pd.Series, batch_id: int):
    """Takes next batch from the dataset of specified size.

    Args:
        job_titles_column (pd.Series): A coulmn with job titles.
        batch_id (int): number of the batch.

    Returns:
        job_titles_column: snippet of the data provided of specified size.
    """
    batch = batch_id
    start_id = batch * BATCH_SIZE
    end_id = batch * BATCH_SIZE + BATCH_SIZE
    return job_titles_column[start_id:end_id].copy()


async def split_in_batches(document: pd.DataFrame):  # pylint: disable=R0914
    """Takes the whole dataset, splits in batches and uses LLM to determine whether
    the job title allows to provide a final SOC code.

    Args:
        document (pd.DataFrame): file.
    """
    # Check if "codable" column exists
    if "codable" not in document:
        document["codable"] = None
        document["codable"] = document["codable"].astype(bool)
    if "llm_soc_code" not in document:
        document["llm_soc_code"] = None
        document["llm_soc_code"] = document["llm_soc_code"].astype(float)
    if "llm_soc_candidates" not in document:
        document["llm_soc_candidates"] = None
        document["llm_soc_candidates"] = document["llm_soc_candidates"].astype(object)
    if "reasoning" not in document:
        document["reasoning"] = None
        document["reasoning"] = document["reasoning"].astype(str)

    final_batch = math.ceil(len(document) / BATCH_SIZE)  # get the amount of batches

    for current_batch_id in range(recent_batch_id, final_batch):

        print(f"batch {current_batch_id}")

        current_batch = await batching(document[JOB_TITLE_COLUMN], current_batch_id)

        tasks = [run_soc_code(jt) for jt in current_batch]
        responses = await asyncio.gather(*tasks)

        for k, llm_response in enumerate(responses):
            codable = llm_response.codable
            soc_code = llm_response.soc_code
            soc_candidates = llm_response.soc_candidates
            reasoning = llm_response.reasoning

            current_row = BATCH_SIZE * current_batch_id + k
            document.at[current_row, "codable"] = codable
            document.at[current_row, "llm_soc_code"] = soc_code
            document.at[current_row, "llm_soc_candidates"] = soc_candidates
            document.at[current_row, "reasoning"] = reasoning

        start_row = current_batch_id * BATCH_SIZE

        rows_to_save = document.iloc[start_row : start_row + BATCH_SIZE][
            [
                "documents",
                "label",
                "corrected_spelling",
                "codable",
                "llm_soc_code",
                "llm_soc_candidates",
                "reasoning",
            ]
        ]

        rows_to_save.to_csv(
            f"{output_folder}/{file_name}{output_file_name}.csv",
            mode="a",
            header=False,
            index=False,
        )

        with open(
            f"{output_folder}/{file_name}{output_file_name}.json", "w", encoding="utf8"
        ) as json_file:
            json.dump(
                {
                    "completed_batches": current_batch_id + 1,
                },
                json_file,
            )

        # if current_batch_id + 1 == final_batch:
        #     document.to_csv(
        #         f"{knowledge_bucket}wip_data/{file_name}{output_file_name}.csv"
        #     )
        #     print("SAVED TO BUCKET")


if not os.path.exists(f"{output_folder}/{file_name}{output_file_name}.csv"):
    all_columns = [
        "documents",
        "label",
        "corrected_spelling",
        "codable",
        "llm_soc_code",
        "llm_soc_candidates",
        "reasoning",
    ]
    pd.DataFrame(columns=all_columns).to_csv(
        f"{output_folder}/{file_name}{output_file_name}.csv", index=False
    )

asyncio.run(split_in_batches(data))
