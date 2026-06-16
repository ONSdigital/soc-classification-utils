# pylint: disable=C0103, R0801
"""This is not a notebook. Run as a script.

Allows to use LLM to assign SOC codes.

To execute, run:
    `python notebooks/assign_soc_code_2026_03.py `

Diasbling duplicate code - methods needs to be changed in other repos to reflect the change in data.
"""

import asyncio
import json
import math
import os

import dotenv
import pandas as pd

from occupational_classification_utils.llm.llm import ClassificationLLM

### Constants ###
knowledge_bucket = dotenv.get_key(".env", "KNOWLEDGE_BUCKET")

# location and names for saving the output files
data_folder = "src/occupational_classification_utils/data/soc_data"
file_name = "ashe_llm_soc_codes"
output_file_name = "_2026_05_19"

JOB_TITLE_COLUMN = "documents"
CODE_COLUMN = "label"

BATCH_SIZE = 10

### Initiate llm connection ###
c_llm = ClassificationLLM("gemini-2.5-flash", verbose=False)

### Access data ###
try:
    data = pd.read_csv(f"{knowledge_bucket}ASHE_classifai_soc_kb.csv")
    print("Database loaded from storage.")

except FileNotFoundError:
    print("File not found in the specified KNOWLEDGE_BUCKET.")
    data = pd.read_csv(f"{data_folder}/ASHE_classifai_soc_kb.csv")
    print("Database loaded from local.")

try:
    with open(
        f"{data_folder}/{file_name}{output_file_name}.json", encoding="utf-8"
    ) as file:
        recent_batch_id = json.load(file)["completed_batches"]
except FileNotFoundError:
    recent_batch_id = 0

print(
    f"STARTING FROM {recent_batch_id} batch (row {recent_batch_id * BATCH_SIZE} out of {len(data)})."  # pylint: disable=C0301
)


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


def batching(job_titles_column: pd.Series, batch_id: int):
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

        current_batch = batching(document[JOB_TITLE_COLUMN], current_batch_id)

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
                # "corrected_spelling",
                "codable",
                "llm_soc_code",
                "llm_soc_candidates",
                "reasoning",
            ]
        ]

        rows_to_save.to_csv(
            f"{data_folder}/{file_name}{output_file_name}.csv",
            mode="a",
            header=False,
            index=False,
        )

        with open(
            f"{data_folder}/{file_name}{output_file_name}.json", "w", encoding="utf8"
        ) as json_file:
            json.dump(
                {
                    "completed_batches": current_batch_id + 1,
                    "total batches": final_batch,
                },
                json_file,
            )

        # if current_batch_id + 1 == final_batch:
        #     document.to_csv(
        #         f"{knowledge_bucket}wip_data/{file_name}{output_file_name}.csv"
        #     )
        #     print("SAVED TO BUCKET")


if not os.path.exists(f"{data_folder}/{file_name}{output_file_name}.csv"):
    all_columns = [
        "documents",
        "label",
        # "corrected_spelling",
        "codable",
        "llm_soc_code",
        "llm_soc_candidates",
        "reasoning",
    ]
    pd.DataFrame(columns=all_columns).to_csv(
        f"{data_folder}/{file_name}{output_file_name}.csv", index=False
    )

asyncio.run(split_in_batches(data))
