# pylint: disable=C0103, C0114
import asyncio
import json
import math

import dotenv
import pandas as pd
from occupational_classification.data_access.soc_data_access import combine_job_title

from occupational_classification_utils.llm.llm import ClassificationLLM

### Constants ###
knowledge_bucket = dotenv.get_key(".env", "KNOWLEDGE_BUCKET")

output_folder = "notebooks/ashe_data_cleaning"
file_name = "ashe_in_soc_index"
# file_name = "ashe_correct_spelling"
input_file_name = "_2026_04_20"
output_file_name = "_llm_soc_codes_framework_2026_04_23"
# JOB_TITLE_COLUMN = "corrected_spelling"
JOB_TITLE_COLUMN = "documents"

### Initiate llm connection ###
c_llm = ClassificationLLM("gemini-2.5-flash", verbose=False)

### Access data ###
try:
    # data = pd.read_csv(f"{output_folder}/checkpoint_spelling_llm{input_file_name}.csv")
    data = pd.read_csv(f"{output_folder}/{file_name}{input_file_name}.csv")
    print("Database loaded from local.")
except FileNotFoundError:
    print("KNOWLEDGE_BUCKET not found in .env file. Please set it.")
    data = pd.read_csv(f"{knowledge_bucket}ASHE_classifai_soc_kb.csv")
    print("Database loaded from storage.")

try:
    with open(
        f"{output_folder}/{file_name}{input_file_name}.json", encoding="utf-8"
    ) as file:
        recent_batch_id = json.load(file)["completed_batches"]
except FileNotFoundError:
    recent_batch_id = 0
print(
    f"STARTING FROM {recent_batch_id} batch (row {recent_batch_id * 10} out of {len(data)})."
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
s_list = s_list[s_list["code"].notna()]

if isinstance(s_list, pd.DataFrame):
    s_list = s_list.to_dict(orient="records")

data = data.drop_duplicates(subset=JOB_TITLE_COLUMN)

# data_sample = data.sample(100)
# data_sample = data[:100]


async def run_soc_code(jt: str):
    """Makes call to LLM to decide whether is codable to SOC.

    Args:
        jt (str): Job title to be categorised.

    Returns:
        codable (bool)
    """
    response = await c_llm.get_soc_code(job_title=jt, short_list=s_list)
    return response


async def batching(job_titles_column: pd.Series, batch_id: int):
    """Takes next batch from the dataset of size 10.

    Args:
        job_titles_column (pd.Series): A coulmn with job titles.
        batch_id (int): number of the batch.

    Returns:
        job_titles_column: snippet of the data provided of size 10.
    """
    batch = batch_id
    start_id = batch * 10
    end_id = batch * 10 + 10
    return job_titles_column[start_id:end_id].copy()


async def split_in_batches(document: pd.DataFrame):
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

    final_batch = math.ceil(len(document) / 10)  # get the amount of batches

    for current_batch_id in range(recent_batch_id, final_batch):

        print(f"batch {current_batch_id}")

        current_batch = await batching(
            document[JOB_TITLE_COLUMN], current_batch_id
        )  # does it have to be async?

        # k = 0

        tasks = [run_soc_code(jt) for jt in current_batch]
        responses = await asyncio.gather(*tasks)

        for k, llm_response in enumerate(responses):
            codable = llm_response.codable
            soc_code = llm_response.soc_code
            soc_candidates = llm_response.soc_candidates
            reasoning = llm_response.reasoning

            current_row = 10 * current_batch_id + k
            document.at[current_row, "codable"] = codable
            document.at[current_row, "llm_soc_code"] = soc_code
            document.at[current_row, "llm_soc_candidates"] = soc_candidates
            document.at[current_row, "reasoning"] = reasoning

        document.to_csv(f"{output_folder}/{file_name}{output_file_name}.csv")
        # document.to_parquet(f"{output_folder}/{file_name}{output_file_name}.parquet")

        # pylint: disable=R0801
        json_data = {
            "completed_batches": current_batch_id,
        }
        with open(
            f"{output_folder}/{file_name}{output_file_name}.json", "w", encoding="utf8"
        ) as json_file:
            json.dump(
                json_data,
                json_file,
            )

        if current_batch_id + 1 == final_batch:
            document.to_csv(
                f"{knowledge_bucket}wip_data/{file_name}{output_file_name}.csv"
            )
            print("SAVED TO BUCKET")


asyncio.run(split_in_batches(data))
# asyncio.run(split_in_batches(data_sample))
# print(data_sample)
# print(data_sample["codable"].value_counts())
