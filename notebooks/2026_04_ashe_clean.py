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
file_name = "_2026_04_20"

soc_coding_index_file = (
    f"{knowledge_bucket}soc2020volume2thecodingindexexcel03122025.xlsx"
)

c_llm = ClassificationLLM("gemini-2.5-flash", verbose=False)

### Access data ###
try:
    data = pd.read_csv(f"{output_folder}/soc_knowledgebase{file_name}.csv")

    with open(f"{output_folder}/checkpoint{file_name}.json", encoding="utf-8") as file:
        recent_batch_id = json.load(file)["completed_batches"]

    print("Database loaded from local.")
    print(f"STARTING FROM {recent_batch_id} batch (row {recent_batch_id * 10}).")

except FileNotFoundError:
    print("KNOWLEDGE_BUCKET not found in .env file. Please set it.")

    data = pd.read_csv(f"{knowledge_bucket}ASHE_classifai_soc_kb.csv")
    recent_batch_id = 0
    print("Database loaded from storage.")


### Read the data ###
def load_soc_index(filepath: str) -> pd.DataFrame:
    """Load SOC index.
    Provides a list of over 32,000 titles associated with employment.

    Args:
        filepath (str): A path to the file containing SOC Index.

    Returns:
        pd.DataFrame: A DataFrame with transformed job titles.
    """
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


soc_list = load_soc_index(soc_coding_index_file)


def load_soc_abbreviations(filepath: str) -> dict:
    """Load abbreviations used in SOC.
    Provides a list of abbreviations recognised in SOC titles.

    Args:
        filepath (str): A path to the file containing SOC Abbreviations.

    Returns:
        dict: A dictioary with abbreviations.
    """
    soc_abbreviation = pd.read_excel(
        filepath,
        sheet_name="Abbreviations",
        usecols=["Abbreviation", "Meaning"],
        dtype=str,
        header=5,
    )

    return soc_abbreviation


soc_abb = load_soc_abbreviations(soc_coding_index_file)
soc_abb_dict = soc_abb.set_index("Abbreviation")["Meaning"].to_dict()

### Remove duplicates ###
data["documents"] = data["documents"].str.strip()  # Remove leading space
data = data.drop_duplicates(subset='documents', keep='last')  # remove duplicates

### Add column for correct spelling ###
if "corrected_spelling" not in data:
    data["corrected_spelling"] = None
    data["corrected_spelling"] = data["corrected_spelling"].astype(str)

### Data manipulation ###
soc_list["title"] = soc_list[
    "title"
].str.upper()  # Upper case, as in the original dataset


### Remove rows that come directly from SOC ###
titles_list = soc_list["title"]
titles_list = titles_list.to_list()

not_in_list = data[~data["documents"].isin(titles_list)].reset_index(
    drop=True
)  # those are titles that didn't appear in soc_list
in_list = data[data["documents"].isin(titles_list)].reset_index(
    drop=True
)  # those are titles that came from soc_list

# save the subset of titles that are repeated from SOC index
in_list.to_csv(f"{output_folder}/ashe_in_soc_index{file_name}.csv")

# in_list.to_parquet(f"{output_folder}/ashe_in_soc_index{file_name}.parquet")
print("ASHE IN SOC SAVED.")


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


async def spelling(jt: str, abb_dict: dict):
    """Makes call to LLM to correct spelling mistakes.

    Args:
        jt (str): Job title to be corrected.
        abb_dict (dict): A dictionary with abbreviations and their meaning {abbreviation: meaning}.

    Returns:
        corrected (str)
    """
    corrected = await c_llm.clean_spelling(
        misspelled_string=jt, abbreviation_dictionary=abb_dict
    )

    return corrected.job_title_spelling.upper()


async def split_in_batches(df: pd.DataFrame):
    """Takes the whole dataset, splits in batches and uses LLM to correct spelling of the job title.

    Args:
        df (pd.DataFrame): file.
    """
    final_batch = math.ceil(len(df) / 10)  # get the amount of batches

    for current_batch_id in range(recent_batch_id, final_batch):
        print(f"batch {current_batch_id}")

        current_batch = await batching(df["documents"], current_batch_id)

        tasks = [spelling(jt=jt, abb_dict=soc_abb_dict) for jt in current_batch]
        responses = await asyncio.gather(*tasks)

        for k, llm_response in enumerate(responses):
            current_row = 10 * current_batch_id + k
            df.loc[current_row, "corrected_spelling"] = llm_response

        df.to_csv(f"{output_folder}/ashe_correct_spelling{file_name}.csv")

        json_data = {
            "completed_batches": current_batch_id,
        }
        with open(
            f"{output_folder}/checkpoint{file_name}.json", "w", encoding="utf8"
        ) as json_file:
            json.dump(
                json_data,
                json_file,
            )

        if current_batch_id + 1 == final_batch:
            df = df.drop_duplicates(
                subset=['corrected_spelling', 'label'],
                keep='last'
            )  # remove duplicates in the corrected spelling

            df.to_csv(f"{output_folder}/ashe_correct_spelling{file_name}.csv")

            # df.to_csv(f"{knowledge_bucket}soc_knowledgebase_clean{file_name}.csv")

            print("SAVED TO BUCKET")


asyncio.run(split_in_batches(not_in_list))
