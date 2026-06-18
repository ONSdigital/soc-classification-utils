# %%
# pylint: disable=C0103, C0114, C0301, R0801, W0105

"""Noetbook attempting to create a SOC DIRECT LOOKUP.

Diasbling duplicate code - methods needs to be changed in other repos to reflect the change in data.
Diasbling line-too-long: commentary and discussion.
Disabling pointless-string-statement: comments to the code for reading clarity.
"""

# %%
import ast
import re

import dotenv
import pandas as pd

# %%
from occupational_classification.data_access.soc_data_access import (
    _combine_soc_index_job_title as combine_job_title,
)

# %%
input_folder = "data/soc_data"
file_name = "ashe_llm_soc_codes"
file_suffix = "_2026_06_16"
# file_suffix = "_2026_05_19"
knowledge_bucket = dotenv.get_key(".env", "KNOWLEDGE_BUCKET")

# %%
# read the data

try:
    data = pd.read_csv(
        f"{knowledge_bucket}wip_data/{file_name}{file_suffix}.csv", dtype={"label": str}
    )
    print("Database loaded from storage.")

except FileNotFoundError:
    print("File not found in the specified KNOWLEDGE_BUCKET.")
    data = pd.read_csv(
        f"{input_folder}/{file_name}{file_suffix}.csv", dtype={"label": str}
    )
    print("Database loaded from local.")

# %%
# use only columns needed
data = data[
    [
        "documents",
        # "corrected_spelling",
        "label",
        "codable",
        "llm_soc_code",
        "llm_soc_candidates",
        "reasoning",
    ]
]

data["documents"] = data["documents"].str.strip()


# %%
def parse_string(text):
    """Convert string to a list of dictionaries for SOC candidates."""
    if isinstance(text, str):
        processed = text.replace("SocCandidate(", "dict(")
        processed = re.sub(r"(\w+)=", r'"\1":', processed)
        processed = processed.replace("dict(", "{").replace(")", "}")
        return ast.literal_eval(processed)
    return []


# %%
# string to list of dictionaries
data["llm_soc_candidates"] = data["llm_soc_candidates"].map(parse_string)

# %%
print(f"llm {data["codable"].value_counts()}")


# %%
def access_soc_code_from_candidate_list(row_values: list[dict]) -> list[str]:
    """From list of potential SOC candidates, access SOC codes.

    Args:
        row_values (list[dict]): list of dictionaries with SOC candidates.

    Return:
        candidates (list[str]): list of 4-digit candidate codes.
    """
    if isinstance(row_values, list):
        candidates = []
        for row in row_values:
            if len(row) < 1:
                return None
            candidates.append(row.get("soc_code"))
    else:
        return None
    return candidates


# %%
def float_to_list_of_codes(row_values: float) -> str:
    """Convert float to a string of codes (str).

    Args:
        row_values (float): SOC code as a float.

    Return:
        row_values (str): SOC code as a string.
    """
    if isinstance(row_values, float):
        codes_list = [f"{row_values:.0f}"]
        print(type(codes_list))
        return codes_list
    return [row_values]


# %%
data["label"] = data["label"].astype(str)

# %%
msk = data["llm_soc_code"].isna()  # take rows, where LLM didn't provide a code.

# %%
data.loc[~msk, "llm_soc_code"] = data.loc[~msk, "llm_soc_code"].apply(
    float_to_list_of_codes
)

# %%
data.loc[msk, "llm_soc_code"] = data.loc[msk, "llm_soc_candidates"].apply(
    access_soc_code_from_candidate_list
)


# %%
def check_agreement(df: pd.DataFrame, df_source: str):
    """Checks agreement between ASHE and LLM assigned codes.

    Args:
        df (pd.DataFrame): dataframe containing columns 'label' and 'llm_soc_code' with codes.
        df_source (str): String indicaitng the source of the dataframe (ASHE or soc index).
    """
    agr, in_cand = 0, 0
    # check if 'label' is the same as 'llm_soc_code'.
    # If LLM uncodable, check if 'label' in candidates.
    for row in range(len(df)):
        if len(df.iloc[row]["llm_soc_code"]) == 1:
            agr += df.iloc[row]["label"] == df.iloc[row]["llm_soc_code"][0]
            df.loc[row, "codable"] = True
        elif len(df.iloc[row]["llm_soc_code"]) > 1:
            in_cand += df.iloc[row]["label"] in df.iloc[row]["llm_soc_code"]

    print(
        f"Agreement full {df_source}: {agr} ({round(agr/len(df), 2) * 100}% of all rows)"
    )
    print(
        f"Agreement (code in candidates) {df_source}: {in_cand} ({round(in_cand/len(df), 2) * 100}% of all rows)"  # pylint: disable=C0301
    )
    print(
        f"Agreement (label the same or within candidates) {df_source}: {agr + in_cand} ({round((agr + in_cand)/len(df), 2) * 100}% of all rows)"  # pylint: disable=C0301
    )


# %%
check_agreement(data, "ASHE and LLM")


# %%
def check_code_count(df: pd.DataFrame, df_source: str):
    """Check if the LLM assigned a sigle, multiple, or none codes when assessing SOC codes.

    Args:
        df (pd.DataFrame): dataframe containing LLM assessment of SOC codes.
            Requires 'llm_soc_code' column.
        df_source (str): String indicaitng the source of the dataframe (ASHE or soc index).
    """
    longer, shorter, one_code = 0, 0, 0
    for code in df["llm_soc_code"]:
        if isinstance(code, list):
            if len(code) > 1:
                longer += 1
            if len(code) < 1:
                shorter += 1
            if len(code) == 1:
                one_code += 1
        else:
            one_code += 1

    print(
        f"More than one code {df_source}: {longer} ({round(longer/len(df) * 100, 2)}%)"
    )
    print(
        f"No codes assigned {df_source}: {shorter} ({round(shorter/len(df) * 100, 2)}%)"
    )
    print(
        f"One code assigned {df_source}: {one_code} ({round(one_code/len(df) * 100, 2)}%)"
    )


# %%
check_code_count(data, "ASHE")

# %%
full_data_codable = data[data["codable"]]

# %%
data_only_columns = data[["documents", "llm_soc_code"]]

# %%
data_one_code = data_only_columns[data_only_columns["llm_soc_code"].str.len() == 1]

# %%
e = data_one_code["llm_soc_code"].str[0]

# %%
numeric = pd.to_numeric(e, errors="coerce")

# %%
data_one_code["llm_soc_code"] = numeric

# %%
data_one_code = data_one_code.dropna(subset=["llm_soc_code"])

# %%
data_one_code["llm_soc_code"] = data_one_code["llm_soc_code"].astype(int)

# %%
data_one_code = data_one_code.rename(
    columns={"corrected_spelling": "documents", "llm_soc_code": "label"}
)

# %%
data_one_code = data_one_code.drop_duplicates(
    subset=["documents", "label"], keep="last", ignore_index=True
)


# %%
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


# %%
knowledge_bucket = dotenv.get_key(".env", "KNOWLEDGE_BUCKET")

# %%
s_list = load_soc_framework(
    f"{knowledge_bucket}soc2020volume2thecodingindexexcel03122025.xlsx"
)
s_list = s_list[s_list["code"].notna()]

# %%
codes_from_framework_str = list(s_list["code"].value_counts().keys())

# %%
codes_from_framework_int = []
for k in codes_from_framework_str:
    codes_from_framework_int.append(int(k))

# %%
phantom_codes = (
    data_one_code[~data_one_code["label"].isin(codes_from_framework_int)]["label"]
    .value_counts()
    .keys()
)

# %%
print("codes that don't appear in the SOC codes list\n", phantom_codes)

# %%
data_one_code_no_phantoms = data_one_code[
    data_one_code["label"].isin(codes_from_framework_int)
]

# %%
coded_all = len(data_one_code)

# %%
coded_no_phantom = len(data_one_code_no_phantoms)

# %%
diff = len(data_one_code) - len(data_one_code_no_phantoms)

# %%
drop = diff / coded_all * 100

# %%
print(
    f"with phantoms: {coded_all}\ncoded no phantoms: {coded_no_phantom}\ndiff: {diff}\ndrop(%): {drop:.2f}"  # pylint: disable=C0301
)

# %%
print(data_one_code_no_phantoms)

# %%
print(
    "check if there is any duplicates\n",
    data_one_code_no_phantoms[
        data_one_code_no_phantoms.duplicated(subset=["documents"])
    ],
)

# %%
data_one_code_no_phantoms = data_one_code_no_phantoms.drop_duplicates(
    subset=["documents"], keep="last", ignore_index=True
)

# %%
""" data_one_code_no_phantoms contains codes assigned by the LLM. Some of the codes were not present in the SOC codes list, and have been removed.
Those codes not neccessairly agree with codes initially assigned in ASHE dataset.
"""

# %%
# data_one_code_no_phantoms.to_csv("soc_data/SOC_DIRECT_LOOKUP.csv")

# %%
# data_one_code_no_phantoms.to_csv(f"{knowledge_bucket}SOC_DIRECT_LOOKUP.csv")

# %% [markdown]
# # AGREEMENT

# %%
"""Select a subset of codes, where LLM and ASHE assign the same code for a given job title.
"""

# %%
msk_codable = data["codable"]

# %%
data_codable = data[msk_codable]

# %%
len(data_codable[data_codable["llm_soc_code"].str.len() > 1])

# %%
len(data_codable[(data_codable["llm_soc_code"].str.len() == 1)])

# %%
len(
    data_codable[data_codable["llm_soc_code"].str.len() < 1]
)  # expect 0 - if is codable, there should be a code available

# %%
print(data_codable[data_codable["llm_soc_code"].str.len() == 1])

# %%
one_code_subset = data_codable[data_codable["llm_soc_code"].str.len() == 1]

# %%
codes_with_agreement = one_code_subset[
    one_code_subset.apply(lambda r: str(r["label"]) in str(r["llm_soc_code"]), axis=1)
].reset_index(drop=True)

# %%
soc_lookup = codes_with_agreement[["documents", "label"]]

# %%
# save this once all is finished

# %% [markdown]
# # One code from LLM - why disagreement?

# %%
full_data_one_code = data[data["llm_soc_code"].str.len() == 1]

# %%
one_code_disagreement = full_data_one_code[
    full_data_one_code.apply(
        lambda r: str(r["label"]) not in str(r["llm_soc_candidates"]), axis=1
    )
].reset_index(drop=True)

# %%
print(one_code_disagreement)

# %%
"""Look at the cases, where:
- LLM claims is codable ('codable' == True)
- ASHE does not agree with LLM ('label' != 'llm_soc_code')
- ASHE is one of the candidates selected by LLM ('label' in 'llm_soc_candidates')
"""


# %%
def get_candidates_list(row: pd.Series) -> list:
    """Get a list of candidates determined by LLM.

    Args:
        row: pd.Series: row with LLM output

    Returns:
        list: lsit of candidates.
    """
    candidates = []
    for i in row["llm_soc_candidates"]:
        candidates.append(i["soc_code"])
    return candidates


# %%
ashe_llm_disagreement_multi_candidate = one_code_disagreement[
    one_code_disagreement["llm_soc_candidates"].str.len() > 1
].reset_index(drop=True)

# %%
ashe_llm_disagreement_multi_candidate.loc[:, "candidate_list"] = (
    ashe_llm_disagreement_multi_candidate.apply(get_candidates_list, axis=1)
)

# %%
ashe_in_canidates = ashe_llm_disagreement_multi_candidate[
    ashe_llm_disagreement_multi_candidate.apply(
        lambda r: str(r["label"]) in str(r["candidate_list"]), axis=1
    )
]

# %%
print(
    f"""There is {len(ashe_in_canidates)} rows, where code determined by ASHE appears in the cadnidates from LLM, when LLM assessed the job title is codable."""
)

# %% [markdown]
# # How many of the rows that are in SOC INDEX have agreement/don't have agreement with ASHE

# %%
# Access SOC_INDEX data
soc_coding_index_file = (
    f"{knowledge_bucket}soc2020volume2thecodingindexexcel03122025.xlsx"
)


# %%
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
        columns={"indexocc-natural_word_order": "indexocc", "soc_2020": "code"}
    )

    soc_index_df = soc_index_df[soc_index_df["code"] != "}}}}"]
    soc_index_df = soc_index_df.dropna(subset=["code", "indexocc"])
    soc_index_df["title"] = soc_index_df.apply(combine_job_title, axis=1)
    soc_index_df = soc_index_df[["code", "title"]]
    soc_index_df["title"] = soc_index_df["title"].str.capitalize()

    return soc_index_df


# %%
soc_list = load_soc_index(soc_coding_index_file)

# %%
soc_list["title"] = soc_list["title"].str.upper()

# %%
titles_list = soc_list["title"]
titles_list = titles_list.to_list()

# %%
# get subset of the ASHE data that comes from soc_index
in_list = data[data["documents"].isin(titles_list)].reset_index(drop=True)

# %%
in_list_codable = in_list[in_list["codable"]]

# %%
in_list_codable_disagreement = in_list_codable[
    in_list_codable.apply(
        lambda r: str(r["label"]) not in str(r["llm_soc_candidates"]), axis=1
    )
].reset_index(drop=True)
in_list_codable_agreement = in_list_codable[
    in_list_codable.apply(
        lambda r: str(r["label"]) in str(r["llm_soc_candidates"]), axis=1
    )
].reset_index(drop=True)

# %%
in_list_codable_disagreement_one_code = in_list_codable_disagreement[
    in_list_codable_disagreement["llm_soc_code"].str.len() == 1
]
in_list_codable_agreement_one_code = in_list_codable_agreement[
    in_list_codable_agreement["llm_soc_code"].str.len() == 1
]

# %%
soc_lookup = (
    pd.concat([soc_lookup, in_list_codable_agreement_one_code[["documents", "label"]]])
    .drop_duplicates(subset=["documents", "label"])
    .reset_index(drop=True)
)

# %% [markdown]
# # LLM candidates - high likelihood (0.9/0.7)

# %%
data_multiple_candidates = data[data["llm_soc_candidates"].str.len() > 1].reset_index(
    drop=True
)


# %%
def get_high_candidate(row: pd.Series) -> str:
    """Get a most likely candidate with likelihood greater than 0.9 (assessed by the LLM),
    where only one candidate got that score.

    Args:
        row: pd.Series: row with LLM output

    Returns:
        str: most likely candidate.
    """
    high_likelihood = []
    for i in row["llm_soc_candidates"]:
        if i["likelihood"] >= 0.9:  # noqa: PLR2004
            high_likelihood.append(i)
    if len(high_likelihood) != 1:
        return None
    return high_likelihood[0]["soc_code"]


# %%
def get_high_candidate_with_low_other(row: pd.Series) -> str:
    """Get a most likely candidate with likelihood greater than 0.9 (assessed by the LLM),
    where only one candidate got that score, and no other candidates got likelihood score above 0.7.

    Args:
        row: pd.Series: row with LLM output

    Returns:
        str: most likely candidate.
    """
    high_likelihood, lower_likelihood = [], []

    for i in row["llm_soc_candidates"]:
        if i["likelihood"] >= 0.9:  # noqa: PLR2004
            high_likelihood.append(i)
        elif i["likelihood"] >= 0.7:  # noqa: PLR2004
            lower_likelihood.append(i)

    if len(high_likelihood) != 1 or len(lower_likelihood) > 0:
        return None
    return high_likelihood[0]["soc_code"]


# %%
data_multiple_candidates.loc[:, "most_likely_candidate"] = (
    data_multiple_candidates.apply(get_high_candidate, axis=1)
)

# %%
print(len(data_multiple_candidates))

# %%
data_high_likelihood = data_multiple_candidates[
    data_multiple_candidates["most_likely_candidate"].notna()
]

# %%
data_high_likelihood_agreement = data_high_likelihood[
    data_high_likelihood.apply(
        lambda r: str(r["label"]) == str(r["most_likely_candidate"]), axis=1
    )
].reset_index(drop=True)

# %%
print(data_high_likelihood_agreement.iloc[0])

# %%
misspelled = 0
for k in data_high_likelihood_agreement["reasoning"]:
    # print(k)
    if "misspelling" in k or "misspelled" in k:
        misspelled += 1

# %%
print(misspelled)

# %%
print(
    f"""We looked at the rows, where the LLM decided thre is more than one possible SOC code candidate {len(data_multiple_candidates)} codes ({round((len(data_multiple_candidates)/len(data)) * 100, 2)}% of all codes).
To that subset of data, we added a new column 'most_likely_candidate'.
It was populated with codes, that were assessed to have a high (0.9) likelihood.
If the LLM assigned more than one code with high likelihood, those were disregarded,
as there is no way to determine which code is more likely, according to the LLM, meaning it is not unambiguous.

Only one candidate with 0.9 likelihood was assigned for {len(data_high_likelihood)} rows.

Next, we compared the agreement between the label assigned in the original data with the "most_likely_candidate",
which resulted in {len(data_high_likelihood_agreement)} cases.
"""
)

# %%
data_high = data[data["llm_soc_candidates"].str.len() > 1].reset_index(drop=True)

# %%
data_high.loc[:, "most_likely_candidate"] = data_high.apply(
    get_high_candidate_with_low_other, axis=1
)

# %%
data_high_notna = data_high[data_high["most_likely_candidate"].notna()]

# %%
data_high_notna_agreement = data_high_notna[
    data_high_notna.apply(
        lambda r: str(r["label"]) == str(r["most_likely_candidate"]), axis=1
    )
].reset_index(drop=True)

# %%
soc_lookup = (
    pd.concat([soc_lookup, data_high_notna_agreement[["documents", "label"]]])
    .drop_duplicates(subset=["documents", "label"])
    .reset_index(drop=True)
)

# %%
# soc_lookup.to_csv(f"{knowledge_bucket}wip_data/soc_kb_for_direct_lookup.csv")

# %%
len(soc_lookup)
