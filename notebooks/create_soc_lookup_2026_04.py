# pylint: disable=C0103, C0114
# %%
import ast
import re

import dotenv
import pandas as pd

# %%
output_folder = "soc_data"
# output_folder = "sample"

file_name = "ashe_in_soc_index_llm_soc_codes_"
index_file = "2026_04_22"

file_name_not_in_index = "ashe_correct_spelling_llm_soc_codes_"
# file_name_not_in_index = "data_subset_llm_soc_codes_"
# framework_file = "index_2026_04_21"
framework_file = "index_2026_04_30"


# %%
data_not_in_index = pd.read_csv(
    f"{output_folder}/{file_name_not_in_index}{framework_file}.csv"
)  # not in index
data_in_index = pd.read_csv(f"{output_folder}/{file_name}{index_file}.csv")

# %%
print(len(data_not_in_index))
print(len(data_in_index))

# %%
data_not_in_index = data_not_in_index[
    [
        "documents",
        "corrected_spelling",
        "label",
        "codable",
        "llm_soc_code",
        "llm_soc_candidates",
        "reasoning",
    ]
]
data_in_index = data_in_index[
    [
        "documents",
        "corrected_spelling",
        "label",
        "codable",
        "llm_soc_code",
        "llm_soc_candidates",
        "reasoning",
    ]
]


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
data_not_in_index["llm_soc_candidates"] = data_not_in_index["llm_soc_candidates"].map(
    parse_string
)
data_in_index["llm_soc_candidates"] = data_in_index["llm_soc_candidates"].map(
    parse_string
)

# %%
print(f"index: {data_not_in_index["codable"].value_counts()}")
print(f"framework: {data_in_index["codable"].value_counts()}")


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
        return codes_list
    return row_values


# %%
data_not_in_index["label"] = data_not_in_index["label"].astype(str)
data_in_index["label"] = data_in_index["label"].astype(str)

# %%
msk_not_index = data_not_in_index[
    "llm_soc_code"
].isna()  # take rows, where LLM didn't provide a code.
msk_index = data_in_index[
    "llm_soc_code"
].isna()  # take rows, where LLM didn't provide a code.

# %%
data_not_in_index.loc[~msk_not_index, "llm_soc_code"] = data_not_in_index.loc[
    ~msk_not_index, "llm_soc_code"
].apply(float_to_list_of_codes)
data_in_index.loc[~msk_index, "llm_soc_code"] = data_in_index.loc[
    ~msk_index, "llm_soc_code"
].apply(float_to_list_of_codes)

# %%
data_not_in_index.loc[msk_not_index, "llm_soc_code"] = data_not_in_index.loc[
    msk_not_index, "llm_soc_candidates"
].apply(access_soc_code_from_candidate_list)
data_in_index.loc[msk_index, "llm_soc_code"] = data_in_index.loc[
    msk_index, "llm_soc_candidates"
].apply(access_soc_code_from_candidate_list)


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
        f"Argeement full {df_source}: {agr} ({round(agr/len(df), 2) * 100}% of all rows)"
    )
    print(
        f"Argeement (code in candidates) {df_source}: {in_cand} ({round(in_cand/len(df), 2) * 100}% of all rows)"  # pylint: disable=C0301
    )
    print(
        f"Argeement (label the same or within candidates) {df_source}: {agr + in_cand} ({round((agr + in_cand)/len(df), 2) * 100}% of all rows)"  # pylint: disable=C0301
    )


# %%
check_agreement(data_not_in_index, "ASHE - not in index")
check_agreement(data_in_index, "SOC index rows")


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
check_code_count(data_not_in_index, "ASHE - not in index")
check_code_count(data_in_index, "SOC index rows")

# %%
data_in_index["corrected_spelling"] = data_in_index["documents"]

# %%
full_data_post_cleaning = pd.concat(
    [data_not_in_index, data_in_index], ignore_index=True
)

# %%
full_data_codable = full_data_post_cleaning[full_data_post_cleaning["codable"]]

# %%
data_only_columns = full_data_post_cleaning[["corrected_spelling", "llm_soc_code"]]

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


# %%
knowledge_bucket = dotenv.get_key("../.env", "KNOWLEDGE_BUCKET")

# %%
s_list = load_soc_framework(
    f"{knowledge_bucket}soc2020volume2thecodingindexexcel03122025.xlsx"
)
s_list = s_list[s_list["code"].notna()]

# %%
codes_from_framework_str = list(s_list["code"].value_counts().keys())

# %%
codes_from_framework_int = []
for i in codes_from_framework_str:
    codes_from_framework_int.append(int(i))

# %%
phantom_codes = (
    data_one_code[~data_one_code["label"].isin(codes_from_framework_int)]["label"]
    .value_counts()
    .keys()
)

# %%
print(phantom_codes)

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
    data_one_code_no_phantoms[
        data_one_code_no_phantoms.duplicated(subset=["documents"])
    ]
)

# %%
print(
    data_one_code_no_phantoms[data_one_code_no_phantoms["documents"] == "BIKE MECHANIC"]
)

# %%
data_one_code_no_phantoms = data_one_code_no_phantoms.drop_duplicates(
    subset=["documents"], keep="last", ignore_index=True
)

# %%
# data_one_code_no_phantoms.to_csv("soc_data/SOC_DIRECT_LOOKUP.csv")

# %%
# data_one_code_no_phantoms.to_csv(f"{knowledge_bucket}SOC_DIRECT_LOOKUP.csv")
