# %%
# pylint: disable=C0103, C0114, C0301, R0801, W0105

"""Noetbook comparing two methods of accessing codes using LLM;
diretly from ashe, as used in the current verion of prompt,
and using the top_canidate.

Compare with create_soc_lookup_2026_04.ipynb.

Diasbling duplicate code - methods needs to be changed in other repos to reflect the change in data.
Diasbling line-too-long: commentary and discussion.
Disabling pointless-string-statement: comments to the code for reading clarity.
"""

# %%
import dotenv
import pandas as pd

# %%
knowledge_bucket = dotenv.get_key(".env", "KNOWLEDGE_BUCKET")

# %%
top_one_data = pd.read_parquet(
    f"{knowledge_bucket}wip_data/soc_kb_top_one_STG2.parquet"
)
ashe_data = pd.read_csv(
    f"{knowledge_bucket}ASHE_classifai_soc_kb.csv", dtype={"label": str}
)

# %%
data = pd.merge(
    ashe_data, top_one_data, left_on="ids", right_on="unique_id", how="inner"
)

# %%
# use only columns needed
data = data[
    [
        "ids",
        "documents",
        "label",
        "initial_code",
        "code_title",
        "likelihood",
        "reasoning",
    ]
]

# %%
data["likelihood"].value_counts()

# %%
print(
    f"""Initally coded {int(data['likelihood'].value_counts().get(0.9))} rows with likelihood score at 0.9.

Candidates are not available in the data provided.
"""
)


# %%
def get_agreement_level(row: pd.Series):
    """Compare the agreement between the label assigned by ASHE and label selected by the LLM.
    Check for mutual digit in the assigned codes.
    0 - no agreement
    1 - Major Group
    2 - Sub-Major Group
    3 - Minor Group
    4 - Unit Group (full agreemnt).

    Args:
        row (pd.Series): row with ASHE label and LLM label.

    Return:
        agreement_level (int): the depth of the agreement

    """
    ashe_code = row["label"]
    llm_code = row["initial_code"]

    agreement_level = 0

    for digit in llm_code:
        if digit == ashe_code[agreement_level]:
            agreement_level += 1
    return agreement_level


# %%
data.loc[:, "agreement_level"] = data.apply(get_agreement_level, axis=1)

# %%
msk09 = data["likelihood"] == 0.9  # noqa: PLR2004
msk08 = data["likelihood"] == 0.8  # noqa: PLR2004
msk06 = data["likelihood"] == 0.6  # noqa: PLR2004
msk04 = data["likelihood"] == 0.4  # noqa: PLR2004
msk02 = data["likelihood"] == 0.2  # noqa: PLR2004

# %% [markdown]
# ### Agreemnet between ASHE and LLM codes
# Agreement defined as "ASHE and LLM assign the same code".
# Disagreement defined as "Codes assigned using ASHE and LLM differ".

# %%
agreemnt_cond = data["agreement_level"] == 4  # noqa: PLR2004

# %%
# check agreement between ASHE and LLM
agreement09 = data[agreemnt_cond & msk09]
disagreement09 = data[~agreemnt_cond & msk09]

# %%
print(
    f"""{len(agreement09)} rows had agreement between ASHE and LLM (same code) with likelihood 0.9.
{len(disagreement09)} rows had disagreement, between ASHe and LLM, despite LLM being highly confident in it's selection (likelihood=0.9).

The disagreement appered in {len(disagreement09)/len(data[msk09]) * 100:.3}% of cases with likelihood=0.9."""
)

# %%
# check agreement between ASHE and LLM
agreement08 = data[agreemnt_cond & msk08]
disagreement08 = data[~agreemnt_cond & msk08]

# %%
print(
    f"""{len(agreement08)} rows had agreement between ASHE and LLM (same code) with likelihood 0.8.
{len(disagreement08)} rows had disagreement, between ASHE and LLM, despite LLM being highly confident in it's selection (likelihood=0.8).
The disagreement appered in {len(disagreement08)/len(data[msk08]) * 100:.3}% of cases with likelihood=0.8.

When the LLM is less confident in it's decision, i.e. the likelihood score drops from 0.9 to 0.8,
the disagreement {"increases" if len(disagreement08)/len(data[msk08]) > len(disagreement09)/len(data[msk09]) else "decreases"} (LLM is less confident => worse results).
"""
)

# %%
# check agreement between ASHE and LLM for specific likelihoods
agreement06 = data[agreemnt_cond & msk06]
disagreement06 = data[~agreemnt_cond & msk06]

agreement04 = data[agreemnt_cond & msk04]
disagreement04 = data[~agreemnt_cond & msk04]

agreement02 = data[agreemnt_cond & msk02]
disagreement02 = data[~agreemnt_cond & msk02]

# %%
print(
    f"""When comparing the disagreements for decreasing likelihood scores, it is expected the disagreement between LLM and ASHE inscreases.
We observe that:
- disagreement {"increases" if len(disagreement06)/len(data[msk06]) > len(disagreement08)/len(data[msk08]) else "decreases"} when the likelihood score drops from 0.8 to 0.6,
- disagreement {"increases" if len(disagreement04)/len(data[msk04]) > len(disagreement06)/len(data[msk06]) else "decreases"} when the likelihood score drops from 0.6 to 0.4,
- disagreement {"increases" if len(disagreement02)/len(data[msk02]) > len(disagreement04)/len(data[msk04]) else "decreases"} when the likelihood score drops from 0.4 to 0.2.
"""
)

# %%
# count the rows, where some agreement between ASHe and LLM is present (count of rows with likelihood x - count of rows with the agreement_level == 0)
partial_agreement09 = int(
    len(data[msk09]) - data[msk09]["agreement_level"].value_counts().get(0)
)
partial_agreement08 = int(
    len(data[msk08]) - data[msk08]["agreement_level"].value_counts().get(0)
)
partial_agreement06 = int(
    len(data[msk06]) - data[msk06]["agreement_level"].value_counts().get(0)
)
partial_agreement04 = int(
    len(data[msk04]) - data[msk04]["agreement_level"].value_counts().get(0)
)
partial_agreement02 = int(
    len(data[msk02]) - data[msk02]["agreement_level"].value_counts().get(0)
)

# %% [markdown]
# ### Partial agreement

# %%
print(
    f"""It is expected that in some cases, ASHE and LLM will return close, but not full matches,
e.g. assign a set of responses to the same Minor Group (3 digit code agreement), but different Unit Group (4 digit code agreement).

Simillarly as above, we expect higher percentage of close matches when the LLM returns high likelihood score.
We found partial agreements between ASHE and LLM in:
- {round(partial_agreement09 / len(data[msk09]) * 100, 1)}% of cases with likelihood 0.9,
- {round(partial_agreement08 / len(data[msk08]) * 100, 1)}% of cases with likelihood 0.8,
- {round(partial_agreement06 / len(data[msk06]) * 100, 1)}% of cases with likelihood 0.6,
- {round(partial_agreement04 / len(data[msk04]) * 100, 1)}% of cases with likelihood 0.4,
- {round(partial_agreement02 / len(data[msk02]) * 100, 1)}% of cases with likelihood 0.2.
"""
)

# %% [markdown]
# ### High likelihood, but disagreement

# %%
dis09agr3 = disagreement09[
    disagreement09["agreement_level"] == 3  # noqa: PLR2004
].reset_index(drop=True)

# %%
print(
    f"""Example of high likelihood assigned by the LLM, but with a disagreement with ASHE:

{dis09agr3.iloc[12]}

Reading further in the "reasoning", the LLM admits it is missing information:
    "Without further information, 'specialist nurse' is a better fit for the 'Accident & emergency' context."

suggesting that more information is required, which invalidates the requirements for the likelihood=0.9.

"""
)

# %%
# pd.options.display.max_colwidth = 100

# print(f"""{dis09agr3.iloc[6]['reasoning']}""")

# %%
# dis09agr3.iloc[6]

# %%
# check if the LLM is hesitant with assigning the n.e.c. codes (XXX9 codes) adn likelihood 0.9???

# %%
print(data[data["initial_code"].str.len() < 4]["reasoning"])  # noqa: PLR2004

# %%
nines_llm = int(
    data[msk09]["initial_code"].apply(lambda x: 1 if x[3] == "9" else 0).sum()
)
nines_ashe = int(data[msk09]["label"].apply(lambda x: 1 if x[3] == "9" else 0).sum())

# %%
print(
    f"""The LLM assigns the "Not Elsewhere Classified" (n.e.c.) codes only in {nines_llm/len(data[msk09]) * 100:.3}% of codes, when high likelihood (0.9),
while ASHE assigns this type of codes to {nines_ashe/len(data[msk09]) * 100:.3}% of codes,
suggesting that LLM is hesitant to assign nec code and be highly confident."""
)


# %%
print(
    f"""Very interesting case (the only one like that).
No code was assigned, the likelihood set to 0.2

{data.iloc[63932]}

Full reasoning:
    {data.iloc[63932]['reasoning']}

"""
)
