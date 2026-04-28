# %%
import dotenv
import pandas as pd

# %%
knowledge_bucket = dotenv.get_key("../.env", "KNOWLEDGE_BUCKET")

# %%
rephrased = pd.read_csv("soc_data/ashe_correct_spelling_2026_04_20.csv")
in_index = pd.read_csv("soc_data/ashe_in_soc_index_2026_04_20.csv")

# %%
in_index['corrected_spelling'] = in_index['documents']

# %%
full_data = pd.concat([rephrased, in_index], ignore_index=True)

# %%
soc_kb = full_data[['corrected_spelling', 'label']]

# %%
soc_kb = soc_kb.rename(columns={'corrected_spelling': 'text'})

# %%
soc_kb['text'] = soc_kb['text'].str.capitalize()

# %%
soc_kb = soc_kb.drop_duplicates(subset=['text', 'label'], keep="last", ignore_index=True)

# %%
# soc_kb.to_csv(f"{knowledge_bucket}SOC_KB.csv")

# %%
mask = soc_kb.groupby('text')['label'].transform('nunique') > 1

# %%
conflict_codes = soc_kb[mask].sort_values('text')

# %%
conflict_codes


