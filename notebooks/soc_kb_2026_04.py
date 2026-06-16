# %%
# pylint: disable=C0103, C0114

# %%
import dotenv
import pandas as pd

# %%
knowledge_bucket = dotenv.get_key(".env", "KNOWLEDGE_BUCKET")
sa_dev = dotenv.get_key(".env", "SA_DEV")
sandbox = dotenv.get_key(".env", "SA_SANDBOX")

# %%
data = pd.read_csv(f"{knowledge_bucket}ASHE_classifai_soc_kb.csv", dtype={"label": str})

# %%
data = data.rename(columns={"documents": "text"})

# %%
data = data[['text','label']].copy()

# %%
data['text'] = data['text'].str.strip()

# %%
# data.to_csv(f"{sa_dev}soc_vector_store_config/data/soc_kb_for_classifai.csv", index=False)
# data.to_csv(f"{sandbox}soc_vector_store_config/data/soc_kb_for_classifai.csv", index=False)

# %%
# data.to_csv(f"{knowledge_bucket}soc_kb_for_classifai.csv", index=False)


