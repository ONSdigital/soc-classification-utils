# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: soc-classification-utils-NytDGhVa-py3.12
#     language: python
#     name: python3
# ---

# %%
from occupational_classification.hierarchy.soc_hierarchy import SOC, load_hierarchy
from occupational_classification.meta.socDB import soc_meta

# %%
from occupational_classification_utils.utils.soc_data_access import (
    load_soc_index,
    load_soc_structure,
)

# %%
from occupational_classification_utils.embed.embedding import get_config, EmbeddingHandler

# %%
get_config()["lookups"]["soc_index"]

# %%
soc_index = load_soc_index(get_config()["lookups"]["soc_index"])

# %%
soc_index.sample()

# %%
EXAMPLE_QUERY = "school teacher primary education"

print("Creating embeddings index...")
# Create the embeddings index
embed = EmbeddingHandler()
embed.embed_index(from_empty=True)
print(
    f"Embeddings index created with {embed._index_size} entries."  # pylint: disable=protected-access
)
results = embed.search_index(EXAMPLE_QUERY)
print(f"Search results for '{EXAMPLE_QUERY}': {results}")

# %%
