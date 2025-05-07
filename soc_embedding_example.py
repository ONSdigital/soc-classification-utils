"""This scripts shows how EmbeddingHandler class is used."""

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

# %%
from occupational_classification_utils.embed.embedding import (
    EmbeddingHandler,
)

# %%

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
