"""Provides data access for key SOC files.

This module contains utility functions to load and process data from
SOC-related Excel and text files. It accepts config-style (package, path) tuples
and resolves them via importlib.resources, then calls the existing SOC library
or reads text as needed. Filepaths are defined in the configuration in `embedding.py`.
"""

import logging
from importlib.resources import as_file, files
from typing import Union

import pandas as pd

from occupational_classification.data_access.soc_data_access import (
    load_soc_index as _lib_load_soc_index,
    load_soc_structure as _lib_load_soc_structure,
)
from occupational_classification.hierarchy.soc_hierarchy import SOC, load_hierarchy
from occupational_classification.meta.soc_meta import SocDB, SocMeta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type for config-style lookup: (package_name, path_within_package)
_ResourceRef = tuple[str, str]


def _resolve_path(ref_or_path: Union[_ResourceRef, str]) -> str:
    """Return a path string. If ref_or_path is a tuple, resolve via as_file."""
    if isinstance(ref_or_path, str):
        return ref_or_path
    pkg, name = ref_or_path
    with as_file(files(pkg).joinpath(name)) as path:
        return str(path)


def load_soc_index(resource_ref: Union[_ResourceRef, str]) -> pd.DataFrame:
    """Loads the SOC index from an Excel file.

    Accepts either a config-style tuple (package, path) or a path string.
    Resolves tuples via importlib.resources and calls the SOC library.

    Args:
        resource_ref: Either (package_name, path) or a path string.

    Returns:
        DataFrame containing the SOC index.
    """
    if isinstance(resource_ref, str):
        return _lib_load_soc_index(resource_ref)
    with as_file(
        files(resource_ref[0]).joinpath(resource_ref[1])
    ) as path:
        return _lib_load_soc_index(str(path))


def load_soc_structure(resource_ref: Union[_ResourceRef, str]) -> pd.DataFrame:
    """Loads the SOC structure from an Excel file.

    Accepts either a config-style tuple (package, path) or a path string.
    Resolves tuples via importlib.resources and calls the SOC library.

    Args:
        resource_ref: Either (package_name, path) or a path string.

    Returns:
        DataFrame containing the SOC structure.
    """
    if isinstance(resource_ref, str):
        return _lib_load_soc_structure(resource_ref)
    with as_file(
        files(resource_ref[0]).joinpath(resource_ref[1])
    ) as path:
        return _lib_load_soc_structure(str(path))


def load_soc_hierarchy(
    index_ref: Union[_ResourceRef, str],
    structure_ref: Union[_ResourceRef, str],
) -> SOC:
    """Loads the SOC hierarchy from index and structure refs.

    Resolves config-style tuples and calls the SOC library. Both refs are
    resolved inside a single context when both are tuples so that paths
    remain valid for load_hierarchy.

    Args:
        index_ref: Config tuple or path for the SOC index.
        structure_ref: Config tuple or path for the SOC structure.

    Returns:
        The loaded SOC hierarchy object.
    """
    if isinstance(index_ref, tuple) and isinstance(structure_ref, tuple):
        with as_file(
            files(index_ref[0]).joinpath(index_ref[1])
        ) as path1, as_file(
            files(structure_ref[0]).joinpath(structure_ref[1])
        ) as path2:
            soc_index_df = _lib_load_soc_index(str(path1))
            soc_df_input = _lib_load_soc_structure(str(path2))
            soc_df = SocDB.create_soc_dataframe(SocDB(soc_df_input).df)
            return load_hierarchy(
                soc_df,
                soc_index_df,
                structure_data_path=str(path2),
            )
    path1 = _resolve_path(index_ref)
    path2 = _resolve_path(structure_ref)
    soc_index_df = _lib_load_soc_index(path1)
    soc_df_input = _lib_load_soc_structure(path2)
    soc_df = SocDB.create_soc_dataframe(SocDB(soc_df_input).df)
    return load_hierarchy(soc_df, soc_index_df, structure_data_path=path2)


def get_soc_meta(structure_ref: Union[_ResourceRef, str]):
    """Returns SocMeta.soc_meta for the given structure ref.

    When structure_ref is a tuple, resolution and loading happen inside
    the resource context so the path is valid for the library.

    Args:
        structure_ref: Config tuple or path for the SOC structure.

    Returns:
        The soc_meta object from SocMeta(structure_path).soc_meta.
    """
    if isinstance(structure_ref, str):
        return SocMeta(structure_ref).soc_meta
    with as_file(
        files(structure_ref[0]).joinpath(structure_ref[1])
    ) as path:
        return SocMeta(str(path)).soc_meta


def load_text_from_config(resource_ref: _ResourceRef) -> str:
    """Loads text content from a configuration file.

    Accepts a config-style tuple (package, path) and resolves it via
    importlib.resources, matching the SIC data-access API.

    Args:
        resource_ref: Tuple (package_name, path) for the text file.

    Returns:
        The file content as a string.
    """
    pkg, filename = resource_ref
    file_path = files(pkg).joinpath(filename)
    logger.debug("Loading text from %s", file_path)
    with file_path.open(encoding="utf-8") as f:
        return f.read()
