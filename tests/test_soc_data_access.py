"""Unit tests for the SOC data access utility functions.

This module contains tests for the `load_soc_index` and `load_soc_structure`
functions from the `occupational_classification_utils.utils.soc_data_access` module.
"""

from unittest.mock import ANY, patch

import pandas as pd
import pytest

from occupational_classification_utils.utils.soc_data_access import (
    load_soc_index,
    load_soc_structure,
)

# pylint: disable=redefined-outer-name
# pylint: disable=duplicate-code


@pytest.fixture
def mock_soc_index_data():
    """Fixture for mock SOC index data."""
    return pd.DataFrame(
        {
            "soc_code": ["2314", "2313"],
            "title": ["Primary teacher", "Secondary teacher"],
        }
    )


@pytest.fixture
def mock_soc_structure_data():
    """Fixture for mock SOC structure data."""
    return pd.DataFrame(
        {
            "description": ["Major group", "Minor group"],
            "section": ["1", "2"],
            "most_disaggregated_level": ["Level 1", "Level 2"],
            "level_headings": ["Heading 1", "Heading 2"],
        }
    )


@pytest.mark.utils
@patch("occupational_classification_utils.utils.soc_data_access._lib_load_soc_index")
def test_load_soc_index(mock_load_soc_index, mock_soc_index_data):
    """Test the `load_soc_index` function.

    Asserts:
        - The underlying library loader is called with a resolved file path.
        - The returned DataFrame matches the mock SOC index data.
    """
    mock_load_soc_index.return_value = mock_soc_index_data

    result = load_soc_index(
        (
            "occupational_classification_utils",
            "data/soc_index/soc2020volume2thecodingindexexcel16102024.xlsx",
        )
    )

    mock_load_soc_index.assert_called_once_with(ANY)

    # Verify the path used in the call
    called_args, _ = mock_load_soc_index.call_args
    assert str(called_args[0]).endswith(
        "soc2020volume2thecodingindexexcel16102024.xlsx"
    )
    assert result.equals(mock_soc_index_data)


@pytest.mark.utils
@patch(
    "occupational_classification_utils.utils.soc_data_access._lib_load_soc_structure"
)
def test_load_soc_structure(mock_load_soc_structure, mock_soc_structure_data):
    """Test the `load_soc_structure` function.

    Asserts:
        - The underlying library loader is called with a resolved file path.
        - The returned DataFrame matches the mock SOC structure data.
    """
    mock_load_soc_structure.return_value = mock_soc_structure_data

    result = load_soc_structure(
        (
            "occupational_classification_utils",
            "data/soc_index/"
            "soc2020volume1structureanddescriptionofunitgroupsexcel16102024.xlsx",
        )
    )

    mock_load_soc_structure.assert_called_once_with(ANY)

    # Verify the path used in the call
    called_args, _ = mock_load_soc_structure.call_args
    assert str(called_args[0]).endswith(
        "soc2020volume1structureanddescriptionofunitgroupsexcel16102024.xlsx"
    )

    assert result.equals(mock_soc_structure_data)
