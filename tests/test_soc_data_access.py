"""Unit tests for the SOC data access utility functions.

This module contains tests for ``load_soc_index`` and ``load_soc_structure``
(``occupational_classification_utils.utils.soc_data_access``), mirroring
``test_sic_data_access.py``.
"""

from unittest.mock import ANY, patch

import pandas as pd
import pytest

from occupational_classification_utils.utils.soc_data_access import (
    _merge_structure_with_index_codes,
    load_soc_hierarchy,
    load_soc_index,
    load_soc_structure,
)


@pytest.fixture
def mock_soc_index_excel_frame():
    """Raw workbook-shaped frame before normalisation in ``load_soc_index``."""
    return pd.DataFrame(
        {
            "SOC_2020": ["2314"],
            "INDEXOCC_-_natural_word_order": ["primary teacher"],
            "ADD": [""],
            "IND": [""],
        }
    )


@pytest.fixture
def mock_soc_structure_excel_frame():
    """Raw workbook-shaped frame for structure columns."""
    return pd.DataFrame(
        {
            "SOC\n2020 Major Group": ["2"],
            "SOC\n2020 Sub-Major Group": ["23"],
            "SOC\n2020 Minor Group": ["231"],
            "SOC 2020 Unit Group": ["2314"],
        }
    )


@pytest.mark.utils
@patch("pandas.read_excel")
def test_load_soc_index(mock_read_excel, mock_soc_index_excel_frame):
    mock_read_excel.return_value = mock_soc_index_excel_frame
    result = load_soc_index(
        (
            "occupational_classification_utils.data.soc_index",
            "soc2020volume2thecodingindexexcel16042025.xlsx",
        )
    )
    mock_read_excel.assert_called_once_with(
        ANY,
        sheet_name="SOC2020 coding index",
        usecols=["SOC_2020", "INDEXOCC_-_natural_word_order", "ADD", "IND"],
        dtype=str,
    )
    called_args, _ = mock_read_excel.call_args
    assert str(called_args[0]).endswith("soc2020volume2thecodingindexexcel16042025.xlsx")
    assert list(result.columns) == ["code", "title"]
    assert result["code"].iloc[0] == "2314"


@pytest.mark.utils
@patch("pandas.read_excel")
def test_load_soc_structure(mock_read_excel, mock_soc_structure_excel_frame):
    mock_read_excel.return_value = mock_soc_structure_excel_frame
    result = load_soc_structure(
        (
            "occupational_classification_utils.data.soc_index",
            "soc2020volume1structureanddescriptionofunitgroupsexcel16042025.xlsx",
        )
    )
    mock_read_excel.assert_called_once_with(
        ANY,
        sheet_name="SOC2020 descriptions",
        usecols=[
            "SOC\n2020 Major Group",
            "SOC\n2020 Sub-Major Group",
            "SOC\n2020 Minor Group",
            "SOC 2020 Unit Group",
        ],
        dtype=str,
    )
    called_args, _ = mock_read_excel.call_args
    assert str(called_args[0]).endswith(
        "soc2020volume1structureanddescriptionofunitgroupsexcel16042025.xlsx"
    )
    assert list(result.columns) == ["code"]
    assert set(result["code"]) == {"2", "23", "231", "2314"}


@pytest.mark.utils
def test_merge_structure_with_index_codes_adds_prefixes():
    struct = pd.DataFrame({"code": ["2", "23"]})
    index = pd.DataFrame({"code": ["2314"]})
    merged = _merge_structure_with_index_codes(struct, index)
    assert set(merged["code"]) == {"2", "23", "231", "2314"}


@pytest.mark.utils
@patch("occupational_classification_utils.utils.soc_data_access.load_soc_structure")
@patch("occupational_classification_utils.utils.soc_data_access.load_soc_index")
def test_load_soc_hierarchy_merges_index_prefixes(mock_idx, mock_struct):
    mock_idx.return_value = pd.DataFrame({"code": ["1111"], "title": ["Chief exec"]})
    mock_struct.return_value = pd.DataFrame({"code": ["1", "11", "111"]})
    soc = load_soc_hierarchy(
        ("occupational_classification_utils.data.soc_index", "a.xlsx"),
        ("occupational_classification_utils.data.soc_index", "b.xlsx"),
    )
    assert "1111" in soc.lookup
    assert soc["1111"].job_titles
