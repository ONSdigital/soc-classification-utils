"""Unit tests for the SOC data access utility functions."""

from unittest.mock import ANY, patch

import pandas as pd
import pytest

from occupational_classification_utils.utils.soc_data_access import (
    load_soc_hierarchy,
    load_soc_index,
    load_soc_structure,
)

# Mirror SIC test lint posture for fixture-injected test arguments.
# pylint: disable=redefined-outer-name
# pylint: disable=duplicate-code


@pytest.fixture
def soc_index_workbook_ref():
    """Return config-style tuple for the SOC index workbook."""
    return (
        "occupational_classification_utils.data.soc_index",
        "soc2020volume2thecodingindexexcel16102024.xlsx",
    )


@pytest.mark.utils
@patch("pandas.read_excel")
def test_load_soc_index(mock_read_excel, soc_index_workbook_ref):
    """SOC index loader reads workbook content and normalises columns."""
    mock_read_excel.return_value = pd.DataFrame(
        {
            "SOC_2020": ["2314", "4111"],
            "INDEXOCC_-_natural_word_order": [
                "Teacher, primary",
                "Investigator, benefits fraud",
            ],
            "ADD": [None, "Senior"],
            "IND": [None, "Fraud team"],
        }
    )
    result = load_soc_index(soc_index_workbook_ref)

    mock_read_excel.assert_called_once_with(
        ANY,
        sheet_name="SOC2020 coding index",
        usecols=["SOC_2020", "INDEXOCC_-_natural_word_order", "ADD", "IND"],
        dtype=str,
    )
    called_args, _ = mock_read_excel.call_args
    assert str(called_args[0]).endswith(
        "soc2020volume2thecodingindexexcel16102024.xlsx"
    )

    assert list(result.columns) == ["code", "title"]
    assert set(result["code"]) == {"2314", "4111"}
    assert "Teacher, primary" in result["title"].tolist()


@pytest.mark.utils
@patch("pandas.read_excel")
def test_load_soc_structure(mock_read_excel):
    """SOC structure loader derives hierarchy prefixes from workbook codes."""
    soc_structure_workbook_ref = (
        "occupational_classification_utils.data.soc_index",
        "soc2020volume1structureanddescriptionofunitgroupsexcel16102024.xlsx",
    )
    mock_read_excel.return_value = pd.DataFrame(
        {
            "SOC\n2020 Major Group": ["2", "4"],
            "SOC\n2020 Sub-Major Group": ["23", "41"],
            "SOC\n2020 Minor Group": ["231", "411"],
            "SOC 2020 Unit Group": ["2314", "4111"],
        }
    )
    result = load_soc_structure(soc_structure_workbook_ref)

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
        "soc2020volume1structureanddescriptionofunitgroupsexcel16102024.xlsx"
    )

    assert list(result.columns) == ["code"]
    assert {"2", "23", "231", "2314", "4", "41", "411", "4111"} <= set(result["code"])


@pytest.mark.utils
@patch("pandas.read_excel")
def test_load_soc_hierarchy_workbook_resources(mock_read_excel, soc_index_workbook_ref):
    """Full hierarchy load works from workbook refs without direct file IO in test."""
    soc_structure_workbook_ref = (
        "occupational_classification_utils.data.soc_index",
        "soc2020volume1structureanddescriptionofunitgroupsexcel16102024.xlsx",
    )
    mock_read_excel.side_effect = [
        pd.DataFrame(
            {
                "SOC_2020": ["2314", "4111"],
                "INDEXOCC_-_natural_word_order": [
                    "Teacher, primary",
                    "Investigator, benefits fraud",
                ],
                "ADD": [None, None],
                "IND": [None, None],
            }
        ),
        pd.DataFrame(
            {
                "SOC\n2020 Major Group": ["2", "4"],
                "SOC\n2020 Sub-Major Group": ["23", "41"],
                "SOC\n2020 Minor Group": ["231", "411"],
                "SOC 2020 Unit Group": ["2314", "4111"],
            }
        ),
    ]
    soc = load_soc_hierarchy(soc_index_workbook_ref, soc_structure_workbook_ref)
    assert "2314" in soc.lookup
    assert soc["2314"].job_titles
