"""Tests for SOC loading via the shared library module."""

from unittest.mock import patch

import pandas as pd
import pytest
from occupational_classification.data_access.soc_data_access import load_soc_hierarchy


@pytest.mark.utils
@patch("occupational_classification.data_access.soc_data_access.pd.read_excel")
def test_utils_can_load_soc_hierarchy_via_library_module(mock_read_excel):
    """Utils keeps working when hierarchy loading comes from the library."""
    soc_index_workbook_ref = (
        "occupational_classification_utils.data.soc_index",
        "soc2020volume2thecodingindexexcel16102024.xlsx",
    )
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
