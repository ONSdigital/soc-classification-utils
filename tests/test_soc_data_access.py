"""Unit tests for the SOC data access utility functions."""

import pandas as pd
import pytest

from occupational_classification_utils.utils.soc_data_access import (
    load_soc_hierarchy,
    load_soc_index,
    load_soc_structure,
)

# Mirror SIC test lint posture for fixture-injected test arguments.
# pylint: disable=redefined-outer-name


@pytest.fixture
def csv_resource_ref(tmp_path):
    """Create a temporary lookup CSV and return config-style resource tuple."""
    csv_path = tmp_path / "mock_soc_lookup.csv"
    pd.DataFrame(
        {
            "label": ["2314", "4111"],
            "description": ["primary teacher", "benefits fraud investigator"],
        }
    ).to_csv(csv_path, index=False)
    return (str(tmp_path), "mock_soc_lookup.csv")


@pytest.mark.utils
def test_load_soc_index(csv_resource_ref):
    """SOC index loader reads library-shaped lookup CSV and normalises columns."""
    result = load_soc_index(csv_resource_ref)
    assert list(result.columns) == ["code", "title"]
    assert set(result["code"]) == {"2314", "4111"}
    assert "Primary teacher" in result["title"].tolist()


@pytest.mark.utils
def test_load_soc_structure(csv_resource_ref):
    """SOC structure loader derives hierarchy prefixes from CSV label codes."""
    result = load_soc_structure(csv_resource_ref)
    assert list(result.columns) == ["code"]
    assert {"2", "23", "231", "2314", "4", "41", "411", "4111"} <= set(result["code"])


@pytest.mark.utils
def test_load_soc_hierarchy_csv_resources(csv_resource_ref):
    """Full hierarchy load works from CSV refs without Excel assets."""
    soc = load_soc_hierarchy(csv_resource_ref, csv_resource_ref)
    assert "2314" in soc.lookup
    assert soc["2314"].job_titles
