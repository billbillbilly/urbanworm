import pytest
from urbanworm.UrbanDataSet import UrbanDataSet
from urbanworm.utils import response2gdf


def test_export_geojson_like_issue():
    """Test export() with empty from_loopUnitChat to simulate ValueError scenario"""
    dataset = UrbanDataSet()

    fake_qna = []
    dataset.results = {
        'from_loopUnitChat': fake_qna,
        'base64_imgs': {},
        'coords': [],
        'meta': []
    }

    with pytest.raises(ValueError, match="No response found in the input dictionary"):
        dataset.export(file_name="./test_output_issue.geojson", out_type='geojson')

def test_response2gdf_empty_input():
    """Test response2gdf with completely empty input"""
    with pytest.raises(ValueError):
        response2gdf([])


def test_response2gdf_nested_empty_input():
    """Test response2gdf with nested empty input"""
    with pytest.raises(ValueError):
        response2gdf([[]])


def test_response2gdf_missing_nesting():
    """Test response2gdf with missing level of nesting"""
    with pytest.raises(ValueError):
        response2gdf([["Q", "A"]])
