import pytest
from shapely.geometry import Point
from urbanworm.utils import response2gdf
from urbanworm.UrbanDataSet import QnA

# QnA generate
def make_qna(question, answer, explanation):
    return QnA(question=question, answer=answer, explanation=explanation)

def test_response2gdf_normal_single():
    """
    Test response2gdf with normal single-response top and street view
    """
    qna_dict = {
        'lon': [-83.1],
        'lat': [42.3],
        'top_view': [[
            make_qna("Top question?", "Yes", "Because it's clear.")
        ]],
        'street_view': [[
            make_qna("Street question?", "No", "Blocked by tree.")
        ]]
    }

    gdf = response2gdf(qna_dict)
    assert len(gdf) == 1
    assert "top_view_question1" in gdf.columns
    assert "street_view_question1" in gdf.columns

def test_response2gdf_multi_image_input():
    """
    Test response2gdf with multiImgInput=True structure (nested list)
    """
    qna_dict = {
        'lon': [-83.2],
        'lat': [42.4],
        'street_view': [
            [[
                make_qna("Q1?", "A1", "E1"),
                make_qna("Q2?", "A2", "E2")
            ], [
                make_qna("Q1?", "A3", "E3"),
                make_qna("Q2?", "A4", "E4")
            ]]
        ]
    }

    gdf = response2gdf(qna_dict)
    assert len(gdf) == 1
    assert "street_view_question1" in gdf.columns

def test_response2gdf_empty_street_view():
    """
    Ensure empty or malformed street_view doesn't crash
    """
    qna_dict = {
        'lon': [-83.3],
        'lat': [42.5],
        'street_view': [[]]  # Simulate empty response from loopUnitChat
    }

    gdf = response2gdf(qna_dict)
    assert len(gdf) == 1
    # Should not throw IndexError even if no street_view columns

def test_response2gdf_missing_all_views():
    """
    If no 'top_view' or 'street_view' is present, raise error
    """
    qna_dict = {'lon': [-83.4], 'lat': [42.6]}
    with pytest.raises(ValueError):
        response2gdf(qna_dict)
