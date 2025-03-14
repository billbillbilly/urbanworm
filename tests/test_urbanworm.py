#!/usr/bin/env python

import unittest
import geopandas as gpd
from unittest.mock import patch, MagicMock
from urbanworm import UrbanDataSet

class TestUrbanDataSet(unittest.TestCase):
    """Unit tests for `UrbanDataSet`."""

    @patch("ollama.list")
    @patch("ollama.pull") 
    def test_preload_model(self, mock_pull, mock_list):
        """Test that the model preloads correctly."""
        mock_list.return_value = {"models": []}  # Simulate no model installed

        dataset = UrbanDataSet()
        dataset.preload_model()

        mock_pull.assert_called_with("llama3.2-vision")  # Ensures model pull happens

    @patch("utils.getOSMbuildings", return_value=MagicMock(empty=False)) 
    def test_bbox2osmBuildings(self, mock_getOSMbuildings):
        """Test extracting buildings from a bounding box."""
        dataset = UrbanDataSet()
        result = dataset.bbox2osmBuildings((-83.235572,42.348092,-83.235154,42.348806))

        self.assertIsInstance(result, gpd.GeoDataFrame)  # Ensures a valid response
        self.assertIn("buildings found", result)  # Confirms buildings detected

    @patch("ollama.chat")
    def test_oneImgChat(self, mock_chat):
        """Test that `oneImgChat()` calls Ollama correctly."""
        mock_chat.return_value = {"message": {"content": "Mocked Response"}}

        dataset = UrbanDataSet(image="test.jpg")
        response = dataset.oneImgChat(system="Describe", prompt="What is in this image?")

        mock_chat.assert_called_once()  # Ensures it calls Ollama
        self.assertEqual(response, "Mocked Response")  # Confirms expected response

    @patch("utils.getSV", return_value=["streetview.jpg"])
    @patch("ollama.chat")
    def test_loopUnitChat(self, mock_chat, mock_getSV):
        """Test processing multiple units for chat with street view."""
        dataset = UrbanDataSet()
        dataset.bbox2osmBuildings((-83.235572,42.348092,-83.235154,42.348806))

        # Simulate Ollama chat returning structured responses
        mock_chat.return_value = {"message": {"content": "Street view analysis"}}

        result = dataset.loopUnitChat(
            system="Analyze this area.",
            prompt={"street": "Is the wall damaged?"},
            type="top",
            epsg=2253
        )

        self.assertIsInstance(result, dict)  # Should return a dictionary
        self.assertIn("street_view", result)  # Key should exist
        self.assertGreater(len(result["street_view"]), 0)  # Should contain responses

if __name__ == "__main__":
    unittest.main()
