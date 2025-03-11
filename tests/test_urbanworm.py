#!/usr/bin/env python

import os
import unittest

from urbanworm import UrbanDataSet

class TestUrbanWorm(unittest.TestCase):
    """Tests for `urbanworm` package."""   
    def setUp(self):
        """Set up test fixtures, if any."""
        bbox = [-83.144079,42.356126,-83.143773,42.356229]
        image = "satellite.tif"
        samgeo.tms_to_geotiff(
            output=image, bbox=bbox, zoom=15, source="Satellite", overwrite=True
        )
        self.source = image

        out_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        checkpoint = os.path.join(out_dir, "sam_vit_h_4b8939.pth")
        self.checkpoint = checkpoint

        sam = samgeo.SamGeo(
            model_type="vit_h",
            checkpoint=checkpoint,
            sam_kwargs=None,
        )

        self.sam = sam