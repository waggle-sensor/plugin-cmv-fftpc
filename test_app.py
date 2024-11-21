import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from app import (
    main,
    initialize_plugin,
    process_frame,
    compute_optical_flow,
    threshold_and_publish,
    create_segments,
    analyze_and_publish_segments,
    upload_image,
)

class TestOpticalFlowApp(unittest.TestCase):

    def setUp(self):
        # Mock arguments
        self.args = MagicMock()
        self.args.input = "mock_input"
        self.args.i = 0
        self.args.c = 0  # Red channel
        self.args.k = 0.9
        self.args.q = 2
        self.args.thr = 50
        self.args.segments = 10
        self.args.seg_pub = 3
        self.args.oneshot = True


    def test_compute_optical_flow(self):
        """Test calculation of optical flow between two frames."""
        sky_prev = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        shift = 5
        sky_curr = np.roll(sky_prev, shift=shift, axis=1)
        inf = {
            "winsize": 15,
            "poly_n": 5,
            "poly_s": 1.2,
        }
        vel_factor = 60 / 1.0  # Mocked time difference
        flow, mag, ang = compute_optical_flow(sky_prev, sky_curr, inf, vel_factor)
        self.assertEqual(flow.shape, (128, 128, 2))
        self.assertEqual(mag.shape, (128, 128))
        self.assertEqual(ang.shape, (128, 128))

    @patch("app.threshold_otsu", return_value=5)
    def test_threshold_and_publish(self, mock_threshold_otsu):
        """Test thresholding and publishing of optical flow magnitude."""
        mag = np.random.rand(128, 128)
        plugin = MagicMock()
        sample = MagicMock()
        sample.timestamp = 1_000_000_010

        # Call the function
        should_upload, thres_otsu = threshold_and_publish(mag, plugin, sample, 4)

        # Assertions
        self.assertTrue(should_upload)
        self.assertEqual(thres_otsu, 5)
        plugin.publish.assert_called_with("cmv.thresh.otsu", 5.0, timestamp=sample.timestamp)

    def test_create_segments(self):
        """Test segment creation from magnitude and angle."""
        mag = np.random.rand(128, 128).astype(np.float32)
        ang = np.random.rand(128, 128).astype(np.float32)
        segments, superpixels = create_segments(np.zeros((128, 128, 3)), mag, ang, 10)
        self.assertEqual(segments.shape, (128, 128))
        self.assertEqual(superpixels.shape, (128, 128, 3))

    @patch("app.Plugin")
    def test_analyze_and_publish_segments(self, MockPlugin):
        """Test analyzing segments and publishing their metrics."""
        segments = np.random.randint(0, 10, (128, 128))
        mag = np.random.rand(128, 128)
        ang = np.random.rand(128, 128)
        plugin = MockPlugin.return_value
        sample = MagicMock()
        sample.timestamp = 1_000_000_010

        # Call the function
        analyze_and_publish_segments(segments, mag, ang, plugin, sample, self.args)

        # Assertions
        self.assertTrue(plugin.publish.called)
        publish_calls = plugin.publish.call_args_list
        topics = [call[0][0] for call in publish_calls]

        # Verify that the expected topics are published
        self.assertIn("cmv.motion.detected", topics)

    @patch("cv2.imwrite")
    def test_upload_image(self, mock_imwrite):
        """Test uploading an image with metadata."""
        plugin = MagicMock()
        sky_curr = np.random.rand(128, 128, 3) * 255
        timestamp = 1_000_000_010
        thres_otsu = 5

        # Call the function
        upload_image(sky_curr, timestamp, thres_otsu, plugin)

        # Assertions
        mock_imwrite.assert_called_once()
        plugin.upload_file.assert_called_once_with(
            f"img2_{timestamp}.jpg", meta={"thres_otsu": "5"}, timestamp=timestamp
        )


if __name__ == "__main__":
    unittest.main()
