"""Which OpenCV backend gets tried first, and why DSHOW keeps that place.

Measured on the same Windows UVC camera at 640x480, requesting 30 fps:

    CAP_DSHOW  opens in ~0.6 s, first frame in 1.0 s, every time
    CAP_MSMF   one run delivered 30 fps; the next took 37 s to a first frame
               and then delivered nothing at all, logging
               MF_E_HW_MFT_FAILED_START_STREAMING

MSMF was briefly put in front on the strength of that single good run and had
to be reverted, it stopped the camera connecting at all. For a rig that has
to start a session unattended, reliable-but-slow beats fast-but-intermittent,
so DSHOW leads and MSMF is the fallback for devices DSHOW cannot open.

The separate hazard DSHOW brings is that it may negotiate uncompressed YUY2,
whose bandwidth caps the real frame rate far below the requested one while
every property still reads back exactly as asked. That is what the FOURCC
warning after opening exists to surface, a starved camera should announce
itself rather than look like a slow GUI.
"""
import os

import pytest

cv2 = pytest.importorskip("cv2")

from source.video.cameras.opencv import _opencv_backend_order


@pytest.mark.skipif(os.name != "nt", reason="Windows backend order")
def test_dshow_is_tried_before_msmf_on_windows():
    """Reverted deliberately. MSMF-first stopped the camera connecting: 37 s
    to a first frame, then MF_E_HW_MFT_FAILED_START_STREAMING and no frames."""
    order = _opencv_backend_order()
    assert order.index(cv2.CAP_DSHOW) < order.index(cv2.CAP_MSMF), (
        "MSMF is ahead of DSHOW again; it was measured failing to stream "
        "entirely on the same camera it had worked on minutes earlier")


@pytest.mark.skipif(os.name != "nt", reason="Windows backend order")
def test_msmf_is_still_available_as_a_fallback():
    """DSHOW cannot open every device; MSMF stays in the list for those."""
    assert cv2.CAP_MSMF in _opencv_backend_order()


def test_every_platform_ends_with_a_generic_fallback():
    """CAP_ANY lets the platform choose when the preferred backend cannot
    open the device, a Jetson CSI node, an unusual driver."""
    assert _opencv_backend_order()[-1] == cv2.CAP_ANY
