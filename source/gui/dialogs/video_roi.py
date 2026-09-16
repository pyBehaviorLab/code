"""Video / ROI dialogs, single module.

Contains:
  * ROISegmentationDialog, multi-box ROI segmentation for camera splits.
  * CameraProxy, read-only proxy over a live CameraThread, so
                            this dialog doesn't open a second device handle.
"""
import cv2
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets


# ----------------------------------------------------------------------------
# CameraProxy, read-only wrapper over a running CameraThread.
# ----------------------------------------------------------------------------
# Exposes the subset of cv2.VideoCapture API the dialogs need over the live
# CameraThread, sharing its frames. Avoids opening a second VideoCapture on
# the same camera_id (Windows DirectShow refuses it; resolution would also
# drift from the live pipeline's).
# ----------------------------------------------------------------------------


class CameraProxy:
    """File-like read interface over a running CameraThread."""

    def __init__(self, camera_thread):
        self._thread = camera_thread
        self._seen_version = 0

    def isOpened(self) -> bool:
        t = self._thread
        if t is None:
            return False
        # Different backends expose ``connected`` differently; fall back
        # to ``isRunning()`` (QThread) when the attribute isn't present.
        connected = getattr(t, "connected", None)
        if connected is None:
            try:
                return bool(t.isRunning())
            except Exception:
                return False
        return bool(connected)

    def read(self) -> Tuple[bool, Optional["np.ndarray"]]:
        """Return ``(ok, frame_copy_or_None)``. Mirrors cv2.VideoCapture.read."""
        if not self.isOpened():
            return False, None
        lock = getattr(self._thread, "last_frame_lock", None)
        if lock is None:
            frame = getattr(self._thread, "last_frame", None)
            return (frame is not None,
                    frame.copy() if frame is not None else None)
        with lock:
            frame = getattr(self._thread, "last_frame", None)
            if frame is None:
                return False, None
            return True, frame.copy()

    def read_if_new(self) -> Tuple[bool, Optional["np.ndarray"]]:
        """``(ok, frame)`` only when the camera produced a frame we haven't
        shown. An unchanged camera costs one lock, no copy and no repaint,
        the BGR→RGB→QPixmap chain is the dialog's dominant per-tick cost.
        """
        t = self._thread
        getter = getattr(t, "get_latest_frame_versioned", None) if t else None
        if getter is None:
            return self.read()
        got = getter(self._seen_version)
        if got is None:
            return False, None
        frame, _ts_ns, version = got
        self._seen_version = version
        return True, frame

    def release(self) -> None:
        """No-op, the proxy doesn't own the camera."""


class BusFrameSource:
    """A ``cv2.VideoCapture``-shaped view of a live FrameBus.

    The preferred source when the camera is already streaming: it subscribes
    to the frames the pipeline is publishing anyway, so drawing ROIs costs no
    device access at all.

    The alternative, opening a second ``cv2.VideoCapture`` on a device the
    capture thread already owns, is what this replaces. On Windows that
    either fails outright or steals the handle from the running camera, and
    it is why "Draw regions…" could stall for seconds mid-session. It also
    reached into ``video_manager.cameras`` to find the thread, going around
    the Pipeline that owns it; ``Pipeline.get_bus`` is the public way in and
    is what the lens-calibration wizard already uses.
    """

    _is_proxy = True          # the dialog must not release a shared device

    def __init__(self, bus):
        self._latest = None
        self._version = 0
        self._seen = 0
        self._unsub = bus.on_camera_frame(self._on_frame)

    def _on_frame(self, cf):
        # Called on the pipeline tick thread. Reference assignment only,
        # the bus allocates a fresh array per frame, so no copy is needed
        # and the GUI thread never blocks the publisher.
        self._latest = getattr(cf, "image", None)
        self._version += 1

    def isOpened(self) -> bool:
        return self._unsub is not None

    def read(self) -> Tuple[bool, Optional["np.ndarray"]]:
        frame = self._latest
        return (frame is not None), frame

    def read_if_new(self) -> Tuple[bool, Optional["np.ndarray"]]:
        if self._version == self._seen:
            return False, None
        self._seen = self._version
        return self.read()

    def release(self) -> None:
        """Unsubscribe. Releases nothing on the device; we never held it."""
        unsub, self._unsub = self._unsub, None
        if unsub is not None:
            try:
                unsub()
            except Exception as e:
                logger.debug("ROI dialog: bus unsubscribe failed: %s", e)



from source.log import get_logger

logger = get_logger()



# =============================================================================
#  roi_segmentation
# =============================================================================

class ROISegmentationDialog(QtWidgets.QDialog):
    """Draw per-box ROIs on a live camera frame.

    Uses ``ROIDrawCanvas`` from ``frame_display`` so the displayed frame
    scales to the dialog window automatically (KeepAspectRatio,
    Qt-native; no QScrollArea, no manual rescaling).  ROIs are drawn in
    image-pixel coords against whatever resolution the camera delivers,
    no MAX-resolution override, no second capture if the camera is
    already running in VideoManager.

    Public API kept stable for existing callers:
      - ``self.segments`` : ``{box_number: (x, y, w, h)}`` in image pixels.
      - ``self.frame_size`` : ``(W, H)`` of the frame the ROIs were drawn on.
      - Callers should compute percent = pixel / frame_size at build time.
    """

    def _apply_default_size(self, pref_w: int, pref_h: int) -> None:
        """Open at ``pref_w × pref_h`` but never larger than the parent GUI
        window (falling back to the available screen), so the dialog fits the
        window it was launched from. Stays resizable for expansion."""
        w, h = int(pref_w), int(pref_h)
        parent = self.parent()
        win = parent.window() if parent is not None else None
        if win is not None:
            g = win.geometry()
            if g.width() > 0 and g.height() > 0:
                w = min(w, max(480, g.width()))
                h = min(h, max(360, g.height()))
        scr = self.screen()
        if scr is not None:
            avail = scr.availableGeometry()
            w = min(w, avail.width() - 40)
            h = min(h, avail.height() - 60)
        self.resize(max(480, w), max(360, h))

    def __init__(self, camera_id, box_items, parent=None, locked_size=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.box_items = box_items
        self.segments: dict = {}     # {box_number: (x, y, w, h)} pixel
        self.frame_size = (0, 0)     # (W, H), set when first frame arrives
        self.current_box_index = 0
        #: A size already fixed on an earlier camera. On a multi-camera rig
        #: every box feeds the same pose model, and the model takes ONE input
        #: shape, so the ROIs have to match across cameras and not merely
        #: within one. ``None`` for the first camera, which sets it.
        self._incoming_lock = tuple(locked_size) if locked_size else None
        #: True when the incoming size does not fit this camera's frame; the
        #: caller reports it rather than quietly drawing a different size.
        self.lock_rejected = False

        self.setWindowTitle(f"Camera {camera_id} - ROI Segmentation")
        self.setModal(True)
        # Open no larger than the GUI it came from (capped to the screen), so
        # it never overflows a small window. Freely resizable, drag it bigger
        # (or maximise) for pixel-precise ROI work.
        self.setSizeGripEnabled(True)
        self._apply_default_size(900, 700)

        # Frame source: prefer VideoManager's running CameraThread (via
        # CameraProxy) so we don't open a second device handle.  Fall
        # back to a transient cv2.VideoCapture only when no live camera
        # exists (the typical pre-connect path).
        self._cap = self._open_frame_source()
        if self._cap is None:
            QtWidgets.QMessageBox.critical(
                self, "Camera Error", f"Could not open camera {camera_id}"
            )
            self.reject()
            return

        self._owns_cap = not getattr(self._cap, "_is_proxy", False)

        self._setup_ui()
        self._apply_dark_theme()

        # Restore existing ROIs from the CANONICAL normalized form, scaled to
        # the resolution actually streaming now, never from the stale pixel
        # ``roi_segment``, which was drawn at some previous session's frame
        # size. Seeding from pixels and then re-deriving percent on Done was
        # the bug that permanently shifted the ROI every time the dialog
        # reopened at a different resolution. Frame size isn't known until the
        # first frame arrives, so defer the seed to ``_tick_frame``.
        self._pending_norm_rois = {}
        for item in box_items:
            bw = item['box_widget']
            norm = getattr(bw, 'roi_normalized', None)
            if norm and len(norm) == 4:
                try:
                    self._pending_norm_rois[item['box_number']] = tuple(
                        float(v) for v in norm)
                except (TypeError, ValueError):
                    pass
        self._rois_seeded = False

        # Frame poll timer, 20 FPS preview is plenty for ROI drawing.
        self._frame_timer = QtCore.QTimer(self)
        self._frame_timer.timeout.connect(self._tick_frame)
        self._frame_timer.start(50)
        # accept()/reject() route through QDialog.done(), which hides rather
        # than closes, closeEvent alone would leave this timer polling the
        # camera on the GUI thread for the rest of the session. ``finished``
        # fires for every exit path.
        self.finished.connect(self._release_frame_source)

        # Pull a frame immediately so the dialog opens with content,
        # not a black canvas.
        self._tick_frame()

    # ── frame source ──────────────────────────────────────────

    @staticmethod
    def _live_thread_for(vm, camera_id):
        """The running CameraThread for ``camera_id``, tolerating id type.

        ``vm.cameras`` is keyed by whatever ``connect_camera`` was handed,
        an int for OpenCV, while callers may hold the id as text. A missed
        lookup is expensive, not harmless: it drops through to probing and
        opening a SECOND handle on a device the capture thread already owns,
        which stalls for seconds on Windows.
        """
        cams = getattr(vm, "cameras", None) or {}
        for key in (camera_id, str(camera_id)):
            if key in cams:
                return cams[key]
        # The dict is keyed by whatever connect_camera was handed, often the
        # index a project saved, while this dialog may hold the identity the
        # picker offers. Same camera, different spelling.
        try:
            from source.video.cameras.identity import matching_key
            key = matching_key(camera_id, cams.keys())
        except Exception:
            key = None
        if key is not None:
            return cams[key]
        try:
            return cams.get(int(camera_id))
        except (TypeError, ValueError):
            return None

    def _open_frame_source(self):
        """Return a cv2-VideoCapture-shaped object for ``self.camera_id``.

        Order: the live FrameBus → the live CameraThread → a transient
        cv2.VideoCapture opened at the resolution the live system will use.
        Matching resolutions keeps the percent-based crop consistent between
        the ROI preview and the live stream.

        The transient capture is the LAST resort and only correct when the
        camera is not yet streaming (the normal pre-connect path). Opening one
        on a running device is what made this dialog stall for seconds.
        """
        # ``mw`` is whatever parented this dialog, so both lookups below are
        # parent-TYPE guards, not mode sniffs, resolved once here rather than
        # re-probed at each use.
        mw = self.parent()
        vm = getattr(mw, 'video_manager', None) if mw else None
        pipe = getattr(mw, 'pipeline', None) if mw is not None else None

        # 1. The pipeline's bus, if this camera is already streaming. Costs no
        #    device access; we read the frames it publishes anyway.
        if pipe is not None:
            try:
                bus = pipe.get_bus(self.camera_id)
            except Exception as e:
                logger.debug("ROI dialog: get_bus(%s) failed: %s",
                             self.camera_id, e)
                bus = None
            if bus is not None:
                logger.info("ROI dialog: using the live FrameBus for camera %s",
                            self.camera_id)
                return BusFrameSource(bus)

        # 2. A live CameraThread the pipeline has not (yet) bussed.
        if vm is not None:
            cam_thread = (self._live_thread_for(vm, self.camera_id)
                          if hasattr(vm, 'cameras') else None)
            if cam_thread is not None:
                proxy = CameraProxy(cam_thread)
                if proxy.isOpened():
                    proxy._is_proxy = True
                    logger.info(f"ROI dialog: using live CameraThread for camera {self.camera_id}")
                    return proxy

        # Determine the resolution the live system WILL use, in priority:
        #   1. main_window.video_camera_resolution (set by Camera Connect dialog
        #      via probe_supported_resolutions before this dialog opens)
        #   2. video_manager.get_target_resolution() (programmatic setting)
        #   3. probe the camera and use max
        target_res = None
        if mw is not None:
            target_res = getattr(mw, 'video_camera_resolution', None)
        if target_res is None and vm is not None:
            try:
                target_res = vm.get_target_resolution()
            except Exception:
                target_res = None
        if not target_res:
            # This camera's own probed mode, recorded by the Camera Setup
            # dialog. Entry points that skip preflight (the "Draw regions…"
            # button) land here; without it they pay a 12-mode reprobe.
            if pipe is not None:
                try:
                    ccfg = pipe.get_camera_config(self.camera_id)
                    target_res = getattr(ccfg, 'selected_resolution', None)
                except Exception:
                    target_res = None
        if not target_res:
            try:
                from source.video.cameras.opencv import OpenCVCamera
                _modes, max_mode = OpenCVCamera.probe_supported_resolutions(self.camera_id)
                if max_mode:
                    target_res = max_mode
            except Exception:
                target_res = None

        # Fallback: transient cv2 capture at the target resolution + MJPG.
        # MJPG is required for most USB webcams to actually deliver high
        # resolutions, the default uncompressed mode caps at 640×480.
        # Same priority order the live capture uses, so the preview opens on
        # the backend that will actually serve the session (MSMF-first here
        # meant the slow backend was tried first, and on a different one than
        # the probe had just measured).
        from source.video.cameras.opencv import (_coerce_cam_id,
                                                 _opencv_backend_order)
        # The id the dialog holds is an IDENTITY ("fp3557d2de"); cv2 takes an
        # index. Handing it the identity is what produced "Could not open
        # camera fp…" on the pre-connect ROI prompt, where this transient
        # capture is the only frame source there is.
        cam_addr = _coerce_cam_id(self.camera_id)
        for backend in _opencv_backend_order():
            cap = cv2.VideoCapture(cam_addr, backend)
            if not cap.isOpened():
                cap.release()
                continue
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                if target_res:
                    tw, th = int(target_res[0]), int(target_res[1])
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, tw)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, th)
            except Exception:
                pass
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(
                    f"ROI dialog: opened transient capture for camera "
                    f"{self.camera_id} at {w}x{h} (requested target {target_res})"
                )
                if target_res and (w, h) != tuple(target_res):
                    logger.warning(
                        f"ROI dialog: camera delivered {w}x{h} but live system "
                        f"will use {target_res}, saved ROI percent will still "
                        f"be correct (computed against actual delivered size)."
                    )
                return cap
            cap.release()
        return None

    def _tick_frame(self):
        if self._cap is None:
            return
        try:
            reader = getattr(self._cap, "read_if_new", self._cap.read)
            ret, frame = reader()
        except Exception as e:
            logger.debug(f"ROI dialog frame read error: {e}")
            return
        if not ret or frame is None:
            return
        h, w = frame.shape[:2]
        if self.frame_size != (w, h):
            self.frame_size = (w, h)
            logger.debug(f"ROI dialog frame size: {w}x{h}")
        self._canvas.set_frame(frame)
        # Seed saved ROIs once the real frame size is known: normalized →
        # pixels at THIS resolution, so a resolution change between sessions
        # can't shift them.
        if not self._rois_seeded and self.frame_size[0] > 0:
            fw, fh = self.frame_size
            seeded = {
                bn: (int(round(nx * fw)), int(round(ny * fh)),
                     int(round(nw * fw)), int(round(nh * fh)))
                for bn, (nx, ny, nw, nh) in self._pending_norm_rois.items()
            }
            if seeded:
                self._canvas.set_rois(seeded)
                self.segments.update(seeded)
            self._rois_seeded = True
            # Adopt the size an earlier camera fixed, now that this camera's
            # frame is known and the fit can actually be judged.
            if self._incoming_lock is not None:
                if not self._canvas.set_external_locked_size(self._incoming_lock):
                    self.lock_rejected = True
                    logger.warning(
                        "ROI: camera %s is %dx%d, too small for the %dx%d "
                        "region fixed on the first camera. Its boxes would "
                        "not match, so the size was not applied.",
                        self.camera_id, fw, fh, *self._incoming_lock)
                self._refresh_size_label()

    # ── UI ────────────────────────────────────────────────────
    def _apply_dark_theme(self):
        from source.gui.style_builders import apply_dialog_theme
        apply_dialog_theme(self)

    def _setup_ui(self):
        from source.gui.widgets.frame_display import ROIDrawCanvas

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        self.instruction_label = QtWidgets.QLabel(
            "Draw Box 1's region by click-dragging on the view, that sets the "
            "size. Every other box uses that SAME size (for consistent tracking) "
            ", just click to position it. Reset All to change the size."
        )
        # Token-driven info banner.
        self.instruction_label.setStyleSheet(
            "background-color: rgba(37,99,235,0.18); color: #93c5fd;"
            " padding: 8px 12px; font-weight: 700;"
            " border-left: 3px solid #2563eb;"
            " border-radius: 4px; font-size: 10pt;"
        )
        layout.addWidget(self.instruction_label)

        # Box selector buttons
        box_row = QtWidgets.QHBoxLayout()
        box_row.setSpacing(8)
        box_row.addWidget(QtWidgets.QLabel("Select Box:"))
        self.box_buttons = []
        for idx, item in enumerate(self.box_items):
            btn = QtWidgets.QPushButton(f"Box {item['box_number']}")
            btn.setFixedSize(80, 35)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked=False, i=idx: self.selectBox(i))
            self.box_buttons.append(btn)
            box_row.addWidget(btn)
        box_row.addStretch()
        layout.addLayout(box_row)

        # The canvas, auto-scales with the dialog window. Small floor so the
        # dialog can shrink to a narrow GUI; the frame scales to fit and the
        # user can expand the window for a bigger drawing area.
        self._canvas = ROIDrawCanvas()
        self._canvas.roi_changed.connect(self._on_roi_changed)
        self._canvas.setMinimumSize(360, 270)
        layout.addWidget(self._canvas, 1)

        # Action row
        action_row = QtWidgets.QHBoxLayout()
        action_row.setSpacing(8)
        reset_btn = QtWidgets.QPushButton("Reset Current Box")
        reset_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; padding: 8px; "
            "border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #F57C00; }"
        )
        reset_btn.clicked.connect(self.resetCurrentBox)
        action_row.addWidget(reset_btn)

        reset_all_btn = QtWidgets.QPushButton("Reset All (re-set size)")
        reset_all_btn.setStyleSheet(
            "QPushButton { background-color: #b45309; color: white; padding: 8px; "
            "border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #92400e; }"
        )
        reset_all_btn.clicked.connect(self.resetAllBoxes)
        action_row.addWidget(reset_all_btn)

        self.size_label = QtWidgets.QLabel("")
        self.size_label.setStyleSheet("color: #93c5fd; font-weight: 700;")
        action_row.addWidget(self.size_label)
        action_row.addStretch()
        layout.addLayout(action_row)

        # Bottom buttons
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(8)

        save_btn = QtWidgets.QPushButton("Save Configuration")
        save_btn.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; padding: 8px; "
            "border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #7B1FA2; }"
        )
        save_btn.clicked.connect(self.saveConfiguration)
        bottom_row.addWidget(save_btn)

        done_btn = QtWidgets.QPushButton("Done")
        done_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 8px; "
            "border: none; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        done_btn.clicked.connect(self.finish)
        bottom_row.addWidget(done_btn)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; padding: 8px; "
            "border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #da190b; }"
        )
        cancel_btn.clicked.connect(self.reject)
        bottom_row.addWidget(cancel_btn)

        layout.addLayout(bottom_row)

        self.selectBox(0)

    # ── event handlers ────────────────────────────────────────
    def selectBox(self, index):
        """Select which box the next click-drag draws an ROI for."""
        self.current_box_index = index
        for i, btn in enumerate(self.box_buttons):
            btn.setChecked(i == index)
            box_num = self.box_items[i]['box_number']
            has_roi = box_num in self.segments
            if i == index:
                if has_roi:
                    btn.setStyleSheet(
                        "QPushButton { background-color: #4CAF50; color: white; "
                        "font-weight: bold; border: 2px solid #2E7D32; border-radius: 4px; }")
                else:
                    btn.setStyleSheet(
                        "QPushButton { background-color: #2196F3; color: white; "
                        "font-weight: bold; border: 2px solid #1976D2; border-radius: 4px; }")
            else:
                if has_roi:
                    btn.setStyleSheet(
                        "QPushButton { background-color: #81C784; color: white; "
                        "border: 1px solid #66BB6A; border-radius: 4px; }")
                else:
                    btn.setStyleSheet(
                        "QPushButton { background-color: #BDBDBD; color: #424242; "
                        "border: 1px solid #9E9E9E; border-radius: 4px; }")
        self._canvas.set_current_box(self.box_items[index]['box_number'])
        self._update_done_state()

    def locked_size(self):
        """The region size this camera settled on, for the next camera.

        Read by the caller so a multi-camera rig fixes ONE size across every
        box, not one per camera.
        """
        try:
            return self._canvas.locked_size()
        except Exception:
            return None

    def _on_roi_changed(self):
        """Canvas pushed an ROI update, sync our public dict + UI."""
        self.segments = dict(self._canvas.get_rois())
        self._refresh_size_label()
        self.selectBox(self.current_box_index)

    def _refresh_size_label(self):
        """Show the locked region size (set by Box 1, shared by all)."""
        if not hasattr(self, "size_label"):
            return
        sz = self._canvas.locked_size()
        if sz and self._incoming_lock and not self.lock_rejected:
            # Say where the size came from, or the operator wonders why this
            # camera opened with a region already sized.
            self.size_label.setText(
                f"Region size: {sz[0]}×{sz[1]} px  (fixed by the first camera)")
        elif sz:
            self.size_label.setText(f"Region size: {sz[0]}×{sz[1]} px  (all boxes)")
        else:
            self.size_label.setText("Region size: draw Box 1 to set")

    def _update_done_state(self):
        all_done = all(item['box_number'] in self.segments
                       for item in self.box_items)
        locked = self._canvas.locked_size()
        if all_done:
            self.instruction_label.setText(
                "All regions defined! Click 'Done' to apply or 'Save Configuration' to export."
            )
            self.instruction_label.setStyleSheet(
                "QLabel { background-color: #4CAF50; color: white; padding: 8px; "
                "font-weight: bold; border-radius: 4px; font-size: 10pt; }"
            )
        else:
            self.instruction_label.setText(
                ("Click a box, then click to POSITION its region, size is fixed "
                 f"at {locked[0]}×{locked[1]} px. (Reset All to change the size.)")
                if locked else
                "Draw Box 1 by click-dragging to set the region size; every other "
                "box then uses that same size, just click to position it."
            )
            self.instruction_label.setStyleSheet(
                "QLabel { background-color: #094771; color: #e0e0e0; padding: 8px; "
                "font-weight: bold; border-radius: 4px; font-size: 10pt; }"
            )

    def resetCurrentBox(self):
        """Clear the current box's ROI (canvas + dict)."""
        item = self.box_items[self.current_box_index]
        bn = item['box_number']
        self._canvas.clear_roi(bn)
        if bn in self.segments:
            del self.segments[bn]
        logger.info(f"Reset ROI for Box {bn}")
        self._refresh_size_label()
        self.selectBox(self.current_box_index)

    def resetAllBoxes(self):
        """Clear every ROI, unlocks the size so the next Box 1 draw re-sets it."""
        for item in list(self.box_items):
            self._canvas.clear_roi(item['box_number'])
        self.segments = {}
        logger.info("Reset all ROIs (size unlocked)")
        self._refresh_size_label()
        self.selectBox(0)

    def finish(self):
        """All ROIs drawn? Then accept; storage is normalized at apply time."""
        for item in self.box_items:
            if item['box_number'] not in self.segments:
                QtWidgets.QMessageBox.warning(
                    self, "Incomplete",
                    f"Please draw ROI for Box {item['box_number']}"
                )
                return
        # Final frame-size snapshot so callers can compute percent without
        # re-reading the camera (read via seg_dialog.frame_size).
        logger.info(
            f"ROI segmentation complete: {self.segments} "
            f"(frame_size={self.frame_size})"
        )
        self.accept()

    def saveConfiguration(self):
        """Export ROIs to a JSON file as percent-only (resolution-independent)."""
        try:
            base_dir = Path("experiments")
            base_dir.mkdir(exist_ok=True)
        except Exception:
            base_dir = Path(".")
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Camera Configuration",
            str(base_dir / f"camera_{self.camera_id}_config.json"),
            "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return
        try:
            import json
            fw, fh = self.frame_size
            if fw <= 0 or fh <= 0:
                QtWidgets.QMessageBox.warning(
                    self, "Save Error", "No frame received yet, try again."
                )
                return
            segments_config = {}
            for box_num, rect in self.segments.items():
                x, y, w, h = rect
                segments_config[str(box_num)] = {
                    'percent': {
                        'x': x / fw, 'y': y / fh,
                        'width': w / fw, 'height': h / fh,
                    }
                }
            config = {
                'camera_id': self.camera_id,
                'segments': segments_config,
            }
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Saved camera configuration to {file_path}")
            QtWidgets.QMessageBox.information(
                self, "Configuration Saved",
                f"Configuration saved to {file_path}"
            )
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Save Error", f"Failed to save configuration: {e}"
            )

    def _release_frame_source(self, *_):
        """Stop the frame timer; release the camera only if we own it.

        Idempotent, reached from both ``finished`` and ``closeEvent``.
        """
        try:
            if getattr(self, '_frame_timer', None) is not None:
                self._frame_timer.stop()
        except Exception:
            pass
        try:
            if getattr(self, '_owns_cap', False) and self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        self._owns_cap = False

    def closeEvent(self, event):
        self._release_frame_source()
        super().closeEvent(event)

