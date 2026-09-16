"""pyBehaveTrack's Analyze tab, ported whole, 2D only.

Same tab, same controls, same layout. The only thing removed is the 3D half,
the workspace toggle, the volume editor, multi-view reconstruction and the
``*_pose3d.txt`` branches, none of which this rig produces.

Its engine is ``tools.offline_analysis.engine``; its theme and zone drawing come
from ``tools.offline_analysis.vendor``; its detectors come through
``engine.trackers``. Nothing here imports the rig.
"""
