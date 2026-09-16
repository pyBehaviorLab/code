"""Video subsystem, capture, pipeline, tracking, recording.

One umbrella for everything that touches video frames or the data
files derived from them. Internal layout::

    cameras/    capture (backends, threads, manager)
    pipeline/   frame distribution (bus + sinks + Qt adapter + controller)
    tracking/   algorithms (blob, pose, smoothing)
    recording/  persistence (ffmpeg, recorder, frame log, drop log)

Both maze and operant reach into the four subpackages directly; this
package has no eager re-exports of its own.
"""
