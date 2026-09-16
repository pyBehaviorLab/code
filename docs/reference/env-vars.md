# Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `QT_QPA_PLATFORM` | Qt platform plugin. `offscreen` for headless testing. | OS default |
| `PYBEHAVIORLAB_LOGLEVEL` | Console log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`). The per-GUI log file is always DEBUG. | display default |
| `PYBEHAVIORLAB_MCU_PARALLELISM` | Max concurrent workers for multi-box MCU operations (parallel task upload / config). Clamped to 1–16. | 4 |
| `PYBL_POSE_DEBUG` | Set to `1` to enable verbose per-batch logging in PoseSink. | unset |
| `PYBL_NO_PYDEV` | Skip pydevd-related setup in launchers (faster startup, no debugger). | unset |
| `PYBL_DLC_GPU` | Force DLC backend to use GPU (0/1). Auto-detected if unset. | auto |
| `PYBL_FFMPEG_PATH` | Override ffmpeg binary path if not on `$PATH`. | system path |
| `PYBL_NVENC_PROBE_TIMEOUT_S` | Seconds for NVENC capability probe at startup. | 5 |

## Logging

Logging is configured at startup in `source/log.py`. The console level is set by
`PYBEHAVIORLAB_LOGLEVEL`; the per-GUI log file under `data/log/` always records at
DEBUG. To capture more detail on the console:

```bash
export PYBEHAVIORLAB_LOGLEVEL=DEBUG
python pyOperant.py
```

The level can also be changed at runtime via `source.log.set_log_level(..)`.
