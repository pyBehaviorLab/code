# Settings (`experiments/config/settings.json`)

App-wide settings (not per-project). Lives at `experiments/config/settings.json`
(a legacy `config/settings.json` is auto-migrated on first read). Read via
`source/gui/config_manager.py::get_setting(category, key)`.

## Categories

```
config_manager.get_setting(category, key)
config_manager.set_setting(category, key, value)
```

Common categories:

- `recents`, last-loaded project paths
- `defaults`, default new-project directory, default task
- `ui`, last-active mode, window geometry hints (advisory)
- `tracking`, persisted per-box `TrackingConfig`s for cross-project re-use
  (`tracking_configs` key, dict keyed by stringified box id)
- `cameras`, cached camera registry across launches
- `mcu`. MCU picker label mode (`display_mode`: `hashed` | `native` | `serial`)
- `display`, live-tile display cap (see below)
- `video`, recording encoder preferences (see below)

## `display` options

| Key | Default | Meaning |
|---|---|---|
| `max_fps` | `null` | Cap the per-tile display refresh. `null` = follow the camera (paint every captured frame). An int caps the *display* only; capture/recording are unaffected. Used by `base.py::_paint_streaming_cameras_once`. |

## `video` options, recording encoder

| Key | Default | Meaning |
|---|---|---|
| `use_gpu` | `"auto"` | `"auto"` = use a hardware encoder if one is present; `true` = force it; `false` = CPU only (predictable, no NVENC session cap). |
| `allow_cpu_fallback` | `true` | When a hardware encoder IS present but its writer won't open, typically the consumer-GPU NVENC concurrent-session cap (~12) being exceeded on a many-box rig, record on CPU `libx264` so the box still records rather than losing video. `false` = strict hardware-only (abort the box's video on failure). |
| `prefer_hevc` | `false` | Use H.265 (`hevc_nvenc` / `libx265`) where available, smaller files; note HEVC and H.264 NVENC sessions are counted separately against the cap. |

Encoder selection is otherwise automatic and platform-aware: NVENC → Intel QSV
→ AMD AMF → Jetson `v4l2m2m` → CPU `libx264` → OpenCV MJPEG/XVID. Live hardware
sessions are tracked (`ffmpeg.hw_encoder_sessions()`) and surfaced on the sticky
"No video. NVENC session limit" per-box alarm.

## When to use settings vs project config

| Concern | Where |
|---|---|
| "I want this project to remember its rig wiring" | `experiment_config.json` |
| "I want every project I open to start with the same default tracker mode" | `config/settings.json` |
| "I want the file picker to start where I last opened" | `config/settings.json` |

Settings travel with the app install, not with the project. Don't commit to settings
anything that should be reproducible from a project file, that defeats the
project-as-source-of-truth contract.
