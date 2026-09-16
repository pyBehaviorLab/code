# Cython build

Some hot-path code is Cython-compiled for speed. Sources live in
`source/cython/`; build artifacts go next to them as `.so` / `.pyd`.

## Auto-build

`source/cython/build.py` is invoked at first import if the compiled extension is
missing. Most users never touch this.

## Manual build

```bash
python -m source.cython.build
```

Forces a rebuild even if artifacts exist. Useful after touching `.pyx` files.

## Modules built

| Module | What |
|---|---|
| `source/cython/<name>.pyx` | Hot-path numerics (zone geometry, image ops) |

(Inventory varies; check `source/cython/` for the current list.)

## Requirements

- C compiler (gcc on Linux, MSVC on Windows, clang on macOS)
- Cython (`pip install cython`)
- NumPy headers (already in `requirements.txt`)

## Failure mode

If Cython build fails at startup, the launcher falls back to pure-Python paths
where available, otherwise raises. Check the per-launch log for the build error.
