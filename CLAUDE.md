# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build
cargo build

# Build (release)
cargo build --release

# Run
cargo run

# Run tests
cargo test

# Run a single test
cargo test test_find_files
```

The binary requires `./input/` directory with `.jpg` puzzle piece images and an `./output/` directory for results.

## Architecture

This is a Rust application that processes jigsaw puzzle piece photos using OpenCV to detect, classify, and match puzzle pieces.

### Processing Pipeline (per image, `main.rs::process`)

1. **Check cache** — if `./output/<file>.json` exists, load it and skip reprocessing
2. **Image preprocessing** — read → grayscale → blur
3. **Threshold search** — iterate thresholds 160–230, pick the one yielding the shortest contour
4. **Contour extraction** — threshold → bitwise-not → morphological open → find contours (largest one kept)
5. **Corner detection** — `goodFeaturesToTrack` to find 4 corners; tries both original grey and filled-poly images, picks the better result
6. **Contour splitting** — splits the single piece contour into 4 directional sides (Up/Down/Left/Right) using polygon-based point-in-polygon tests
7. **Gender classification** — classifies each side as `Male` (tab sticking out), `Female` (blank), or `Line` (flat/straight edge)
8. **Output** — saves annotated image to `./output/<file>_contours.jpg` and JSON to `./output/<file>.json`

### Shape Matching (`main.rs::match_shapes`)

After all pieces are processed, every pair of pieces is compared using shape descriptors (`d1`–`d5` on `ContourWithDir`). A match is accepted when complementary genders (Male↔Female) have sufficiently close descriptors. Matched links are stored in `ContourWithDir.links` as `HashMap<filename, Direction>`.

### Key Types

- **`PuzzlePiece`** (`main.rs`) — owns all per-image state: raw image, grey/blurred mats, contours, corners, bounding geometry, and the 4 directional `ContourWithDir` segments
- **`ContourWithDir`** (`contour_with_dir.rs`) — one directional side of a piece; holds the raw contour points, translated/rotated contour, shape descriptors `d1`–`d5`, gender, and matched links; serialized to JSON (contour points are skipped in serde)
- **`Direction`** / **`Genders`** (`contour_with_dir.rs`) — enums with `EnumIter`, `Serialize`/`Deserialize`
- **`SingleContourParams`** (`single_contour_params.rs`) — transient struct returned by `split_contour_by_direction`

### Module Responsibilities

| Module | Role |
|--------|------|
| `cv_utils.rs` | Thin wrappers around OpenCV calls (I/O, blur, threshold, morph, corner detection, etc.) |
| `contour_with_dir.rs` | `ContourWithDir` struct + `Direction`/`Genders` enums + `get_extreme` helper |
| `draw.rs` | Drawing helpers (corners as circles, directional contours) |
| `utils.rs` | `find_files` — lists `.jpg` files from a directory |
| `not_used_fn.rs` | Dead code; safe to ignore |

### Parallelism

Images are processed in parallel using `rayon`'s `into_par_iter`. The JSON caching step allows incremental runs — previously processed pieces are loaded instantly.
