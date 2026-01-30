# SfM CLI Usage

The Structure from Motion pipeline uses Tyro with a dataclass-based configuration system for easy experimentation.

## Configuration System

The pipeline uses a `SfMConfig` dataclass defined in `config.py`. You can:
1. **Modify defaults** in `config.py` for persistent changes
2. **Override via CLI** using `--cfg.param_name value` for one-off experiments

## Basic Usage

```bash
# Run with defaults from config.py
uv run python sfm.py

# Show help and all available options
uv run python sfm.py --help
```

## Modifying Defaults

Edit `config.py` to change default values:

```python
@dataclass
class SfMConfig:
    feature_type: Literal["sift", "disk"] = "disk"  # Change to "sift"
    num_features: int = 2048  # Change to 4096
    matcher_type: Literal["bf", "lightglue"] = "lightglue"
    # ... etc
```

## Command-Line Overrides

All parameters can be overridden via CLI using the `--cfg.` prefix:

### Feature Extraction Options

```bash
# Use SIFT features (override default)
uv run python sfm.py --cfg.feature-type sift

# Use DISK features with more features
uv run python sfm.py --cfg.feature-type disk --cfg.num-features 4096

# Limit image size
uv run python sfm.py --cfg.max-size 2048
```

### Matching Options

```bash
# Use BF matcher with custom lowe_ratio
uv run python sfm.py --cfg.matcher-type bf --cfg.lowe-ratio 0.8

# Use LightGlue matcher with custom min_dist
uv run python sfm.py --cfg.matcher-type lightglue --cfg.min-dist 0.2
```

### Dataset Selection

```bash
# Use custom dataset
uv run python sfm.py --cfg.dataset my_dataset

# Adjust minimum inliers threshold
uv run python sfm.py --cfg.dataset temple --cfg.min-inliers 30
```

### Bundle Adjustment

```bash
# Skip bundle adjustment
uv run python sfm.py --cfg.no-run-ba

# Run BA with first camera fixed
uv run python sfm.py --cfg.run-ba --cfg.fix-first-camera
```

## Complete Examples

```bash
# SIFT + BF with custom parameters
uv run python sfm.py --cfg.feature-type sift --cfg.matcher-type bf --cfg.lowe-ratio 0.8

# DISK + LightGlue with more features
uv run python sfm.py --cfg.feature-type disk --cfg.matcher-type lightglue --cfg.min-dist 0.2 --cfg.num-features 4096

# Custom dataset with SIFT features
uv run python sfm.py --cfg.feature-type sift --cfg.dataset temple --cfg.num-features 4096

# Quick reconstruction (fewer features, no BA)
uv run python sfm.py --cfg.num-features 1024 --cfg.min-inliers 30 --cfg.no-run-ba

# High-quality reconstruction (more features, fixed first camera)
uv run python sfm.py --cfg.num-features 8192 --cfg.fix-first-camera
```

## Configuration Display

The pipeline prints the full configuration at startup, showing all parameter values being used:

```
============================================================
SfM Configuration
============================================================
Feature Extraction:
  Type:         disk
  Num features: 2048
  Max size:     4080

Keypoint Matching:
  Type:         lightglue
  Min distance: 0.0

Dataset:
  Name:         statue
  Min inliers:  50

Optimization:
  Bundle adjustment: True
  Fix first camera:  False
============================================================
```

## Output

The pipeline will:
1. Display the configuration being used
2. Extract features from images in `data/raw/{dataset}/`
3. Construct the view graph
4. Process the reconstruction
5. Save initial reconstruction to `data/out/{dataset}/{dataset}_{feature_type}_{matcher_type}.ply`
6. Run bundle adjustment (if enabled)
7. Save optimized reconstruction to `data/out/{dataset}/{dataset}_{feature_type}_{matcher_type}_ba.ply`

## Tips for Experimentation

1. **Edit `config.py`** for parameters you change frequently
2. **Use CLI overrides** for quick one-off experiments
3. **Check the printed config** at startup to verify your settings
4. **Start with fewer features** (`--cfg.num-features 1024`) for faster iteration
5. **Disable BA** (`--cfg.no-run-ba`) during initial testing

