# SfM CLI Usage

The Structure from Motion pipeline now has a Typer CLI with configurable options.

## Basic Usage

```bash
# Run with defaults (SIFT features + BF matcher with lowe_ratio=0.75)
uv run python sfm.py

# Show help
uv run python sfm.py --help
```

## Feature Extraction Options

Choose between SIFT and DISK features:

```bash
# Use SIFT features (default)
uv run python sfm.py --features sift

# Use DISK features
uv run python sfm.py --features disk
# or short form:
uv run python sfm.py -f disk
```

## Matching Options

### Brute-Force Matcher (with Lowe's ratio test)

```bash
# Use BF matcher with default lowe_ratio=0.75
uv run python sfm.py --matcher bf

# Use BF matcher with custom lowe_ratio
uv run python sfm.py --matcher bf --lowe-ratio 0.8
# or short form:
uv run python sfm.py -m bf -l 0.8
```

### LightGlue Matcher

```bash
# Use LightGlue matcher with default min_dist=0.75
uv run python sfm.py --matcher lightglue

# Use LightGlue matcher with custom min_dist
uv run python sfm.py --matcher lightglue --min-dist 0.2
# or short form:
uv run python sfm.py -m lightglue -d 0.2
```

## Dataset Selection

```bash
# Use default dataset (statue_orbit)
uv run python sfm.py

# Use custom dataset
uv run python sfm.py --dataset my_dataset
# or short form:
uv run python sfm.py -s my_dataset
```

## Complete Examples

```bash
# SIFT + BF with lowe_ratio=0.8
uv run python sfm.py -f sift -m bf -l 0.8

# DISK + LightGlue with min_dist=0.2
uv run python sfm.py -f disk -m lightglue -d 0.2

# SIFT + LightGlue on custom dataset
uv run python sfm.py -f sift -m lightglue -d 0.5 -s temple

# DISK + BF with custom parameters
uv run python sfm.py -f disk -m bf -l 0.7 -s statue
```

## Output

The pipeline will:
1. Display the configuration being used
2. Extract features from images in `data/raw/{dataset}/`
3. Construct the view graph
4. Process the reconstruction
5. Save initial reconstruction to `data/out/{dataset}/{dataset}.ply`
6. Run bundle adjustment
7. Save optimized reconstruction to `data/out/{dataset}/{dataset}_ba.ply`

