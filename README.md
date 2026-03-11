# Wagner Leitmotif Pattern Recognition & Matching in Film Composer Scores

## Overview

A computational musicology pipeline that searches for Wagner leitmotif patterns across film composer scores using music21. The pipeline employs three complementary matching strategies:

1. **Exact match** — interval-based, transposition-invariant sliding window
2. **Approximate match** — Levenshtein edit distance with configurable tolerance
3. **Contour match** — melodic direction (shape-based) comparison

## Repository Structure

```
├── pipeline.py                          # Main pipeline script
├── wagner_leitmotif_demo.ipynb          # Self-contained
├── leitmotifs/                          # 10 Wagner leitmotif MusicXML files
├── scores/                              # Film composer scores (MusicXML)
│   ├── ennio_morricone/
│   ├── erich_wolfgang_korngold/
│   ├── howard_shore/
│   ├── john_williams/
│   └── max_steiner/
└── output/                              # Pipeline results
    ├── matches_full.csv                 # All matches with metadata
    ├── summary_counts.csv               # Leitmotif × composer pivot table
    ├── top20_matches.csv                # Top 20 highest-similarity matches
    ├── heatmap_matches.png              # Match frequency heatmap
    ├── boxplot_similarity.png           # Similarity distribution per composer
    ├── barchart_match_types.png         # Match type breakdown
    └── excerpts/                        # MusicXML snippets of top matches
```

## Quick Start

### 1. Install dependencies

```bash
pip install music21 numpy pandas scipy matplotlib seaborn openpyxl nbformat nbconvert
```

### 2. Run the pipeline on real MusicXML files

```bash
python3 pipeline.py
```

Results are written to `output/`.

### 3. Demo notebook

The demo notebook uses the data in leitmotifs/ and scores/ to run the pipeline and visualize results. To execute it end-to-end:

```bash
jupyter nbconvert --to notebook --execute --inplace wagner_leitmotif_demo.ipynb
```

Or open it in Jupyter/VS Code and click **Run All**.

## Using Your Own MusicXML Files

To swap in your own real scores, edit `pipeline.py` and update:

```python
LEITMOTIF_DIR = "path/to/your/leitmotifs/"
SCORE_DIR = "path/to/your/scores/"
OUTPUT_DIR = "path/to/output/"
```

Score directories should be organized by composer:
```
scores/
├── composer_a/
│   ├── score1.musicxml
│   └── score2.musicxml
└── composer_b/
    └── score3.musicxml
```

## Pipeline Parameters

| Parameter | Default | Description |
|---|---|---|
| `MAX_EDIT_DISTANCE` | 2 | Levenshtein tolerance (0 = exact only) |
| `MIN_SIMILARITY` | 0.80 | Minimum match quality threshold (0.0–1.0) |

## Theoretical Grounding

Based on Janssen, de Haas, Volk & van Kranenburg (2013) — *"Musical Pattern Discovery"*, CMMR 2013:

- Multiple music representations (pitch interval, contour, rhythm)
- String-based sliding-window search
- Approximate matching via Levenshtein edit distance
- Length + similarity filtering

## License

Research use only.
