# Wagner Leitmotif Pattern Recognition & Matching in Film Composer Scores

## Overview

A computational musicology pipeline that searches for 10 Wagner leitmotifs across 70 film composer scores (MusicXML) using music21. The project provides two implementations that share the same music representation layer but differ in their matching strategies:

- **`pipeline.py`** — standalone script using three sequential matching passes (exact, approximate, contour), suitable for batch processing with configurable thresholds.
- **`wagner_leitmotif_demo.ipynb`** — full research notebook with a unified multi-dimensional scorer, adaptive thresholds, match classification, statistical significance analysis, and visualisations.

### Matching strategies

Both implementations extract the same five music representations (MIDI pitch, semitone interval, melodic contour, rhythm, combined interval+rhythm) and apply a sliding-window search. The notebook extends this with:

1. **Unified combined score** — weighted combination of interval similarity (50%), contour similarity (30%), and rhythm similarity (20%), evaluated at every window position.
2. **Adaptive thresholds** — minimum combined-score threshold scales with motif length (0.70 for n ≤ 3 intervals down to 0.45 for n > 15), preventing short motifs from matching by contour alone.
3. **Match classification** — each accepted window is labelled `exact`, `near-exact`, `strong`, `moderate`, or `approximate` based on the interval and rhythm similarity components.
4. **Approximate matching (Levenshtein)** — secondary pass on interval sequences; maximum edit distance scales with pattern length (`max(1, n // 4)`, capped at 3).
5. **Statistical significance baseline** — expected random-match counts are estimated per motif and composer to contextualise observed match frequencies.

## Repository Structure

```
├── pipeline.py                          # Main pipeline script
├── wagner_leitmotif_demo.ipynb          # Full research notebook
├── leitmotifs/                          # 10 Wagner leitmotif MusicXML files
│   ├── ForestMurmurs_motif.musicxml
│   ├── Horn_motif.musicxml
│   ├── Mime_motif.musicxml
│   ├── Nibelungs_motif.musicxml
│   ├── NibelungsHate_motif.musicxml
│   ├── Ride_motif.musicxml
│   ├── Ring_motif.musicxml
│   ├── SiblingsLove_motif.musicxml
│   ├── SwirlingBlaze_motif.musicxml
│   └── Sword_motif.musicxml
├── scores/                              # Film composer scores (MusicXML)
│   ├── ennio_morricone/                 # 19 scores
│   ├── erich_wolfgang_korngold/         # 13 scores
│   ├── howard_shore/                    # 8 scores
│   ├── john_williams/                   # 15 scores
│   └── max_steiner/                     # 15 scores
└── output/                              # Pipeline results
    ├── matches_full.csv                 # All matches with metadata
    ├── summary_counts.csv               # Leitmotif x composer pivot table
    ├── top20_matches.csv                # Top 20 highest-similarity matches
    ├── heatmap_matches.png              # Match frequency heatmap
    ├── boxplot_similarity.png           # Similarity distribution per composer
    ├── barchart_match_classes.png       # Match class breakdown (notebook)
    ├── density_significance.png         # Per-composer significance ratios (notebook)
    └── excerpts/                        # MusicXML snippets of top matches
```

## Quick Start

### 1. Install dependencies

```bash
pip install music21 numpy pandas matplotlib seaborn openpyxl nbformat nbconvert
```

Requires **Python >= 3.10**.

### 2. Run the pipeline on real MusicXML files

```bash
python3 pipeline.py
```

Results are written to `output/`.

### 3. Demo notebook

The notebook uses the data in `leitmotifs/` and `scores/` to run the full pipeline and produce all visualisations. To execute it end-to-end:

```bash
jupyter nbconvert --to notebook --execute --inplace wagner_leitmotif_demo.ipynb
```

Or open it in Jupyter / VS Code and click **Run All**.

## Using Your Own MusicXML Files

To swap in your own scores, edit `pipeline.py` and update:

```python
LEITMOTIF_DIR = "path/to/your/leitmotifs/"
SCORE_DIR = "path/to/your/scores/"
OUTPUT_DIR = "path/to/output/"
```

Score directories should be organised by composer:
```
scores/
├── composer_a/
│   ├── score1.musicxml
│   └── score2.musicxml
└── composer_b/
    └── score3.musicxml
```

## Pipeline Parameters

### `pipeline.py`

| Parameter | Default | Description |
|---|---|---|
| `MAX_EDIT_DISTANCE` | 2 | Levenshtein tolerance for approximate interval matching |
| `MIN_SIMILARITY` | 0.80 | Minimum similarity threshold applied to all three match types |

### `wagner_leitmotif_demo.ipynb`

| Parameter | Default | Description |
|---|---|---|
| `W_INTERVAL` | 0.50 | Weight of interval similarity in the combined score |
| `W_CONTOUR` | 0.30 | Weight of contour similarity in the combined score |
| `W_RHYTHM` | 0.20 | Weight of rhythm similarity in the combined score |
| Adaptive threshold | 0.45 – 0.70 | Minimum combined score; scales with motif length (see table below) |
| Max edit distance | `max(1, n // 4)`, capped at 3 | Levenshtein tolerance; scales with pattern length |

**Adaptive threshold by motif length:**

| Motif length (intervals) | Threshold | Rationale |
|---|---|---|
| <= 3 (e.g. Mime) | 0.70 | Very short: pure contour match peaks at ~0.46, so interval evidence is required |
| 4 – 6 (e.g. Ride, Sword) | 0.60 | Short: partial interval match needed |
| 7 – 10 (e.g. Ring, Horn) | 0.55 | Medium: moderate combined evidence filters noise |
| 11 – 15 (e.g. NibelungsHate) | 0.50 | Longer: patterns are inherently more selective |
| > 15 (e.g. ForestMurmurs) | 0.45 | Very long: even moderate combined evidence is significant |