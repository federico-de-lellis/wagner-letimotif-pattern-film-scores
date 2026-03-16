#!/usr/bin/env python3
"""
Wagner Leitmotif Pattern Recognition & Matching in Film Composer Scores
Computational musicology pipeline using music21 for musical pattern discovery.
"""

import os
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from music21 import converter, interval, stream, note, chord

warnings.filterwarnings('ignore', category=UserWarning)

# Music representation module

def extract_representations(path: str) -> dict:
    """
    Extract multiple music representations from a MusicXML file.
    Returns a dict with keys: 'midi', 'interval', 'contour', 'rhythm', 'combined'
    """
    score = converter.parse(path)

    # Select melody: use the top part (index 0), flatten, extract notes
    part = score.parts[0].flatten()

    # Extract individual notes (skip chords — take highest note from chords)
    notes_list = []
    for el in part.notesAndRests:
        if isinstance(el, note.Note):
            notes_list.append(el)
        elif isinstance(el, chord.Chord):
            # Take the highest pitch note from the chord
            top_note = note.Note(el.pitches[-1])
            top_note.duration = el.duration
            notes_list.append(top_note)

    if len(notes_list) < 2:
        raise ValueError(f"Too few notes in {path}")

    # MIDI pitch sequence
    midi_seq = [n.pitch.midi for n in notes_list]

    # Chromatic interval sequence (transposition-invariant)
    interval_seq = []
    for i in range(len(notes_list) - 1):
        try:
            iv = interval.Interval(noteStart=notes_list[i], noteEnd=notes_list[i + 1])
            interval_seq.append(iv.semitones)
        except Exception:
            interval_seq.append(midi_seq[i + 1] - midi_seq[i])

    # Melodic contour (Dowling 1978: -1, 0, +1)
    contour_seq = [
        0 if interval_seq[i] == 0 else (1 if interval_seq[i] > 0 else -1)
        for i in range(len(interval_seq))
    ]

    # Rhythmic duration sequence (in quarter-note units)
    rhythm_seq = [float(n.duration.quarterLength) for n in notes_list]

    # Combined (interval + rhythm tuples)
    combined_seq = list(zip(interval_seq, rhythm_seq[:-1]))

    return {
        'midi': midi_seq,
        'interval': interval_seq,
        'contour': contour_seq,
        'rhythm': rhythm_seq,
        'combined': combined_seq,
        'notes': notes_list,
        'path': path
    }


# Pattern Matching Engine

def find_exact_matches(leitmotif_repr: dict, score_repr: dict, rep_type: str = 'interval') -> list:
    """
    Sliding-window search for the leitmotif interval sequence in the score.
    """
    pattern = leitmotif_repr[rep_type]
    text = score_repr[rep_type]
    n = len(pattern)
    matches = []

    for i in range(len(text) - n + 1):
        window = text[i:i + n]
        if window == pattern:
            matches.append({
                'start_note_idx': i,
                'end_note_idx': i + n,
                'match_type': 'exact',
                'similarity': 1.0,
                'representation': rep_type
            })
    return matches


def levenshtein_distance(seq1: list, seq2: list, max_dist: int = None) -> int:
    """Levenshtein distance with optional early termination."""
    m, n = len(seq1), len(seq2)
    if max_dist is not None and abs(m - n) > max_dist:
        return max_dist + 1
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        if max_dist is not None and min(dp[i]) > max_dist:
            return max_dist + 1
    return dp[m][n]


def find_approximate_matches(
    leitmotif_repr: dict,
    score_repr: dict,
    rep_type: str = 'interval',
    max_distance: int = 2
) -> list:
    """
    Sliding-window approximate matching using Levenshtein distance.
    """
    pattern = leitmotif_repr[rep_type]
    text = score_repr[rep_type]
    n = len(pattern)
    matches = []

    # Skip approximate matching for very long patterns (too slow)
    if n > 20:
        return matches

    for i in range(len(text) - n + 1):
        window = text[i:i + n]
        dist = levenshtein_distance(pattern, window, max_dist=max_distance)
        if 0 < dist <= max_distance:
            similarity = 1.0 - (dist / n)
            matches.append({
                'start_note_idx': i,
                'end_note_idx': i + n,
                'match_type': 'approximate',
                'edit_distance': dist,
                'similarity': round(similarity, 4),
                'representation': rep_type
            })

    return sorted(matches, key=lambda x: x['similarity'], reverse=True)


def find_contour_matches(
    leitmotif_repr: dict,
    score_repr: dict,
    min_similarity: float = 0.75
) -> list:
    """
    Search for matches based on melodic contour (direction sequence).
    """
    pattern = leitmotif_repr['contour']
    text = score_repr['contour']
    n = len(pattern)
    matches = []

    if n == 0:
        return matches

    for i in range(len(text) - n + 1):
        window = text[i:i + n]
        agreements = sum(1 for a, b in zip(pattern, window) if a == b)
        similarity = agreements / n
        if similarity >= min_similarity:
            matches.append({
                'start_note_idx': i,
                'end_note_idx': i + n,
                'match_type': 'contour',
                'similarity': round(similarity, 4),
                'representation': 'contour'
            })
    return sorted(matches, key=lambda x: x['similarity'], reverse=True)


# Filtering

def filter_matches(matches: list, min_similarity: float = 0.80, top_n: int = 20) -> list:
    """
    Filter matches by similarity threshold, remove overlaps, return top N.
    """
    if not matches:
        return []

    filtered = [m for m in matches if m['similarity'] >= min_similarity]
    filtered.sort(key=lambda x: x['similarity'], reverse=True)

    non_overlapping = []
    used_ranges = []

    for match in filtered:
        start, end = match['start_note_idx'], match['end_note_idx']
        overlap = any(not (end <= r[0] or start >= r[1]) for r in used_ranges)
        if not overlap:
            non_overlapping.append(match)
            used_ranges.append((start, end))

    return non_overlapping[:top_n]

# Full Pipeline Orchestrator

def run_full_pipeline(
    leitmotif_dir: str,
    score_dir: str,
    output_dir: str,
    representations: list = None,
    max_edit_distance: int = 2,
    min_similarity: float = 0.80
) -> pd.DataFrame:
    """
    Run the complete leitmotif matching pipeline.
    """
    if representations is None:
        representations = ['interval', 'contour']

    os.makedirs(output_dir, exist_ok=True)
    results = []

    # Load all leitmotifs
    leitmotif_files = sorted(
        list(Path(leitmotif_dir).glob("*.xml")) +
        list(Path(leitmotif_dir).glob("*.musicxml"))
    )
    print(f"Found {len(leitmotif_files)} leitmotif files.")

    leitmotifs = {}
    for lf in leitmotif_files:
        try:
            leitmotifs[lf.stem] = extract_representations(str(lf))
            n_notes = len(leitmotifs[lf.stem]['midi'])
            n_intervals = len(leitmotifs[lf.stem]['interval'])
            print(f"  ✓ Loaded leitmotif: {lf.stem} ({n_notes} notes, {n_intervals} intervals)")
        except Exception as e:
            print(f"  ✗ Error loading {lf.stem}: {e}")

    # Process each composer directory
    composer_dirs = sorted([d for d in Path(score_dir).iterdir() if d.is_dir()])

    for composer_dir in composer_dirs:
        composer = composer_dir.name
        score_files = sorted(
            list(composer_dir.glob("*.xml")) +
            list(composer_dir.glob("*.musicxml")) +
            list(composer_dir.glob("*.mxl"))
        )
        print(f"\nProcessing {composer}: {len(score_files)} scores")

        for score_file in score_files:
            try:
                score_repr = extract_representations(str(score_file))
                n_notes = len(score_repr['midi'])
                print(f"  ✓ Parsed: {score_file.name} ({n_notes} notes)")
            except Exception as e:
                print(f"  ✗ Skipping {score_file.name}: {e}")
                continue

            for leit_name, leit_repr in leitmotifs.items():
                all_matches = []

                # Exact interval match
                exact = find_exact_matches(leit_repr, score_repr, 'interval')
                all_matches.extend(exact)

                # Approximate interval match
                approx = find_approximate_matches(
                    leit_repr, score_repr, 'interval', max_edit_distance
                )
                all_matches.extend(approx)

                # Contour match
                contour = find_contour_matches(leit_repr, score_repr, min_similarity)
                all_matches.extend(contour)

                # Filter
                filtered = filter_matches(all_matches, min_similarity)

                for match in filtered:
                    results.append({
                        'composer': composer,
                        'score': score_file.stem,
                        'leitmotif': leit_name,
                        'match_type': match['match_type'],
                        'similarity': match['similarity'],
                        'start_note_idx': match['start_note_idx'],
                        'end_note_idx': match['end_note_idx'],
                        'representation': match['representation'],
                        'edit_distance': match.get('edit_distance', 0)
                    })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('similarity', ascending=False)
        df.to_csv(os.path.join(output_dir, 'matches_full.csv'), index=False)
        print(f"\n✅ Saved {len(df)} matches to matches_full.csv")
    else:
        print("\n⚠️ No matches found above threshold.")

    return df


# Analysis & Reporting

def generate_summary(df: pd.DataFrame, output_dir: str):
    """Generate summary tables and visualizations."""
    if df.empty:
        print("No data to summarize.")
        return

    # --- Match counts per leitmotif per composer ---
    pivot = df.groupby(['leitmotif', 'composer']).size().unstack(fill_value=0)
    pivot.to_csv(os.path.join(output_dir, 'summary_counts.csv'))
    print("\n=== Match Counts per Leitmotif per Composer ===")
    print(pivot.to_string())

    # --- Top 20 highest-similarity matches ---
    top20 = df.nlargest(20, 'similarity')[
        ['composer', 'score', 'leitmotif', 'match_type', 'similarity']
    ]
    top20.to_csv(os.path.join(output_dir, 'top20_matches.csv'), index=False)
    print("\n=== Top 20 Matches by Similarity ===")
    print(top20.to_string())

    # --- Heatmap of match frequency ---
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd')
    plt.title("Leitmotif Match Frequency Across Film Composers")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_matches.png'), dpi=150)
    plt.close()

    # --- Similarity distribution per composer ---
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='composer', y='similarity')
    plt.title("Similarity Score Distribution by Composer")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_similarity.png'), dpi=150)
    plt.close()

    # --- Match type breakdown ---
    plt.figure(figsize=(8, 4))
    df['match_type'].value_counts().plot(kind='bar', color=['#2a9d8f', '#e9c46a', '#e76f51'])
    plt.title("Match Types (Exact / Approximate / Contour)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'barchart_match_types.png'), dpi=150)
    plt.close()

    print(f"\nVisualizations saved to {output_dir}")


def export_match_excerpt(
    leitmotif_repr: dict,
    score_repr: dict,
    match: dict,
    output_path: str
):
    """
    Export a matched excerpt from the score as a MusicXML snippet.
    """
    from music21 import stream as m21stream

    start = match['start_note_idx']
    end = match['end_note_idx']
    matched_notes = score_repr['notes'][start:end]

    excerpt = m21stream.Part()
    for n in matched_notes:
        excerpt.append(n)

    excerpt.write('musicxml', fp=output_path)
    print(f"  Excerpt saved: {output_path}")


# Execution
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LEITMOTIF_DIR = os.path.join(BASE_DIR, "leitmotifs")
    SCORE_DIR = os.path.join(BASE_DIR, "scores")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    MAX_EDIT_DISTANCE = 2
    MIN_SIMILARITY = 0.80

    print("=" * 70)
    print("Wagner Leitmotif Pattern Recognition & Matching Pipeline")
    print("=" * 70)

    results_df = run_full_pipeline(
        leitmotif_dir=LEITMOTIF_DIR,
        score_dir=SCORE_DIR,
        output_dir=OUTPUT_DIR,
        max_edit_distance=MAX_EDIT_DISTANCE,
        min_similarity=MIN_SIMILARITY
    )

    generate_summary(results_df, OUTPUT_DIR)

    # Export top 5 excerpts for qualitative review
    if not results_df.empty:
        excerpt_dir = os.path.join(OUTPUT_DIR, "excerpts")
        os.makedirs(excerpt_dir, exist_ok=True)
        print("\nExporting top match excerpts for qualitative analysis...")

        # Re-load representations for top matches
        top5 = results_df.nlargest(5, 'similarity')
        leitmotif_files = sorted(
            list(Path(LEITMOTIF_DIR).glob("*.xml")) +
            list(Path(LEITMOTIF_DIR).glob("*.musicxml"))
        )
        leitmotifs = {}
        for lf in leitmotif_files:
            try:
                leitmotifs[lf.stem] = extract_representations(str(lf))
            except Exception:
                pass

        for idx, row in top5.iterrows():
            composer_dir = Path(SCORE_DIR) / row['composer']
            score_files = (
                list(composer_dir.glob("*.xml")) +
                list(composer_dir.glob("*.musicxml")) +
                list(composer_dir.glob("*.mxl"))
            )
            score_file = None
            for sf in score_files:
                if sf.stem == row['score']:
                    score_file = sf
                    break
            if score_file and row['leitmotif'] in leitmotifs:
                try:
                    score_repr = extract_representations(str(score_file))
                    match_info = {
                        'start_note_idx': row['start_note_idx'],
                        'end_note_idx': row['end_note_idx']
                    }
                    fname = f"excerpt_{idx}_{row['leitmotif']}_in_{row['composer']}.musicxml"
                    export_match_excerpt(
                        leitmotifs[row['leitmotif']],
                        score_repr,
                        match_info,
                        os.path.join(excerpt_dir, fname)
                    )
                except Exception as e:
                    print(f"  ✗ Could not export excerpt {idx}: {e}")

    print("\n" + "=" * 70)
    print("Pipeline Complete")
    print("=" * 70)
