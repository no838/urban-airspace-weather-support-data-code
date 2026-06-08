#!/usr/bin/env python3
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    'Manuscript/main_manuscript.docx',
    'Supplementary_Information/Supplementary_Information.docx',
    'Cover_Letter/cover_letter.docx',
    'Source_Data/component_year_panel_full.csv',
    'Source_Data/diagnostic_ledger.csv',
    'Source_Data/public_support_score_fields.csv',
    'Source_Data/need_composite_weights.csv',
    'Source_Data/event_validation_design.csv',
    'Source_Data/event_true_source_panel.csv',
    'Source_Data/event_matched_low_gap_controls.csv',
    'Source_Data/event_permutation_results.csv',
    'Source_Data/event_robustness_estimates.csv',
    'Figures/Main/Figure_1.pdf',
    'Figures/Main/Figure_2.pdf',
    'Figures/Main/Figure_3.pdf',
    'Figures/Main/Figure_4.pdf',
]

def count_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)

def main():
    missing = [p for p in REQUIRED if not (ROOT/p).exists()]
    if missing:
        print('Missing required files:')
        for p in missing: print(' -', p)
        return 1
    panel = ROOT/'Source_Data/component_year_panel_full.csv'
    try:
        rows = count_rows(panel)
        print(f'component_year_panel_full.csv rows: {rows}')
    except Exception as e:
        print('Could not count component-year panel rows:', e)
    print('Required-file audit passed.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
