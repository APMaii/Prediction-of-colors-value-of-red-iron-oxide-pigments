'''
In The name of GOD

Author : Ali Pilehvar Meibody

CLI Prediction Guide for SVR Color Models.
'''

import os
import csv
import sys
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    'L*': os.path.join(BASE_DIR, 'svr_L_star.joblib'),
    'a*': os.path.join(BASE_DIR, 'svr_a_star.joblib'),
    'b*': os.path.join(BASE_DIR, 'svr_b_star.joblib'),
}
CSV_PATH = os.path.join(BASE_DIR, 'full_input_space_predictions.csv')

REDOX_LABELS = {0: 'No Redox', 1: '1x Redox', 2: '2x Redox'}


def banner():
    print()
    print('=' * 60)
    print('   Welcome to the Color Prediction Guide!')
    print('=' * 60)
    print()
    print('   This tool uses trained SVR machine-learning models')
    print('   to help you with ceramic glaze color prediction.')
    print()
    print('   Features  : Redox Agent (0/1/2), Temperature, Time')
    print('   Targets   : L*, a*, b*  (CIE Lab color space)')
    print()
    print('   We have two modes:')
    print()
    print('   [1] Forward Prediction')
    print('       You give process parameters  -->  We predict L*, a*, b*')
    print('       (uses the trained ML models directly)')
    print()
    print('   [2] Inverse Lookup')
    print('       You give desired L*, a*, b*  -->  We find the best')
    print('       process parameters from the prediction table (.csv)')
    print()
    print('   [q] Quit')
    print()
    print('-' * 60)


# =====================================================================
# Load ML models
# =====================================================================

def load_models():
    models = {}
    for target, path in MODEL_PATHS.items():
        if not os.path.isfile(path):
            print(f'  [ERROR] Model file not found: {path}')
            print('  Run  prepared_pre_trainedmodel.py  first.')
            sys.exit(1)
        models[target] = joblib.load(path)
    return models


# =====================================================================
# Load prediction CSV
# =====================================================================

def load_prediction_table():
    if not os.path.isfile(CSV_PATH):
        print(f'  [ERROR] CSV not found: {CSV_PATH}')
        print('  Run  prepared_pre_trainedmodel.py  first.')
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    return df


# =====================================================================
# Mode 1 — Forward: process params --> color
# =====================================================================

def forward_mode(models):
    print()
    print('=' * 60)
    print('  MODE 1 : Forward Prediction')
    print('  Give process parameters  -->  Get predicted color values')
    print('=' * 60)

    results = []
    while True:
        print()
        try:
            redox = int(input('  Redox Agent  (0 / 1 / 2)  : '))
            if redox not in (0, 1, 2):
                print('  [!] Redox Agent must be 0, 1, or 2.')
                continue
            temp = float(input('  Temperature  (°C)         : '))
            time = float(input('  Time         (min)        : '))
        except ValueError:
            print('  [!] Please enter valid numbers.')
            continue

        X = np.array([[redox, temp, time]])
        pred_L = models['L*'].predict(X).flatten()[0]
        pred_a = models['a*'].predict(X).flatten()[0]
        pred_b = models['b*'].predict(X).flatten()[0]

        print()
        print('  +-------------------------------+')
        print('  |    Predicted Color Values      |')
        print('  +-------------------------------+')
        print(f'  |  L*  =  {pred_L:>8.4f}              |')
        print(f'  |  a*  =  {pred_a:>8.4f}              |')
        print(f'  |  b*  =  {pred_b:>8.4f}              |')
        print('  +-------------------------------+')

        results.append([redox, temp, time, pred_L, pred_a, pred_b])

        again = input('\n  Predict another? (y/n): ').strip().lower()
        if again != 'y':
            break

    if results:
        out = os.path.join(BASE_DIR, 'forward_results.csv')
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Redox Agent', 'Temperature', 'Time',
                        'L*_pred', 'a*_pred', 'b*_pred'])
            for r in results:
                w.writerow([r[0], r[1], r[2],
                            round(r[3], 4), round(r[4], 4), round(r[5], 4)])
        print(f'\n  Results saved to {out}')


# =====================================================================
# Mode 2 — Inverse: color --> process params (CSV lookup)
# =====================================================================

def inverse_mode(table):
    print()
    print('=' * 60)
    print('  MODE 2 : Inverse Lookup')
    print('  Give desired L*, a*, b*  -->  Find process parameters')
    print('  (searches the pre-computed prediction table)')
    print('=' * 60)

    pred_L = table['L*_pred'].values
    pred_a = table['a*_pred'].values
    pred_b = table['b*_pred'].values

    all_results = []
    while True:
        print()
        try:
            tgt_L = float(input('  Desired L*  : '))
            tgt_a = float(input('  Desired a*  : '))
            tgt_b = float(input('  Desired b*  : '))
            n_str = input('  How many best matches? (default 5): ').strip()
            n_top = int(n_str) if n_str else 5
        except ValueError:
            print('  [!] Please enter valid numbers.')
            continue

        dist = np.sqrt(
            (pred_L - tgt_L) ** 2 +
            (pred_a - tgt_a) ** 2 +
            (pred_b - tgt_b) ** 2
        )
        best_idx = np.argsort(dist)[:n_top]

        print()
        print(f'  Top {n_top} closest matches for L*={tgt_L}, a*={tgt_a}, b*={tgt_b}')
        print()
        print(f'  {"Rank":<5} {"Redox":>6} {"Temp(°C)":>9} {"Time(min)":>10}'
              f'  {"L*_pred":>8} {"a*_pred":>8} {"b*_pred":>8} {"ΔE":>8}')
        print('  ' + '-' * 70)

        for rank, idx in enumerate(best_idx, 1):
            row = table.iloc[idx]
            de = dist[idx]
            print(f'  {rank:<5} {int(row["Redox Agent"]):>6}'
                  f' {int(row["Temperature"]):>9} {int(row["Time"]):>10}'
                  f'  {row["L*_pred"]:>8.3f} {row["a*_pred"]:>8.3f}'
                  f' {row["b*_pred"]:>8.3f} {de:>8.3f}')
            all_results.append([
                tgt_L, tgt_a, tgt_b, rank,
                int(row['Redox Agent']), int(row['Temperature']), int(row['Time']),
                round(row['L*_pred'], 4), round(row['a*_pred'], 4),
                round(row['b*_pred'], 4), round(de, 4),
            ])

        print('  ' + '-' * 70)

        again = input('\n  Search another color? (y/n): ').strip().lower()
        if again != 'y':
            break

    if all_results:
        out = os.path.join(BASE_DIR, 'inverse_results.csv')
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Desired_L*', 'Desired_a*', 'Desired_b*', 'Rank',
                         'Redox Agent', 'Temperature', 'Time',
                         'Pred_L*', 'Pred_a*', 'Pred_b*', 'DeltaE'])
            for r in all_results:
                w.writerow(r)
        print(f'\n  Results saved to {out}')


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    print('\n  Loading models and prediction table ...')
    models = load_models()
    table = load_prediction_table()
    print('  Ready!\n')

    while True:
        banner()
        choice = input('  Select mode (1 / 2 / q): ').strip().lower()

        if choice == '1':
            forward_mode(models)
        elif choice == '2':
            inverse_mode(table)
        elif choice in ('q', 'quit', 'exit'):
            print('\n  Goodbye!\n')
            break
        else:
            print('  [!] Invalid choice, please try again.')
