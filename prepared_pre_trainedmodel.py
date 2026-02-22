'''
In The name of GOD

Author : Ali Pilehvar Meibody

Train the best-tuned SVR models for L*, a*, b* and save them to disk.
'''

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor as TTR
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV

# =====================================================================
# Load data
# =====================================================================

DATA_PATH = 'Experimental_data.xlsx'

data = pd.read_excel(DATA_PATH)

X  = np.array(data[['Redox Agent', 'Temperature', 'Time']])
y1 = np.array(data[['L*']]).reshape(-1, 1)   # L*
y2 = np.array(data[['a*']]).reshape(-1, 1)   # a*
y3 = np.array(data[['b*']]).reshape(-1, 1)   # b*

# =====================================================================
# Best hyperparameters (from tuned grid-search in mean-std-main.py)
# =====================================================================

configs = {
    'L_star': {
        'y': y1,
        'random_state': 42,
        'params': {
            'regressor__model__C': [100],
            'regressor__model__epsilon': [0.3],
            'regressor__model__gamma': ['scale'],
            'regressor__model__kernel': ['rbf'],
            'regressor__scaler': [StandardScaler()],
            'transformer': [StandardScaler()],
        },
    },
    'a_star': {
        'y': y2,
        'random_state': 20,
        'params': {
            'regressor__model__C': [60],
            'regressor__model__epsilon': [0.3],
            'regressor__model__gamma': ['scale'],
            'regressor__model__kernel': ['poly'],
            'regressor__scaler': [None],
            'transformer': [StandardScaler()],
        },
    },
    'b_star': {
        'y': y3,
        'random_state': 16,
        'params': {
            'regressor__model__C': [9],
            'regressor__model__epsilon': [0.3],
            'regressor__model__gamma': ['scale'],
            'regressor__model__kernel': ['rbf'],
            'regressor__scaler': [StandardScaler()],
            'transformer': [StandardScaler()],
        },
    },
}

# =====================================================================
# Train & save each model
# =====================================================================

for name, cfg in configs.items():
    print(f'Training SVR for {name} ...')

    model = SVR()
    pipe_model = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
    fold = KFold(n_splits=7, shuffle=True, random_state=cfg['random_state'])

    gs = GridSearchCV(
        regressor,
        param_grid=cfg['params'],
        cv=fold,
        scoring='neg_mean_absolute_percentage_error',
        n_jobs=1,
    )
    gs.fit(X, cfg['y'])

    best = gs.best_estimator_
    out_path = f'svr_{name}.joblib'
    joblib.dump(best, out_path)
    print(f'  Saved  -> {out_path}')
    print(f'  Score  -> {gs.best_score_:.6f}')

print('\nAll models saved successfully.')

# =====================================================================
# Generate predictions over the full input space and save to CSV
# =====================================================================

print('\nGenerating predictions over full input space ...')

a1_values = np.array([0, 1, 2])
a2_values = np.arange(0, 1001, 10)
a3_values = np.arange(60, 241, 1)
g1, g2, g3 = np.meshgrid(a1_values, a2_values, a3_values, indexing='ij')
input_space = np.column_stack((g1.flatten(), g2.flatten(), g3.flatten()))

trained_models = {}
for name in configs:
    trained_models[name] = joblib.load(f'svr_{name}.joblib')

pred_L = trained_models['L_star'].predict(input_space).flatten()
pred_a = trained_models['a_star'].predict(input_space).flatten()
pred_b = trained_models['b_star'].predict(input_space).flatten()

predictions_df = pd.DataFrame({
    'Redox Agent':  input_space[:, 0].astype(int),
    'Temperature':  input_space[:, 1].astype(int),
    'Time':         input_space[:, 2].astype(int),
    'L*_pred':      np.round(pred_L, 4),
    'a*_pred':      np.round(pred_a, 4),
    'b*_pred':      np.round(pred_b, 4),
})

csv_path = f'full_input_space_predictions.csv'
predictions_df.to_csv(csv_path, index=False)
print(f'Predictions table ({len(predictions_df)} rows) saved to {csv_path}')
