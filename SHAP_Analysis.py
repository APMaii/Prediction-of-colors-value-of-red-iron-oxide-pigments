'''
In The Name of GOD
Author : Ali Pilehvar Meibody


SHAP ANalysis

'''
#===========================================================
#===========================================================
'Importing Libs'
#===========================================================
#===========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from  sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR

from sklearn.pipeline import Pipeline as PPL
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor as TTR



#====================================
#-----------Import Data-------------
#====================================

data=pd.read_excel('Experimental_data.xlsx')
data.columns
# Index(['Additive', 'Temp', 'Time', 'L*', 'a*', 'b*'], dtype='object')


X=data[['Redox Agent', 'Temperature', 'Time']]
X=np.array(X)



L=data[['L*']]
L=np.array(L)

A=data[['a*']]
A=np.array(A)

B=data[['b*']]
B=np.array(B)



#===========================================================
# SHAP Analysis for Best Models (SVR for L*, a*, b*)
#===========================================================

try:
    import shap
    shap_available = True
except ImportError:
    print("SHAP library not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap
    shap_available = True

# Set matplotlib backend for SHAP plots
plt.ioff()  # Turn off interactive mode for better SHAP plot saving

print("\n" + "="*60)
print("Starting SHAP Analysis")
print("="*60)

# Feature names for SHAP plots
feature_names = ['Redox Agent', 'Temperature', 'Time']

#===========================================================
# 1. SHAP Analysis for L* (SVR Model)
#===========================================================
print("\n1. Performing SHAP analysis for L* (SVR model)...")

# Re-train the L* model to get the best estimator
yy = L
yyy = 'L*'
model = SVR()
pipe_model = PPL([("scaler", MinMaxScaler()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
fold1 = KFold(n_splits=7, shuffle=True, random_state=42)

myparams = {'regressor__model__C': [100],
            'regressor__model__epsilon': [0.3],
            'regressor__model__gamma': ['scale'],
            'regressor__model__kernel': ['rbf'],
            'regressor__scaler': [StandardScaler()],
            'transformer': [StandardScaler()]}

gs_L = GridSearchCV(regressor, param_grid=myparams, cv=fold1, 
                    scoring='neg_mean_absolute_percentage_error', n_jobs=10)
gs_L.fit(X, yy)
best_model_L = gs_L.best_estimator_

# Create a wrapper function for SHAP (handles preprocessing)
def predict_wrapper_L(X_input):
    pred = best_model_L.predict(X_input)
    # Return as 1D array for SHAP
    return pred.flatten() if pred.ndim > 1 else pred

# Use a sample of data for background (SHAP needs background data)
background_data = shap.sample(X, 10)  # Use 10 samples as background

# Create SHAP explainer
explainer_L = shap.KernelExplainer(predict_wrapper_L, background_data)

# Calculate SHAP values for all data
shap_values_L = explainer_L.shap_values(X, nsamples=100)

# Create SHAP plots
print("   Generating SHAP plots for L*...")

# Summary plot
plt.figure(figsize=(10, 8), dpi=600)
shap.summary_plot(shap_values_L, X, feature_names=feature_names, show=False)
plt.title(f'SHAP Summary Plot - {yyy} (SVR)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'SHAP_Summary_{yyy}_SVR.png', dpi=600, bbox_inches='tight')
plt.close()

# Bar plot (mean absolute SHAP values)
plt.figure(figsize=(10, 6), dpi=600)
shap.summary_plot(shap_values_L, X, feature_names=feature_names, plot_type="bar", show=False)
plt.title(f'SHAP Feature Importance - {yyy} (SVR)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'SHAP_Bar_{yyy}_SVR.png', dpi=600, bbox_inches='tight')
plt.close()

# Waterfall plot for a single prediction
try:
    plt.figure(figsize=(10, 6), dpi=600)
    # Ensure shap_values are properly shaped - get first sample
    shap_vals_single = np.array(shap_values_L[0])
    if shap_vals_single.ndim > 1:
        shap_vals_single = shap_vals_single.flatten()
    
    # Handle expected_value (could be scalar or array)
    base_val = explainer_L.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.array(base_val).flatten()[0])
    else:
        base_val = float(base_val)
    
    # Create explanation object
    explanation = shap.Explanation(values=shap_vals_single, 
                                   base_values=base_val,
                                   data=X[0], 
                                   feature_names=feature_names)
    
    # Use the newer API if available, otherwise use waterfall_plot
    try:
        shap.plots.waterfall(explanation, show=False)
    except AttributeError:
        shap.waterfall_plot(explanation, show=False)
    
    plt.title(f'SHAP Waterfall Plot - {yyy} (SVR) - First Sample', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'SHAP_Waterfall_{yyy}_SVR.png', dpi=600, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"   Warning: Could not create waterfall plot for {yyy}: {e}")
    print("   Skipping waterfall plot...")

print(f"   ✓ SHAP plots saved for {yyy}")

#===========================================================
# 2. SHAP Analysis for a* (SVR Model)
#===========================================================
print("\n2. Performing SHAP analysis for a* (SVR model)...")

yy = A
yyy = 'a*'
model = SVR()
pipe_model = PPL([("scaler", MinMaxScaler()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
fold2 = KFold(n_splits=7, shuffle=True, random_state=20)

myparams = {'regressor__model__C': [60],
            'regressor__model__epsilon': [0.3],
            'regressor__model__gamma': ['scale'],
            'regressor__model__kernel': ['poly'],
            'regressor__scaler': [None],
            'transformer': [StandardScaler()]}

gs_A = GridSearchCV(regressor, param_grid=myparams, cv=fold2, 
                    scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
gs_A.fit(X, yy)
best_model_A = gs_A.best_estimator_

# Create wrapper function
def predict_wrapper_A(X_input):
    pred = best_model_A.predict(X_input)
    # Return as 1D array for SHAP
    return pred.flatten() if pred.ndim > 1 else pred

# Create SHAP explainer
background_data = shap.sample(X, 10)
explainer_A = shap.KernelExplainer(predict_wrapper_A, background_data)

# Calculate SHAP values
shap_values_A = explainer_A.shap_values(X, nsamples=100)

# Create SHAP plots
print("   Generating SHAP plots for a*...")

# Summary plot
plt.figure(figsize=(10, 8), dpi=600)
shap.summary_plot(shap_values_A, X, feature_names=feature_names, show=False)
plt.title(f'SHAP Summary Plot - {yyy} (SVR)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'SHAP_Summary_{yyy}_SVR.png', dpi=600, bbox_inches='tight')
plt.close()

# Bar plot
plt.figure(figsize=(10, 6), dpi=600)
shap.summary_plot(shap_values_A, X, feature_names=feature_names, plot_type="bar", show=False)
plt.title(f'SHAP Feature Importance - {yyy} (SVR)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'SHAP_Bar_{yyy}_SVR.png', dpi=600, bbox_inches='tight')
plt.close()

# Waterfall plot
try:
    plt.figure(figsize=(10, 6), dpi=600)
    # Ensure shap_values are properly shaped - get first sample
    shap_vals_single = np.array(shap_values_A[0])
    if shap_vals_single.ndim > 1:
        shap_vals_single = shap_vals_single.flatten()
    
    # Handle expected_value (could be scalar or array)
    base_val = explainer_A.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.array(base_val).flatten()[0])
    else:
        base_val = float(base_val)
    
    # Create explanation object
    explanation = shap.Explanation(values=shap_vals_single, 
                                   base_values=base_val,
                                   data=X[0], 
                                   feature_names=feature_names)
    
    # Use the newer API if available, otherwise use waterfall_plot
    try:
        shap.plots.waterfall(explanation, show=False)
    except AttributeError:
        shap.waterfall_plot(explanation, show=False)
    
    plt.title(f'SHAP Waterfall Plot - {yyy} (SVR) - First Sample', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'SHAP_Waterfall_{yyy}_SVR.png', dpi=600, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"   Warning: Could not create waterfall plot for {yyy}: {e}")
    print("   Skipping waterfall plot...")

print(f"   ✓ SHAP plots saved for {yyy}")

#===========================================================
# 3. SHAP Analysis for b* (SVR Model)
#===========================================================
print("\n3. Performing SHAP analysis for b* (SVR model)...")

yy = B
yyy = 'b*'
model = SVR()
pipe_model = PPL([("scaler", MinMaxScaler()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
fold2 = KFold(n_splits=7, shuffle=True, random_state=16)

myparams = {'regressor__model__C': [9],
            'regressor__model__epsilon': [0.3],
            'regressor__model__gamma': ['scale'],
            'regressor__model__kernel': ['rbf'],
            'regressor__scaler': [StandardScaler()],
            'transformer': [StandardScaler()]}

gs_B = GridSearchCV(regressor, param_grid=myparams, cv=fold2, 
                    scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
gs_B.fit(X, yy)
best_model_B = gs_B.best_estimator_

# Create wrapper function
def predict_wrapper_B(X_input):
    pred = best_model_B.predict(X_input)
    # Return as 1D array for SHAP
    return pred.flatten() if pred.ndim > 1 else pred

# Create SHAP explainer
background_data = shap.sample(X, 10)
explainer_B = shap.KernelExplainer(predict_wrapper_B, background_data)

# Calculate SHAP values
shap_values_B = explainer_B.shap_values(X, nsamples=100)

# Create SHAP plots
print("   Generating SHAP plots for b*...")

# Summary plot
plt.figure(figsize=(10, 8), dpi=600)
shap.summary_plot(shap_values_B, X, feature_names=feature_names, show=False)
plt.title(f'SHAP Summary Plot - {yyy} (SVR)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'SHAP_Summary_{yyy}_SVR.png', dpi=600, bbox_inches='tight')
plt.close()

# Bar plot
plt.figure(figsize=(10, 6), dpi=600)
shap.summary_plot(shap_values_B, X, feature_names=feature_names, plot_type="bar", show=False)
plt.title(f'SHAP Feature Importance - {yyy} (SVR)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'SHAP_Bar_{yyy}_SVR.png', dpi=600, bbox_inches='tight')
plt.close()

# Waterfall plot
try:
    plt.figure(figsize=(10, 6), dpi=600)
    # Ensure shap_values are properly shaped - get first sample
    shap_vals_single = np.array(shap_values_B[0])
    if shap_vals_single.ndim > 1:
        shap_vals_single = shap_vals_single.flatten()
    
    # Handle expected_value (could be scalar or array)
    base_val = explainer_B.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.array(base_val).flatten()[0])
    else:
        base_val = float(base_val)
    
    # Create explanation object
    explanation = shap.Explanation(values=shap_vals_single, 
                                   base_values=base_val,
                                   data=X[0], 
                                   feature_names=feature_names)
    
    # Use the newer API if available, otherwise use waterfall_plot
    try:
        shap.plots.waterfall(explanation, show=False)
    except AttributeError:
        shap.waterfall_plot(explanation, show=False)
    
    plt.title(f'SHAP Waterfall Plot - {yyy} (SVR) - First Sample', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'SHAP_Waterfall_{yyy}_SVR.png', dpi=600, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"   Warning: Could not create waterfall plot for {yyy}: {e}")
    print("   Skipping waterfall plot...")

print(f"   ✓ SHAP plots saved for {yyy}")

#===========================================================
# Combined SHAP Comparison
#===========================================================
print("\n4. Creating combined SHAP comparison plots...")

# Create a combined bar plot showing feature importance for all three models
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=600)

# Calculate mean absolute SHAP values for each feature
shap_means_L = np.abs(shap_values_L).mean(0)
shap_means_A = np.abs(shap_values_A).mean(0)
shap_means_B = np.abs(shap_values_B).mean(0)

# Plot for L*
axes[0].barh(feature_names, shap_means_L, color='#2E86AB', edgecolor='black', linewidth=1.2)
axes[0].set_title('L* (SVR)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Plot for a*
axes[1].barh(feature_names, shap_means_A, color='#A23B72', edgecolor='black', linewidth=1.2)
axes[1].set_title('a* (SVR)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

# Plot for b*
axes[2].barh(feature_names, shap_means_B, color='#F18F01', edgecolor='black', linewidth=1.2)
axes[2].set_title('b* (SVR)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='x')

plt.suptitle('SHAP Feature Importance Comparison Across L*, a*, and b*', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('SHAP_Combined_Comparison.png', dpi=600, bbox_inches='tight')
plt.close()

print("   ✓ Combined SHAP comparison plot saved")

# Print summary statistics
print("\n" + "="*60)
print("SHAP Analysis Summary")
print("="*60)
print("\nMean Absolute SHAP Values (Feature Importance):")
print("\nL* Model:")
for i, feat in enumerate(feature_names):
    print(f"  {feat}: {shap_means_L[i]:.4f}")

print("\na* Model:")
for i, feat in enumerate(feature_names):
    print(f"  {feat}: {shap_means_A[i]:.4f}")

print("\nb* Model:")
for i, feat in enumerate(feature_names):
    print(f"  {feat}: {shap_means_B[i]:.4f}")

print("\n" + "="*60)
print("SHAP Analysis Complete!")
print("="*60)
print("\nGenerated files:")
print("  - SHAP_Summary_L*_SVR.png")
print("  - SHAP_Bar_L*_SVR.png")
print("  - SHAP_Waterfall_L*_SVR.png")
print("  - SHAP_Summary_a*_SVR.png")
print("  - SHAP_Bar_a*_SVR.png")
print("  - SHAP_Waterfall_a*_SVR.png")
print("  - SHAP_Summary_b*_SVR.png")
print("  - SHAP_Bar_b*_SVR.png")
print("  - SHAP_Waterfall_b*_SVR.png")
print("  - SHAP_Combined_Comparison.png")
print("\n")

# Re-enable interactive mode
plt.ion()
