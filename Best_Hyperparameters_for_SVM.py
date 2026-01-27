''''
In The Name of GOD
Author : Ali Pilehvar Meibody


Content : Best  Hyperparameters for SVM
'''
#========================================
#--------- L*----------------------------
#========================================
best_params_={'regressor__model__C': 100,
 'regressor__model__epsilon': 0.3,
 'regressor__model__gamma': 'scale',
 'regressor__model__kernel': 'rbf',
 'regressor__scaler': StandardScaler(),
 'transformer': StandardScaler()}

#========================================
#--------- a*----------------------------
#========================================
best_params_={'regressor__model__C': 60,
 'regressor__model__epsilon': 0.3,
 'regressor__model__gamma': 'scale',
 'regressor__model__kernel': 'poly',
 'regressor__scaler': None,
 'transformer': StandardScaler()}

#========================================
#--------- b*----------------------------
#========================================
best_params_={'regressor__model__C': 9,
 'regressor__model__epsilon': 0.3,
 'regressor__model__gamma': 'scale',
 'regressor__model__kernel': 'rbf',
 'regressor__scaler': StandardScaler(),
 'transformer': StandardScaler()}

