''''
In The Name of GOD
Author : Ali Pilehvar Meibody

 Best Hyperparameters for MLP

'''



#========================================
#========================================
#------MLP-------------------------------
#========================================
#========================================





#========================================
#--------- L*----------------------------
#========================================
best_params_={'regressor__model__activation': 'tanh',
 'regressor__model__alpha': 1e-06,
 'regressor__model__hidden_layer_sizes': (10, 5, 1),
 'regressor__model__learning_rate_init': 0.1,
 'regressor__model__solver': 'adam',
 'regressor__scaler': MinMaxScaler(),
 'transformer': StandardScaler()}



#========================================
#--------- a*----------------------------
#========================================
best_params_={'regressor__model__activation': 'relu',
 'regressor__model__alpha': 0.6,
 'regressor__model__hidden_layer_sizes': (10,),
 'regressor__model__learning_rate_init': 0.4,
 'regressor__model__solver': 'adam',
 'regressor__scaler': StandardScaler(),
 'transformer': StandardScaler()}




#========================================
#--------- b*----------------------------
#========================================
best_params_={'regressor__model__activation': 'relu',
 'regressor__model__alpha': 1e-08,
 'regressor__model__hidden_layer_sizes': (3, 1),
 'regressor__model__learning_rate_init': 1e-07,
 'regressor__model__solver': 'lbfgs',
 'regressor__scaler': MinMaxScaler(),
 'transformer': MinMaxScaler()}
