''''
In The Name of GOD
Author : Ali Pilehvar Meibody


Content : Hyperparameters space for each Supervised Machine learning algorithem


'''
#--------------------- LR
myparams={'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler()],
          'transformer': [None,StandardScaler(),MinMaxScaler(),RobustScaler()]}



#--------------------- KNN
myparams={'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler()],
          "regressor__model__n_neighbors": np.arange(1,16),
          'regressor__model__metric': ['braycurtis','canberra','chebyshev','cityblock','correlation','cosine','euclidean','minkowski','sqeuclidean','hamming'],
          'transformer': [None,StandardScaler(),MinMaxScaler(),RobustScaler()]}



#--------------------- DT
myparams={'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler()],
          'regressor__model__max_depth': np.arange(1,16),
          'regressor__model__criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
          'regressor__model__splitter': ['best','random'],
          'regressor__model__min_samples_leaf': [0.1,0.3,0.5,0.7,0.9,1,2,3,4,5,6,7,8,9],
          'regressor__model__min_samples_split': [0.1,0.3,0.5,0.7,0.9,2,3,4,5,6,7,8,9],
          'transformer': [None,StandardScaler(),MinMaxScaler(),RobustScaler()]}



#--------------------- RF
myparams={'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler()],
          'regressor__model__n_estimators': np.arange(1,11),
          'regressor__model__max_depth': np.arange(1,16),
          'regressor__model__criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
          'regressor__model__min_samples_leaf': [0.1,0.3,0.5,0.7,0.9,1,2,3,4,5,6,7,8,9],
          'regressor__model__min_samples_split': [0.1,0.3,0.5,0.7,0.9,2,3,4,5,6,7,8,9],
          'transformer': [None,StandardScaler(),MinMaxScaler(),RobustScaler()]}

#--------------------- SVR
myparams={'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler()],
          'regressor__model__kernel': ['linear','poly','rbf','sigmoid'],
          'regressor__model__C': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100],
          'regressor__model__gamma': ['scale','auto'],
          'regressor__model__epsilon':[0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
          'transformer': [None,StandardScaler(),MinMaxScaler(),RobustScaler()]}



#------------------------ MLP
myparams={'regressor__scaler': [None,Normalizer(),MinMaxScaler(),StandardScaler()],
          'regressor__model__solver': ['adam','sgd'],
          'regressor__model__hidden_layer_sizes': [(10,),(100,),(1,1),(2,1),(3,1),(4,1),(5,1),(2,2),(3,2),(4,2),(5,2),(6,2),(3,3),(4,3),(5,3),(6,2),(7,2),(10,5),(10,10),(15,10),(100,100),(3,2,1),(4,3,2),(5,4,3),(6,5,4),(7,6,5),(10,5,1),(15,10,5)],
          'regressor__model__activation': ['relu','tanh','logistic','identity'],
          'regressor__model__alpha': [10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,0.1,0.2,0.4,0.6,0.8,1],
          'regressor__model__learning_rate_init': [0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
          'transformer': [None,Normalizer(),MinMaxScaler(),StandardScaler()]}
