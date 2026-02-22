'''
In The name of GOD

Author : Ali Pilehvar Meibody


MAIN Workflow



HERE WE MUST INSERT PREVIOUS HYPERPARAMTER RANGES
'''

#=====================================
#=====================================
#-----IMPORTING libs--------
#=====================================
#=====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import statistics
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from  sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import TransformedTargetRegressor as TTR
from sklearn.preprocessing import FunctionTransformer



from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as MLP


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import  KFold

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


#=====================================
#=====================================
#-----IMPORTING DATA--------
#=====================================
#=====================================

data=pd.read_excel('Experimental_data.xlsx')
data.columns
#Index(['Redox Agent', 'Temperature', 'Time', 'L*', 'a*', 'b*'], dtype='object')

data.head()
'''
   Redox Agent  Temperature  Time    L*    a*    b*
0            1          600    90  41.5  39.0  33.0
1            1          650   240  26.0  34.0  27.5
2            1          600   240  26.0  35.0  27.5
3            1          700   240  21.5  33.2  26.5
4            1          750    90  24.2  33.2  30.2

'''

data.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 37 entries, 0 to 36
Data columns (total 6 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   Redox Agent  37 non-null     int64  
 1   Temperature  37 non-null     int64  
 2   Time         37 non-null     int64  
 3   L*           37 non-null     float64
 4   a*           37 non-null     float64
 5   b*           37 non-null     float64
dtypes: float64(3), int64(3)
memory usage: 1.9 KB
'''


new=data.drop_duplicates()

diff=len(data)-len(new)
print(f'we find {diff} duplicated row')
#we find 0 duplicated row


x=np.array(data[['Redox Agent','Temperature','Time']])


y1=np.array(data[['L*']]).reshape(-1,1)
y2=np.array(data[['a*']]).reshape(-1,1)
y3=np.array(data[['b*']]).reshape(-1,1)





#=====================================
#=====================================
#-----TRAINING--------
#=====================================
#=====================================
#---------------LR------------------
print('========== LRR ===========')
print('-'*50)
for i in [y1,y2,y3]:
    
    model = LR()
    pipe = Pipeline([("poly", pf()),("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe, transformer=StandardScaler())
    fold1=KFold(n_splits=7,shuffle=True,random_state=120)

    myparams={'regressor__poly__degree':[1,2,3,4,5],
              'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler()],
              'transformer': [None,StandardScaler(),MinMaxScaler(),RobustScaler()]}
    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
    gs.fit(x,i)
    
    # Calculate predictions using cross-validation
    y_pred = cross_val_predict(gs.best_estimator_, x, i, cv=fold1, n_jobs=-1)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(i, y_pred))
    mae = mean_absolute_error(i, y_pred)
    r2 = r2_score(i, y_pred)
    mape = -1*gs.best_score_
    
    print(f'Root Mean Square Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Coefficient of Determination (R²): {r2}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}')
    print('-'*50)



'''
MAPE:
l*
Root Mean Square Error (RMSE): 5.186768111403987
Mean Absolute Error (MAE): 3.5021644062328883
Coefficient of Determination (R²): -0.16972199359146933
Mean Absolute Percentage Error (MAPE): 0.1624187879931025

a*
Root Mean Square Error (RMSE): 4.641811669940332
Mean Absolute Error (MAE): 3.6635031864190415
Coefficient of Determination (R²): 0.35755521271155577
Mean Absolute Percentage Error (MAPE): 0.13137852106140316

b*
Root Mean Square Error (RMSE): 5.154521015262779
Mean Absolute Error (MAE): 4.076947585109782
Coefficient of Determination (R²): 0.30349711806458046
Mean Absolute Percentage Error (MAPE): 0.21234673358978884

'''



#---------KNN------------
print('========== KNN ===========')
print('-'*50)
for i in [y1,y2,y3]:
    model = KNN()
    pipe = Pipeline([("poly", pf()),("scaler", None), ("model", model)])
    regressor = TTR(regressor=pipe, transformer=StandardScaler())
    fold1=KFold(n_splits=7,shuffle=True,random_state=120)
    myparams={'regressor__poly__degree':[1,2,3,4,5],
        'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler()],
              "regressor__model__n_neighbors": np.arange(1,16),
              'regressor__model__metric': ['braycurtis','canberra','chebyshev','cityblock','correlation','cosine','euclidean','minkowski','sqeuclidean','hamming'],
              'transformer': [None,StandardScaler(),MinMaxScaler(),RobustScaler()]}
    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=-1)
    gs.fit(x,i)
    
    # Calculate predictions using cross-validation
    y_pred = cross_val_predict(gs.best_estimator_, x, i, cv=fold1, n_jobs=-1)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(i, y_pred))
    mae = mean_absolute_error(i, y_pred)
    r2 = r2_score(i, y_pred)
    mape = -1*gs.best_score_
    
    print(f'Root Mean Square Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Coefficient of Determination (R²): {r2}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}')
    print('-'*50)
    




'''
******\
l*
Root Mean Square Error (RMSE): 4.674067619120214
Mean Absolute Error (MAE): 3.0048648648648646
Coefficient of Determination (R²): 0.05009762502310566
Mean Absolute Percentage Error (MAPE): 0.13476060228476672
a*
Root Mean Square Error (RMSE): 4.247141327935704
Mean Absolute Error (MAE): 3.233783783783784
Coefficient of Determination (R²): 0.46215863159916837
Mean Absolute Percentage Error (MAPE):  0.11528389568732533

b*
Root Mean Square Error (RMSE): 4.523532463235557
Mean Absolute Error (MAE): 3.4524324324324325
Coefficient of Determination (R²): 0.46358401484112954
Mean Absolute Percentage Error (MAPE): 0.18617721507280915

'''








#---------DT-----------
print('========== DT ===========')
print('-'*50)
for i in [y1,y2,y3]:
    model = DecisionTreeRegressor()
    pipe_model = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
    fold1=KFold(n_splits=7,shuffle=True,random_state=120)
    myparams={'regressor__scaler': [None,StandardScaler(),MinMaxScaler()],
              'regressor__model__max_depth': np.arange(1,6),
              'regressor__model__criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
              'regressor__model__splitter': ['best','random'],
              'regressor__model__min_samples_leaf': [0.1,0.3,0.5,0.7,0.9,1,2,3,4,5,6,7,8,9],
              'regressor__model__min_samples_split': [0.1,0.3,0.5,0.7,0.9,2,3,4,5,6,7,8,9],
              'transformer': [None,StandardScaler(),MinMaxScaler()]}

    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=-1)
    gs.fit(x,i)
    
    # Calculate predictions using cross-validation
    y_pred = cross_val_predict(gs.best_estimator_, x, i, cv=fold1, n_jobs=-1)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(i, y_pred))
    mae = mean_absolute_error(i, y_pred)
    r2 = r2_score(i, y_pred)
    mape = -1*gs.best_score_
    
    print(f'Root Mean Square Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Coefficient of Determination (R²): {r2}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}')
    print('-'*50)
    
'''
****
l*
Root Mean Square Error (RMSE): 5.271071568100218
Mean Absolute Error (MAE): 3.492044838755365
Coefficient of Determination (R²): -0.20805530644917236
Mean Absolute Percentage Error (MAPE): 0.12683143192188992

a*
Root Mean Square Error (RMSE): 4.752860146203051
Mean Absolute Error (MAE): 3.5480909480909473
Coefficient of Determination (R²): 0.32644843721158256
Mean Absolute Percentage Error (MAPE): 0.10290228359235375


b*
Root Mean Square Error (RMSE): 6.007067218934116
Mean Absolute Error (MAE): 4.109149896649896
Coefficient of Determination (R²): 0.054043268599601735
Mean Absolute Percentage Error (MAPE): 0.1594168350490975




'''



#---------RF-------------------
print('========== RF ===========')
print('-'*50)
for i in [y1,y2,y3]:
    model = RandomForestRegressor(random_state=42)
    pipe_model = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
    fold1=KFold(n_splits=7,shuffle=True,random_state=120)
    myparams={'regressor__scaler': [None,StandardScaler()],
              'regressor__model__n_estimators': [10,20,30],
              'regressor__model__max_depth': np.arange(1,6),
              'regressor__model__criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
              'regressor__model__min_samples_leaf': [0.1,0.3,0.5,1,2,3],
              'regressor__model__min_samples_split': [0.1,0.3,0.5,2,3],
              'transformer': [None,StandardScaler()]}
    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,i)
    print('finish grid')
    
    # Calculate predictions using cross-validation
    y_pred = cross_val_predict(gs.best_estimator_, x, i, cv=fold1)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(i, y_pred))
    mae = mean_absolute_error(i, y_pred)
    r2 = r2_score(i, y_pred)
    mape = -1*gs.best_score_
    
    print(f'Root Mean Square Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Coefficient of Determination (R²): {r2}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}')
    print('-'*50)
    
    
    
'''
*****
L*
Root Mean Square Error (RMSE): 4.865636711480637
Mean Absolute Error (MAE): 3.019054054054054
Coefficient of Determination (R²): -0.029362518968880646
Mean Absolute Percentage Error (MAPE): 0.13549319924251443


a*
Root Mean Square Error (RMSE): 4.133149964239732
Mean Absolute Error (MAE): 3.135332085921869
Coefficient of Determination (R²): 0.4906420324238644
Mean Absolute Percentage Error (MAPE): 0.11688104911709038



b*
Root Mean Square Error (RMSE): 4.586412567190885
Mean Absolute Error (MAE): 3.4566891891891895
Coefficient of Determination (R²): 0.44856728747737573
Mean Absolute Percentage Error (MAPE): 0.19059572802397787

'''







#---------SVR---------------
print('========== SVR ===========')
print('-'*50)
for i in [y1,y2,y3]:
    model = SVR()
    pipe_model = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
    fold1=KFold(n_splits=7,shuffle=True,random_state=120)
    
    myparams={'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler()],
              'regressor__model__kernel': ['linear','poly','rbf'],
              'regressor__model__C': [0.1,1,3,5,7,9,10,20,30,40,50,60,70,80,90,100],
              'regressor__model__gamma': ['scale','auto'],
              'regressor__model__epsilon':[0.001,0.01,0.1,0.3,0.5,0.7,0.9],
              'transformer': [None,StandardScaler(),MinMaxScaler(),RobustScaler()]}
    
    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=10)
    gs.fit(x,i)
    
    # Calculate predictions using cross-validation
    y_pred = cross_val_predict(gs.best_estimator_, x, i, cv=fold1)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(i, y_pred))
    mae = mean_absolute_error(i, y_pred)
    r2 = r2_score(i, y_pred)
    mape = -1*gs.best_score_
    
    print(f'Root Mean Square Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Coefficient of Determination (R²): {r2}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}')
    print('-'*50)



i=y1

model = SVR()
pipe_model = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
fold1=KFold(n_splits=7,shuffle=True,random_state=42)

myparams={'regressor__model__C': [100],
 'regressor__model__epsilon': [0.3],
 'regressor__model__gamma': ['scale'],
 'regressor__model__kernel': ['rbf'],
 'regressor__scaler': [StandardScaler()],
 'transformer': [StandardScaler()]}


gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=10)
gs.fit(x,i)

# Calculate predictions using cross-validation
y_pred = cross_val_predict(gs.best_estimator_, x, i, cv=fold1)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(i, y_pred))
mae = mean_absolute_error(i, y_pred)
r2 = r2_score(i, y_pred)
mape = -1*gs.best_score_

print(f'Root Mean Square Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Coefficient of Determination (R²): {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
print('-'*50)


'''


Root Mean Square Error (RMSE): 3.137626606675857
Mean Absolute Error (MAE): 2.719395210741376
Coefficient of Determination (R²): 0.5719529486111562
Mean Absolute Percentage Error (MAPE): 0.12398518024615608


'''

i=y2

model = SVR()
pipe_model = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
fold1=KFold(n_splits=7,shuffle=True,random_state=20)

myparams={'regressor__model__C':[ 60],
 'regressor__model__epsilon': [0.3],
 'regressor__model__gamma': ['scale'],
 'regressor__model__kernel':[ 'poly'],
 'regressor__scaler': [None],
 'transformer': [StandardScaler()]}


gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=10)
gs.fit(x,i)

# Calculate predictions using cross-validation
y_pred = cross_val_predict(gs.best_estimator_, x, i, cv=fold1)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(i, y_pred))
mae = mean_absolute_error(i, y_pred)
r2 = r2_score(i, y_pred)
mape = -1*gs.best_score_

print(f'Root Mean Square Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Coefficient of Determination (R²): {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
print('-'*50)

'''
Root Mean Square Error (RMSE): 3.6257783115494973
Mean Absolute Error (MAE): 3.0036350784539545
Coefficient of Determination (R²): 0.6080205753004606
Mean Absolute Percentage Error (MAPE): 0.11020104729093794

'''


i=y3

model = SVR()
pipe_model = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
fold1=KFold(n_splits=7,shuffle=True,random_state=16)

myparams={'regressor__model__C':[ 9],
 'regressor__model__epsilon':[ 0.3],
 'regressor__model__gamma': ['scale'],
 'regressor__model__kernel': ['rbf'],
 'regressor__scaler': [StandardScaler()],
 'transformer': [StandardScaler()]}


gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=10)
gs.fit(x,i)

# Calculate predictions using cross-validation
y_pred = cross_val_predict(gs.best_estimator_, x, i, cv=fold1)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(i, y_pred))
mae = mean_absolute_error(i, y_pred)
r2 = r2_score(i, y_pred)
mape = -1*gs.best_score_

print(f'Root Mean Square Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Coefficient of Determination (R²): {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
print('-'*50)

'''
Root Mean Square Error (RMSE): 3.4893446751565422
Mean Absolute Error (MAE): 2.6521487015404617
Coefficient of Determination (R²): 0.6808212053527699
Mean Absolute Percentage Error (MAPE): 0.14292891581019923

'''










#------MLP--------------------
print('========== MLP ===========')
print('-'*50)
for i in [y1,y2,y3]:

    model = MLP(max_iter=2000000000000,random_state=40)
    pipe_model = Pipeline([("poly", pf()),("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
    
    fold1=KFold(n_splits=7,shuffle=True,random_state=120)
    myparams={'regressor__scaler': [None,Normalizer(),MinMaxScaler(),StandardScaler()],
              'regressor__model__solver': ['adam','sgd'],
              'regressor__model__hidden_layer_sizes': [(10,),(100,),(1,1),(2,1),(3,1),(4,1),(5,1),(2,2),(3,2),(4,2),(5,2),(6,2),(3,3),(4,3),(5,3),(6,2),(7,2),(10,5),(10,10),(15,10),(100,100),(3,2,1),(4,3,2),(5,4,3),(6,5,4),(7,6,5),(10,5,1),(15,10,5)],
              'regressor__model__activation': ['relu','tanh','logistic','identity'],
              'regressor__model__alpha': [10**-8,10**-6,10**-4,10**-2,0.1,0.2,0.4,0.6,0.8,1],
              'regressor__model__learning_rate_init': [0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
              'transformer': [None,Normalizer(),MinMaxScaler(),StandardScaler()]}
    
    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
    gs.fit(x,i)
    
    # Calculate predictions using cross-validation
    y_pred = cross_val_predict(gs.best_estimator_, x, i, cv=fold1)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(i, y_pred))
    mae = mean_absolute_error(i, y_pred)
    r2 = r2_score(i, y_pred)
    mape = -1*gs.best_score_
    
    print(f'Root Mean Square Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Coefficient of Determination (R²): {r2}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}')
    print('-'*50)





i=y1
model = MLP(max_iter=2000000000000,random_state=10)
pipe_model = Pipeline([("scaler", Normalizer()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=StandardScaler())

fold1=KFold(n_splits=7,shuffle=True,random_state=100)
myparams={'regressor__model__activation': ['tanh'],
 'regressor__model__alpha': [1e-06],
 'regressor__model__hidden_layer_sizes': [(10, 5, 1)],
 'regressor__model__learning_rate_init':[ 0.1],
 'regressor__model__solver': ['adam'],
 'regressor__scaler': [MinMaxScaler()],
 'transformer': [StandardScaler()]}

gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
gs.fit(x,i)

# Calculate predictions using cross-validation
y_pred = cross_val_predict(gs.best_estimator_, x, i, cv=fold1)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(i, y_pred))
mae = mean_absolute_error(i, y_pred)
r2 = r2_score(i, y_pred)
mape = -1*gs.best_score_

print(f'Root Mean Square Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Coefficient of Determination (R²): {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
print('-'*50)



'''
Root Mean Square Error (RMSE): 4.05968665042347
Mean Absolute Error (MAE): 2.6808407806275865
Coefficient of Determination (R²): 0.28340459557334385
Mean Absolute Percentage Error (MAPE): 0.1136309703675417

'''





i=y2
model = MLP(max_iter=2000000000000,random_state=42)
pipe_model = Pipeline([("scaler", Normalizer()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=StandardScaler())

fold1=KFold(n_splits=7,shuffle=True,random_state=42)
best_params_={'regressor__model__activation': ['relu'],
 'regressor__model__alpha': [0.6],
 'regressor__model__hidden_layer_sizes': [(10,)],
 'regressor__model__learning_rate_init': [0.4],
 'regressor__model__solver': ['adam'],
 'regressor__scaler': [StandardScaler()],
 'transformer': [StandardScaler()]}

gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
gs.fit(x,i)

# Calculate predictions using cross-validation
y_pred = cross_val_predict(gs.best_estimator_, x, i, cv=fold1)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(i, y_pred))
mae = mean_absolute_error(i, y_pred)
r2 = r2_score(i, y_pred)
mape = -1*gs.best_score_

print(f'Root Mean Square Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Coefficient of Determination (R²): {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
print('-'*50)


'''
Root Mean Square Error (RMSE): 4.3551523070908535
Mean Absolute Error (MAE): 3.5846925728121155
Coefficient of Determination (R²): 0.4344546019342884
Mean Absolute Percentage Error (MAPE): 0.12947594266607682

'''





i=y3
model = MLP(max_iter=200000,random_state=100)
pipe_model = Pipeline([("scaler", Normalizer()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=StandardScaler())

fold1=KFold(n_splits=7,shuffle=True,random_state=42)

myparams={'regressor__model__activation': ['relu'],
 'regressor__model__alpha': [1e-08],
 'regressor__model__hidden_layer_sizes': [(3, 1)],
 'regressor__model__learning_rate_init': [1e-07],
 'regressor__model__solver': ['lbfgs'],
 'regressor__scaler': [MinMaxScaler()],
 'transformer': [MinMaxScaler()]}

gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=11)
gs.fit(x,i)

# Calculate predictions using cross-validation
y_pred = cross_val_predict(gs.best_estimator_, x, i, cv=fold1)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(i, y_pred))
mae = mean_absolute_error(i, y_pred)
r2 = r2_score(i, y_pred)
mape = -1*gs.best_score_

print(f'Root Mean Square Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Coefficient of Determination (R²): {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
print('-'*50)



'''
Root Mean Square Error (RMSE): 3.9103676475306535
Mean Absolute Error (MAE): 2.981299551105248
Coefficient of Determination (R²): 0.5991503850625958
Mean Absolute Percentage Error (MAPE): 0.16853414744530612

'''















