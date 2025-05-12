'''
In The name of GOD


Author : Ali Pilehvar Meibody

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


#=====================================
#=====================================
#-----IMPORTING DATA--------
#=====================================
#=====================================

data=pd.read_excel('Experimental_data.xlsx')
data.columns
#Index(['Redox Agent', 'Temperature', 'Time', 'L*', 'a*', 'b*'], dtype='object')

data.head()



data.info()


new=data.drop_duplicates()

diff=len(data)-len(new)
print(f'we find {diff} duplicated row')



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

for i in [y1,y2,y3]:
    model = LR()
    pipe = Pipeline([("poly", pf()),("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe, transformer=StandardScaler())
    fold1=KFold(n_splits=7,shuffle=True,random_state=120)

    myparams={'regressor__poly__degree':[1,2,3,4,5],
              'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler()],
              'transformer': [None,StandardScaler(),MinMaxScaler(),RobustScaler()]}
    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error')
    gs.fit(x,i)
    print(1+gs.best_score_)



'''
MAPE:
l*
a*
b*

0.8375812120068975
0.8686214789385969
0.7876532664102112

'''



#---------KNN------------
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
    print(1+gs.best_score_)


'''
******\
l*
a*
b*

0.8652393977152333
0.8847161043126747
0.8138227849271908


'''








#---------DT-----------
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
    print(1+gs.best_score_)
    
'''
****


'''



#---------RF-------------------
for i in [y1,y2,y3]:
    model = RandomForestRegressor(random_state=42,n_jobs=-1)
    pipe_model = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
    fold1=KFold(n_splits=7,shuffle=True,random_state=120)
    myparams={'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler()],
              'regressor__model__n_estimators': np.arange(1,11),
              'regressor__model__max_depth': np.arange(1,16),
              'regressor__model__criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
              'regressor__model__min_samples_leaf': [0.1,0.3,0.5,0.7,0.9,1,2,3,4,5,6,7,8,9],
              'regressor__model__min_samples_split': [0.1,0.3,0.5,0.7,0.9,2,3,4,5,6,7,8,9],
              'transformer': [None,StandardScaler(),MinMaxScaler(),RobustScaler()]}
    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=-1)
    gs.fit(x,i)
    print(1+gs.best_score_)
    
    
    
'''
*****




'''







#---------SVR---------------
for i in [y1,y2,y3]:
    model = SVR()
    pipe_model = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
    fold1=KFold(n_splits=7,shuffle=True,random_state=120)
    
    myparams=myparams={'regressor__scaler': [None,StandardScaler(),MinMaxScaler(),RobustScaler()],
              'regressor__model__kernel': ['linear','poly','rbf','sigmoid'],
              'regressor__model__C': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100],
              'regressor__model__gamma': ['scale','auto'],
              'regressor__model__epsilon':[0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
              'transformer': [None,StandardScaler(),MinMaxScaler(),RobustScaler()]}
    
    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=-1)
    gs.fit(x,i)
    print(1+gs.best_score_)



#------MLP--------------------
for i in [y1,y2,y3]:

    model = MLP(max_iter=2000000000000,random_state=40)
    pipe_model = Pipeline([("poly", pf()),("scaler", MinMaxScaler()), ("model", model)])
    regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
    
    fold1=KFold(n_splits=7,shuffle=True,random_state=120)
    myparams=myparams={'regressor__scaler': [None,Normalizer(),MinMaxScaler(),StandardScaler()],
              'regressor__model__solver': ['adam','sgd'],
              'regressor__model__hidden_layer_sizes': [(10,),(100,),(1,1),(2,1),(3,1),(4,1),(5,1),(2,2),(3,2),(4,2),(5,2),(6,2),(3,3),(4,3),(5,3),(6,2),(7,2),(10,5),(10,10),(15,10),(100,100),(3,2,1),(4,3,2),(5,4,3),(6,5,4),(7,6,5),(10,5,1),(15,10,5)],
              'regressor__model__activation': ['relu','tanh','logistic','identity'],
              'regressor__model__alpha': [10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,0.1,0.2,0.4,0.6,0.8,1],
              'regressor__model__learning_rate_init': [0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
              'transformer': [None,Normalizer(),MinMaxScaler(),StandardScaler()]}
    
    gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=-1)
    gs.fit(x,i)
    print(1+gs.best_score)

'''
****



'''



    
