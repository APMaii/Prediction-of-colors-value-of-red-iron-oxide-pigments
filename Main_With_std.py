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


def print_cv_metrics(estimator, X, y, cv, n_jobs=-1):
    y_pred = cross_val_predict(estimator, X, y, cv=cv, n_jobs=n_jobs)

    rmse_agg = np.sqrt(mean_squared_error(y, y_pred))
    mae_agg = mean_absolute_error(y, y_pred)
    r2_agg = r2_score(y, y_pred)
    mape_agg = mean_absolute_percentage_error(y, y_pred)

    rmse_list, mae_list, r2_list, mape_list = [], [], [], []
    for train_idx, test_idx in cv.split(X):
        y_true_fold = y[test_idx]
        y_pred_fold = y_pred[test_idx]
        rmse_list.append(np.sqrt(mean_squared_error(y_true_fold, y_pred_fold)))
        mae_list.append(mean_absolute_error(y_true_fold, y_pred_fold))
        r2_list.append(r2_score(y_true_fold, y_pred_fold))
        mape_list.append(mean_absolute_percentage_error(y_true_fold, y_pred_fold))

    print(f'Root Mean Square Error (RMSE): {rmse_agg} ± {np.std(rmse_list)}')
    print(f'Mean Absolute Error (MAE): {mae_agg} ± {np.std(mae_list)}')
    print(f'Coefficient of Determination (R²): {r2_agg} ± {np.std(r2_list)}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape_agg} ± {np.std(mape_list)}')
    print('-'*50)


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
    
    print_cv_metrics(gs.best_estimator_, x, i, fold1)



'''
L*
--------------------------------------------------
Root Mean Square Error (RMSE): 5.186768111403987 ± 2.1298297799458106
Mean Absolute Error (MAE): 3.5021644062328883 ± 1.2188842868417489
Coefficient of Determination (R²): -0.16972199359146933 ± 0.4722189058133985
Mean Absolute Percentage Error (MAPE): 0.1624187879931025 ± 0.07293849690830818

a*
--------------------------------------------------
Root Mean Square Error (RMSE): 4.641811669940332 ± 1.833979933333946
Mean Absolute Error (MAE): 3.6635031864190415 ± 1.5505655794535276
Coefficient of Determination (R²): 0.35755521271155577 ± 0.9755081193807643
Mean Absolute Percentage Error (MAPE): 0.13137852106140316 ± 0.0678693444777195

b*
--------------------------------------------------
Root Mean Square Error (RMSE): 5.154521015262779 ± 1.8387448812298919
Mean Absolute Error (MAE): 4.076947585109782 ± 1.5501824691593693
Coefficient of Determination (R²): 0.30349711806458046 ± 1.4191479030756047
Mean Absolute Percentage Error (MAPE): 0.21234673358978884 ± 0.11878892082052618
--------------------------------------------------


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
    
    print_cv_metrics(gs.best_estimator_, x, i, fold1)
    




'''
========== KNN ===========
L*
--------------------------------------------------
Root Mean Square Error (RMSE): 4.674067619120214 ± 2.065797795255245
Mean Absolute Error (MAE): 3.0048648648648646 ± 0.9319235903395152
Coefficient of Determination (R²): 0.05009762502310566 ± 0.31113417557784173
Mean Absolute Percentage Error (MAPE): 0.13476060228476672 ± 0.053556165076990424

a*
--------------------------------------------------
Root Mean Square Error (RMSE): 4.247141327935704 ± 1.8018960392215688
Mean Absolute Error (MAE): 3.233783783783784 ± 1.4589613905111634
Coefficient of Determination (R²): 0.46215863159916837 ± 0.5806182642734152
Mean Absolute Percentage Error (MAPE): 0.11528389568732533 ± 0.08013583609674253

b*
--------------------------------------------------
Root Mean Square Error (RMSE): 4.523532463235557 ± 1.5423565200392861
Mean Absolute Error (MAE): 3.4524324324324325 ± 1.049643438001103
Coefficient of Determination (R²): 0.46358401484112954 ± 0.6508863295374269
Mean Absolute Percentage Error (MAPE): 0.18617721507280915 ± 0.12256368658687272
--------------------------------------------------



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
    
    print_cv_metrics(gs.best_estimator_, x, i, fold1)
    
'''
****
l*
--------------------------------------------------
Root Mean Square Error (RMSE): 5.271071568100218 ± 2.653649001190861
Mean Absolute Error (MAE): 3.492044838755365 ± 1.455113783334135
Coefficient of Determination (R²): -0.20805530644917236 ± 0.5855533822239357
Mean Absolute Percentage Error (MAPE): 0.12683143192188992 ± 0.07517986134945087
--------------------------------------------------


a*
--------------------------------------------------
Root Mean Square Error (RMSE): 4.752860146203051 ± 1.9100762165176524
Mean Absolute Error (MAE): 3.5480909480909473 ± 1.7398729714036834
Coefficient of Determination (R²): 0.32644843721158256 ± 0.5568429342597367
Mean Absolute Percentage Error (MAPE): 0.10290228359235375 ± 0.0942938882339527
--------------------------------------------------

b*
Root Mean Square Error (RMSE): 6.007067218934116 ± 1.6037053006316333
Mean Absolute Error (MAE): 4.109149896649896 ± 1.1643479662510554
Coefficient of Determination (R²): 0.054043268599601735 ± 0.6783008808523809
Mean Absolute Percentage Error (MAPE): 0.1594168350490975 ± 0.12626092030620867
--------------------------------------------------




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
    
    print_cv_metrics(gs.best_estimator_, x, i, fold1)
    
    
    
'''
========== RF ===========
--------------------------------------------------
finish grid
Root Mean Square Error (RMSE): 4.865636711480637 ± 2.3350629161637975
Mean Absolute Error (MAE): 3.019054054054054 ± 1.2135648387761697
Coefficient of Determination (R²): -0.029362518968880646 ± 0.4527160273993358
Mean Absolute Percentage Error (MAPE): 0.13549319924251443 ± 0.0608004425630077
--------------------------------------------------
finish grid
Root Mean Square Error (RMSE): 4.133149964239732 ± 1.678904412269217
Mean Absolute Error (MAE): 3.135332085921869 ± 1.3536260713184434
Coefficient of Determination (R²): 0.4906420324238644 ± 0.6083405830842002
Mean Absolute Percentage Error (MAPE): 0.11688104911709038 ± 0.08227704539394441
--------------------------------------------------
finish grid
Root Mean Square Error (RMSE): 4.586412567190885 ± 1.6683336031851903
Mean Absolute Error (MAE): 3.4566891891891895 ± 1.2069780782388746
Coefficient of Determination (R²): 0.44856728747737573 ± 0.57608726979281
Mean Absolute Percentage Error (MAPE): 0.19059572802397787 ± 0.13005964316831778

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
    
    print_cv_metrics(gs.best_estimator_, x, i, fold1)


'''

Root Mean Square Error (RMSE): 3.137626606675857 ± 0.7809400976451418
Mean Absolute Error (MAE): 2.719395210741376 ± 0.7119871562227832
Coefficient of Determination (R²): 0.5719529486111562 ± 0.6042373328841402
Mean Absolute Percentage Error (MAPE): 0.12398518024615608 ± 0.0403826868068676
--------------------------------------------------

'''


'''
Root Mean Square Error (RMSE): 3.6257783115494973 ± 0.9990888393267315
Mean Absolute Error (MAE): 3.0036350784539545 ± 0.8877552676764799
Coefficient of Determination (R²): 0.6080205753004606 ± 0.23463237365753545
Mean Absolute Percentage Error (MAPE): 0.11020104729093794 ± 0.043869777190980085
--------------------------------------------------

'''


'''
Root Mean Square Error (RMSE): 3.4893446751565422 ± 1.1174415533393909
Mean Absolute Error (MAE): 2.6521487015404617 ± 0.907223054534789
Coefficient of Determination (R²): 0.6808212053527699 ± 0.2609218753602334
Mean Absolute Percentage Error (MAPE): 0.14292891581019923 ± 0.06327034557555494
--------------------------------------------------
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
    
    print_cv_metrics(gs.best_estimator_, x, i, fold1)



'''
Root Mean Square Error (RMSE): 4.05968665042347 ± 1.7214519393424546
Mean Absolute Error (MAE): 2.6808407806275865 ± 0.8295978188400385
Coefficient of Determination (R²): 0.28340459557334385 ± 0.36432713625636626
Mean Absolute Percentage Error (MAPE): 0.1136309703675417 ± 0.03633825003374096
--------------------------------------------------

'''



'''
Root Mean Square Error (RMSE): 4.3551523070908535 ± 1.0106010616033225
Mean Absolute Error (MAE): 3.5846925728121155 ± 0.8659242709383688
Coefficient of Determination (R²): 0.4344546019342884 ± 1.0210815635542017
Mean Absolute Percentage Error (MAPE): 0.12947594266607682 ± 0.05350103801404911
--------------------------------------------------

'''





'''
Root Mean Square Error (RMSE): 3.9103676475306535 ± 1.2827897552650502
Mean Absolute Error (MAE): 2.981299551105248 ± 0.937327475591874
Coefficient of Determination (R²): 0.5991503850625958 ± 0.40986506105311843
Mean Absolute Percentage Error (MAPE): 0.16853414744530612 ± 0.09688577746343935
--------------------------------------------------

'''
