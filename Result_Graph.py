'''
In The Name of GOD
Author : Ali Pilehvar Meibody

'''
#===========================================================
#===========================================================
'Importing Libs'
#===========================================================
#===========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import statistics
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from  sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.pipeline import Pipeline as PPL
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.compose import TransformedTargetRegressor as TTR
#from sklearn.preprocessing import PowerTransformer
#from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import PowerTransformer

from sklearn.compose import TransformedTargetRegressor as TTR
from sklearn.preprocessing import FunctionTransformer

from mpl_toolkits.mplot3d import Axes3D




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
#===========================================================
'Dispersion'
#===========================================================
#===========================================================

fig = plt.figure(figsize = (12,12))
ax = plt.axes(projection='3d')
ax.grid()
d3=X[:,0] #addition
d1 = X[:,1] #temp
d2 = X[:,2] #time

ax.plot3D(d3, d2, d1,'om')
ax.set_title('Data Space',size=17, fontweight='bold')
# Set axes label
ax.set_zlabel('Temperature', labelpad=20,size=17, fontweight='bold')
ax.set_ylabel('Time', labelpad=20,size=17, fontweight='bold')
ax.set_xlabel('Redox Agent', labelpad=20,size=17, fontweight='bold')
plt.show()




#===========================================================
#===========================================================
'Histograms'
#===========================================================
#===========================================================

#basic one-------------
# Plot histograms for each feature
data.hist(figsize=(20, 16))
plt.show()



#advanced --------------
# Plot histograms
fig, axes = plt.subplots(figsize=(20, 16))
axes = data.hist(ax=axes)

# Set big and bold titles for each subplot
for ax in axes.flatten():
    ax.set_title(ax.get_title(), fontsize=20, fontweight='bold')

plt.tight_layout()  # Adjusts layout to prevent overlap
plt.show()




#===========================================================
#===========================================================
'Box plots'
#===========================================================
#===========================================================
#basic------------------
# Boxplots to visualize outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=data)
plt.show()


#advanced-------------------
plt.figure(figsize=(10, 6))
sns.boxplot(data=data)

# Set title with bold font
#plt.title('Boxplot of Features', fontsize=18, fontweight='bold')

# Set x-axis and y-axis labels
plt.xlabel('Features', fontsize=14, fontweight='bold')
plt.ylabel('Value', fontsize=14, fontweight='bold')

plt.show()




#===========================================================
'Correlation'
#===========================================================


plt.figure(figsize=(10, 6))  # Set the figure size (width: 10 inches, height: 6 inches)
plt.rcParams["figure.dpi"] = 600 
plt.rcParams['font.family']='Arial'

correlation = data.corr()  
#xt=['Add','Temp','Time','L','a','b']
#yt=['Add','Temp','Time','L','a','b']
#xticklabels=xt,yticklabels=yt
sns.heatmap(correlation,cmap="coolwarm",annot=True)
plt.tick_params(labelsize=16,pad=12)

name='correlation.jpg'
plt.show()
plt.savefig(name,dpi=600,format='jpg')



#---------Pearson correlation--------
correlation_matrix = data.corr()

# Plot heatmap for Pearson correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Pearson Correlation Heatmap')
plt.show()





#===========================================================
#===========================================================
'PAIR PLOT'
#===========================================================
#===========================================================

sns.pairplot(data)
plt.show()




#===========================================================
#===========================================================
'3D Plot for final results'
#===========================================================
#===========================================================


#===========================================================
'Generate space for process parameters'
#===========================================================


def generate_bumpy():
    a1_values = np.array([0, 1, 2])
    a2_values = np.arange(0, 1001, 10)
    a3_values = np.arange(60, 241, 1)

    a1, a2, a3 = np.meshgrid(a1_values, a2_values, a3_values, indexing='ij')

    result = np.column_stack((a1.flatten(), a2.flatten(), a3.flatten()))

    return result

input_space = generate_bumpy()






#===========================================================
'Best Supervised Algorithem with bEST Hyperparameters'
#===========================================================


#---------------------
#svr
#---------------------

#===========================================================
'L* WITH best MLP'
#===========================================================

#L*
#MLP
yy=L
yyy='L*'
mm='MLP'

model = MLP(max_iter=2000000000000,random_state=10)
pipe_model = PPL([("scaler", Normalizer()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
fold2=KFold(n_splits=7,shuffle=True,random_state=100)
myparams={'regressor__model__activation': ['tanh'],
 'regressor__model__alpha': [1e-06],
 'regressor__model__hidden_layer_sizes': [(10, 5, 1)],
 'regressor__model__learning_rate_init':[ 0.1],
 'regressor__model__solver': ['adam'],
 'regressor__scaler': [MinMaxScaler()],
 'transformer': [StandardScaler()]}
gs = GridSearchCV(regressor, param_grid=myparams, cv=fold2, scoring='neg_mean_absolute_percentage_error',n_jobs=-1)
gs.fit(X,yy)

l_yp=gs.predict(input_space)
scoree=gs.best_score_

print(1+scoree)




print(len(xx1), len(xx2), len(yy3))  # Check lengths






fig = plt.figure(figsize=(12, 10))
norm = plt.Normalize(yy3.min(), yy3.max())  # Normalizing colors based on yy3 values

ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(xx1, xx2, yy3, c=yy3, cmap='gist_gray', marker='o', norm=norm)

# Set labels
ax.set_xlabel('Temperature', size=12, labelpad=15, fontweight='bold')
ax.set_ylabel('Time', size=15, labelpad=12, fontweight='bold')
ax.set_zlabel(f'{yyy}', size=15, labelpad=12, fontweight='bold')
ax.set_title(f'{yyy} for {add} Redox Agent', size=25, fontweight='bold')

# Add a colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label(f'{yyy}')

# Show the plot
plt.show()



add='No'
if add=='No':
    xx1=input_space[0:18281,1]
    xx2=input_space[0:18281,2]
    yy3=l_yp[0:18281]

fig = plt.figure(figsize=(12,10))
norm = plt.Normalize(yy3.min(), yy3.max())

ax = fig.add_subplot(111, projection='3d')
#scatter = ax.scatter(xx1,xx2,yy3, marker='o')



scatter = ax.scatter(xx1,xx2,yy3,c=yy3, cmap='gist_gray', marker='o', norm=norm)
#c=yy3

# Set labels
ax.set_xlabel('Temperature',size=12,labelpad=15, fontweight='bold')
ax.set_ylabel('time',size=15,labelpad=12, fontweight='bold')
ax.set_zlabel(f'{yyy}',size=15,labelpad=12, fontweight='bold')
ax.set_title(f'{yyy}  for {add} Redox Agent',size=25, fontweight='bold')
# Add a colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label(f'{yyy}')

# Show the plot
plt.show()




add='1x'
if add=='1x':
    xx1=input_space[18281:36562,1]
    xx2=input_space[18281:36562,2]
    yy3=l_yp[18281:36562]
    
fig = plt.figure(figsize=(12,10))
norm = plt.Normalize(yy3.min(), yy3.max())

ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(xx1,xx2,yy3, c=yy3, cmap='gist_gray', marker='o', norm=norm)


# Set labels
ax.set_xlabel('Temperature',size=12,labelpad=15, fontweight='bold')
ax.set_ylabel('time',size=15,labelpad=12, fontweight='bold')
ax.set_zlabel(f'{yyy}',size=15,labelpad=12, fontweight='bold')
ax.set_title(f'{yyy}  for {add} Redox Agent',size=25, fontweight='bold')
# Add a colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label(f'{yyy}')

# Show the plot
plt.show()


add='2x'
if add=='2x':
    xx1=input_space[36562:54843,1]
    xx2=input_space[36562:54843,2]
    yy3=l_yp[36562:54843]




fig = plt.figure(figsize=(12,10))
norm = plt.Normalize(yy3.min(), yy3.max())

ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(xx1,xx2,yy3, c=yy3, cmap='gist_gray', marker='o', norm=norm)


# Set labels
ax.set_xlabel('Temperature',size=12,labelpad=15, fontweight='bold')
ax.set_ylabel('time',size=15,labelpad=12, fontweight='bold')
ax.set_zlabel(f'{yyy}',size=15,labelpad=12, fontweight='bold')
ax.set_title(f'{yyy}  for {add} Redox Agent',size=25, fontweight='bold')
# Add a colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label(f'{yyy}')

# Show the plot
plt.show()



#===========================================================
'a* WITH best SVM'
#===========================================================


##A---------
#also getting with n_splits7
yy=A
yyy='a*'
mm='SVR'
model = SVR()
pipe_model = PPL([("scaler",MinMaxScaler()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
#fold2=KFold(n_splits=10,shuffle=True,random_state=20)
fold2=KFold(n_splits=7,shuffle=True,random_state=20)

myparams={'regressor__model__C':[ 60],
 'regressor__model__epsilon': [0.3],
 'regressor__model__gamma': ['scale'],
 'regressor__model__kernel':[ 'poly'],
 'regressor__scaler': [None],
 'transformer': [StandardScaler()]}
gs = GridSearchCV(regressor, param_grid=myparams, cv=fold2, scoring='neg_mean_absolute_percentage_error',n_jobs=-1)
gs.fit(X,yy)

l_yp=gs.predict(input_space)
scoree=gs.best_score_

print(1+scoree)




add='No'
if add=='No':
    xx1=input_space[0:18281,1]
    xx2=input_space[0:18281,2]
    yy3=l_yp[0:18281]

fig = plt.figure(figsize=(12,10))
norm = plt.Normalize(yy3.min(), yy3.max())

ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(xx1,xx2,yy3, c=yy3, cmap='RdYlGn_r', marker='o', norm=norm)




# Set labels
ax.set_xlabel('Temperature',size=12,labelpad=15, fontweight='bold')
ax.set_ylabel('time',size=15,labelpad=12, fontweight='bold')
ax.set_zlabel(f'{yyy}',size=15,labelpad=12, fontweight='bold')
ax.set_title(f'{yyy}  for {add} Redox Agent',size=25, fontweight='bold')
# Add a colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label(f'{yyy}')

# Show the plot
plt.show()




add='1x'
if add=='1x':
    xx1=input_space[18281:36562,1]
    xx2=input_space[18281:36562,2]
    yy3=l_yp[18281:36562]
    
fig = plt.figure(figsize=(12,10))
norm = plt.Normalize(yy3.min(), yy3.max())

ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(xx1,xx2,yy3, c=yy3, cmap='RdYlGn_r', marker='o', norm=norm)


# Set labels
ax.set_xlabel('Temperature',size=12,labelpad=15, fontweight='bold')
ax.set_ylabel('time',size=15,labelpad=12, fontweight='bold')
ax.set_zlabel(f'{yyy}',size=15,labelpad=12, fontweight='bold')
ax.set_title(f'{yyy}  for {add} Redox Agent',size=25, fontweight='bold')
# Add a colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label(f'{yyy}')

# Show the plot
plt.show()



add='2x'
if add=='2x':
    xx1=input_space[36562:54843,1]
    xx2=input_space[36562:54843,2]
    yy3=l_yp[36562:54843]




fig = plt.figure(figsize=(12,10))
norm = plt.Normalize(yy3.min(), yy3.max())

ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(xx1,xx2,yy3, c=yy3, cmap='RdYlGn_r', marker='o', norm=norm)


# Set labels
ax.set_xlabel('Temperature',size=12,labelpad=15, fontweight='bold')
ax.set_ylabel('time',size=15,labelpad=12, fontweight='bold')
ax.set_zlabel(f'{yyy}',size=15,labelpad=12, fontweight='bold')
ax.set_title(f'{yyy}  for {add} Redox Agent',size=25, fontweight='bold')
# Add a colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label(f'{yyy}')

# Show the plot
plt.show()






#===========================================================
'b* WITH best SVM'
#===========================================================

##B---------
yy=B
yyy='b*'
mm='SVR'
model = SVR()
pipe_model = PPL([("scaler",MinMaxScaler()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
fold2=KFold(n_splits=7,shuffle=True,random_state=16)
myparams={'regressor__model__C':[ 9],
 'regressor__model__epsilon': [0.3],
 'regressor__model__gamma': ['scale'],
 'regressor__model__kernel': ['rbf'],
 'regressor__scaler': [StandardScaler()],
 'transformer': [StandardScaler()]}
gs = GridSearchCV(regressor, param_grid=myparams, cv=fold2, scoring='neg_mean_absolute_percentage_error',n_jobs=-1)
gs.fit(X,yy)

l_yp=gs.predict(input_space)
scoree=gs.best_score_

print(1+scoree)




add='No'
if add=='No':
    xx1=input_space[0:18281,1]
    xx2=input_space[0:18281,2]
    yy3=l_yp[0:18281]

fig = plt.figure(figsize=(12,10))
norm = plt.Normalize(yy3.min(), yy3.max())

ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(xx1,xx2,yy3, c=yy3, cmap='cividis', marker='o', norm=norm)
#scatter = ax.scatter(xx1,xx2,yy3, c=yy3, cmap='YlGnBu_r', marker='o', norm=norm)
#plasma
# Set labels
ax.set_xlabel('Temperature',size=12,labelpad=15, fontweight='bold')
ax.set_ylabel('time',size=15,labelpad=12, fontweight='bold')
ax.set_zlabel(f'{yyy}',size=15,labelpad=12, fontweight='bold')
ax.set_title(f'{yyy}  for {add} Redox Agent',size=25, fontweight='bold')
# Add a colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label(f'{yyy}')

# Show the plot
plt.show()


add='1x'
if add=='1x':
    xx1=input_space[18281:36562,1]
    xx2=input_space[18281:36562,2]
    yy3=l_yp[18281:36562]
    
fig = plt.figure(figsize=(12,10))
norm = plt.Normalize(yy3.min(), yy3.max())

ax = fig.add_subplot(111, projection='3d')


scatter = ax.scatter(xx1,xx2,yy3, c=yy3, cmap='cividis', marker='o', norm=norm)
#scatter = ax.scatter(xx1,xx2,yy3, c=yy3, cmap='YlGnBu_r', marker='o', norm=norm)
#plasma

# Set labels
ax.set_xlabel('Temperature',size=12,labelpad=15, fontweight='bold')
ax.set_ylabel('time',size=15,labelpad=12, fontweight='bold')
ax.set_zlabel(f'{yyy}',size=15,labelpad=12, fontweight='bold')
ax.set_title(f'{yyy}  for {add} Redox Agent',size=25, fontweight='bold')
# Add a colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label(f'{yyy}')

# Show the plot
plt.show()


add='2x'
if add=='2x':
    xx1=input_space[36562:54843,1]
    xx2=input_space[36562:54843,2]
    yy3=l_yp[36562:54843]




fig = plt.figure(figsize=(12,10))
norm = plt.Normalize(yy3.min(), yy3.max())

ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(xx1,xx2,yy3, c=yy3, cmap='cividis', marker='o', norm=norm)
#scatter = ax.scatter(xx1,xx2,yy3, c=yy3, cmap='YlGnBu_r', marker='o', norm=norm)
#plasma
# Set labels
ax.set_xlabel('Temperature',size=12,labelpad=15, fontweight='bold')
ax.set_ylabel('time',size=15,labelpad=12, fontweight='bold')
ax.set_zlabel(f'{yyy}',size=15,labelpad=12, fontweight='bold')
ax.set_title(f'{yyy}  for {add} Redox Agent',size=25, fontweight='bold')
# Add a colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label(f'{yyy}')

# Show the plot
plt.show()

