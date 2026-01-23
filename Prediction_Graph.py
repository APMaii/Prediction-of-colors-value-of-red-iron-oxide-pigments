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
yy=L
yyy='L*'
mm='SVR'


model = SVR()
pipe_model = PPL([("scaler", MinMaxScaler()), ("model", model)])
regressor = TTR(regressor=pipe_model, transformer=StandardScaler())
fold1=KFold(n_splits=7,shuffle=True,random_state=42)

myparams={'regressor__model__C': [100],
 'regressor__model__epsilon': [0.3],
 'regressor__model__gamma': ['scale'],
 'regressor__model__kernel': ['rbf'],
 'regressor__scaler': [StandardScaler()],
 'transformer': [StandardScaler()]}


gs = GridSearchCV(regressor, param_grid=myparams, cv=fold1, scoring='neg_mean_absolute_percentage_error',n_jobs=10)
gs.fit(X,yy)

l_yp=gs.predict(input_space)
scoree=gs.best_score_


add='No'
if add=='No':
    xx1=input_space[0:18281,1].flatten()
    xx2=input_space[0:18281,2].flatten()
    yy3=l_yp[0:18281].flatten()

fig = plt.figure(figsize=(12,10), dpi=600)
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

# Save and show the plot
plt.savefig(f'{yyy}_{add}_RedoxAgent.png', dpi=600, bbox_inches='tight')
plt.show()



add='1x'
if add=='1x':
    xx1=input_space[18281:36562,1].flatten()
    xx2=input_space[18281:36562,2].flatten()
    yy3=l_yp[18281:36562].flatten()
    
fig = plt.figure(figsize=(12,10), dpi=600)
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

# Save and show the plot
plt.savefig(f'{yyy}_{add}_RedoxAgent.png', dpi=600, bbox_inches='tight')
plt.show()


add='2x'
if add=='2x':
    xx1=input_space[36562:54843,1].flatten()
    xx2=input_space[36562:54843,2].flatten()
    yy3=l_yp[36562:54843].flatten()




fig = plt.figure(figsize=(12,10), dpi=600)
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

# Save and show the plot
plt.savefig(f'{yyy}_{add}_RedoxAgent.png', dpi=600, bbox_inches='tight')
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
    xx1=input_space[0:18281,1].flatten()
    xx2=input_space[0:18281,2].flatten()
    yy3=l_yp[0:18281].flatten()

fig = plt.figure(figsize=(12,10), dpi=600)
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

# Save and show the plot
plt.savefig(f'{yyy}_{add}_RedoxAgent.png', dpi=600, bbox_inches='tight')
plt.show()




add='1x'
if add=='1x':
    xx1=input_space[18281:36562,1].flatten()
    xx2=input_space[18281:36562,2].flatten()
    yy3=l_yp[18281:36562].flatten()
    
fig = plt.figure(figsize=(12,10), dpi=600)
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

# Save and show the plot
plt.savefig(f'{yyy}_{add}_RedoxAgent.png', dpi=600, bbox_inches='tight')
plt.show()



add='2x'
if add=='2x':
    xx1=input_space[36562:54843,1].flatten()
    xx2=input_space[36562:54843,2].flatten()
    yy3=l_yp[36562:54843].flatten()




fig = plt.figure(figsize=(12,10), dpi=600)
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

# Save and show the plot
plt.savefig(f'{yyy}_{add}_RedoxAgent.png', dpi=600, bbox_inches='tight')
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
    xx1=input_space[0:18281,1].flatten()
    xx2=input_space[0:18281,2].flatten()
    yy3=l_yp[0:18281].flatten()

fig = plt.figure(figsize=(12,10), dpi=600)
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

plt.savefig(f'{yyy}_{add}_RedoxAgent.png', dpi=600, bbox_inches='tight')
# Show the plot
plt.show()


add='1x'
if add=='1x':
    xx1=input_space[18281:36562,1].flatten()
    xx2=input_space[18281:36562,2].flatten()
    yy3=l_yp[18281:36562].flatten()
    
fig = plt.figure(figsize=(12,10), dpi=600)
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

# Save and show the plot
plt.savefig(f'{yyy}_{add}_RedoxAgent.png', dpi=600, bbox_inches='tight')
plt.show()


add='2x'
if add=='2x':
    xx1=input_space[36562:54843,1].flatten()
    xx2=input_space[36562:54843,2].flatten()
    yy3=l_yp[36562:54843].flatten()




fig = plt.figure(figsize=(12,10), dpi=600)
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

# Save and show the plot
plt.savefig(f'{yyy}_{add}_RedoxAgent.png', dpi=600, bbox_inches='tight')
plt.show()



