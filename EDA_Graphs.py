'''
In The Name of GOD
Author : Ali Pilehvar Meibody


Graphs
'''
#===========================================================
#===========================================================
'Importing Libs'
#===========================================================
#===========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
sns.heatmap(correlation, cmap="coolwarm", annot=True, cbar=True, cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8})
plt.tick_params(labelsize=16, pad=12)

name='correlation.jpg'
plt.show()
plt.savefig(name,dpi=600,format='jpg')



#---------Pearson correlation--------
correlation_matrix = data.corr()

# Plot heatmap for Pearson correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1 ,cbar=True, cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8})
plt.title('Pearson Correlation Heatmap')
plt.show()





#===========================================================
#===========================================================
'PAIR PLOT'
#===========================================================
#===========================================================

sns.pairplot(data)
plt.show()


