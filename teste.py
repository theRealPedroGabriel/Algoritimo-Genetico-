import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

def read_dat_file(file_path):
    # Use pandas para ler o arquivo
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    # Supondo que o arquivo tem duas colunas
    data.columns = ['X', 'Y']
    return data
# Caminho para o arquivo .dat
file_path = r'C:\Users\pgsmc\Documents\aerogerador.dat'

# Ler os dados
data = read_dat_file(file_path)

# Extrair as colunas X e Y
DataX = data['X']
DataY = data['Y']


y = DataY.values

x1 = DataX[ 'MolWeight' ].values

x1.shape = (len(x1),1)

x2 = DataX[ 'NunCarbon' ].values

x2.shape = (len(x2),1)

X = np.concatenate(( x1,x2),axis=1)
X = np.concatenate( (np.ones ((X.shape[0],1)),X) ,axis=1)

B = np. linalg. pinv(X.T@X)@X. T@y

x_lin = np.linspace(0, 600,200)

y_lim= np.linspace(0,30,200)

xx,yy = np.meshgrid(x_lin,y_lim)

zz = B[0] + B[1]*xx + B[2]*yy

fig = plt.figure()

ax = fig.add_subplot (projection="3d")

ax. scatter(x1,x2,y, color="#004040")

ax.set_xlabel ("MolWeight")
ax.set_ylabel("NunCarbon")
ax.plot_surface(xx,yy,zz,cnap="viridis",rstride=10,cstride=10)
plt.show()