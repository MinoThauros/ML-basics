import pandas as pd
import numpy as np

df_train=pd.read_csv('./files/train.csv')
df_test=pd.read_csv('./files/test.csv')

x_train=df_train['x']
y_train=df_train['y']
x_test=df_test['x']
y_test=df_test['y']

x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)

x_train=x_train.reshape(-1,1)#-1 means the number of rows is not specified
x_test=x_test.reshape(-1,1)
y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

def exportX_train():
    return x_train

def exportX_test():
    return x_test

def exportY_train():
    return y_train

def exportY_test():
    return y_test