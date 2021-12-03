import numpy as np
from data_impo import *
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

n=len(exportX_train())
x_train=exportX_train()
x_test=exportX_test()
y_train=exportY_train()
y_test=exportY_test()

alpha=0.0001

a_0=np.zeros((n,1))
a_1=np.zeros((n,1))

epochs=0


while(epochs<1000):
    y = a_0 + a_1 * x_train
    error = y - y_train
    mean_sq_er = np.sum(error**2)
    mean_sq_er = float(mean_sq_er/n)
    a_0 = a_0 - alpha * 2 * np.sum(error)/n 
    a_1 = a_1 - alpha * 2 * np.sum(error * x_train)/n
    epochs += 1


y_prediction = a_0[0:300] + a_1[0:300] * x_test

print('R2 Score:',metrics.r2_score(y_test,y_prediction))
"""
y_plot = []
for i in range(100):
    y_plot.append(a_0 + a_1 * i)
plt.figure(figsize=(10,10))
plt.scatter(x_test,y_test,color='red',label='GT')
plt.plot(range(len(y_plot)),y_plot,color='black',label = 'pred')
plt.legend()
plt.show()"""