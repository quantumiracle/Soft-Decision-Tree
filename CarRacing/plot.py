import numpy as np
import matplotlib.pyplot as plt

# file_name='learn_Freeway-v0'
file_name='learn_Enduro-v0'

r = np.load(file_name+'.npy')
x=np.arange(r.shape[0])

plt.plot(x,r)
plt.savefig('img/'+file_name+'.png')
plt.show()