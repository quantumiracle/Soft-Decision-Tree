import numpy as np
import matplotlib.pyplot as plt

file_name='cdt_ppo_discrete_CartPole-v1_id5'

r = np.load(file_name+'.npy')
x=np.arange(r.shape[0])

plt.plot(x,r)
plt.savefig(file_name+'.png')
plt.show()