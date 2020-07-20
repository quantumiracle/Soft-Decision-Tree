import numpy as np
import matplotlib.pyplot as plt

file_name='cdt_ppo_discrete_freewaydepth_33_id0'

def smooth(y, radius=200, mode='two_sided'):
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        return np.convolve(y, convkernel, mode='same') / \
               np.convolve(np.ones_like(y), convkernel, mode='same')
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / \
              np.convolve(np.ones_like(y), convkernel, mode='full')
        return out[:-radius+1]

r = np.load(file_name+'.npy')
x=np.arange(r.shape[0])

plt.plot(x,smooth(r))
plt.plot(x,r, alpha=0.7)
plt.savefig(file_name+'.png')
plt.show()