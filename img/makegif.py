import imageio
import os
images = []
directory = './eval/'
for filename in os.listdir(directory):
    images.append(imageio.imread(directory+filename))
imageio.mimsave('eval.gif', images)