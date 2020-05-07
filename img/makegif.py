import imageio
import os
import argparse

parser = argparse.ArgumentParser(description='path')
parser.add_argument('--path', dest='path')
args = parser.parse_args()

images = []
directory = args.path+'/'
for filename in os.listdir(directory):
    images.append(imageio.imread(directory+filename))
imageio.mimsave(args.path+'.gif', images)