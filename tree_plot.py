# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from SDT import SDT
from torch.utils import data
from utils.dataset import Dataset
import numpy as np

def get_binary_index(tree):
    """
    Get binary index for tree nodes:
    From

    0
    1 2
    3 4 5 6 

    to 

    '0'
    '00' '01' 
    '000' '001' '010' '011'

    """
    index_list = []
    for layer_idx in range(0, tree.max_depth+1):
        index_list.append([bin(i)[2:].zfill(layer_idx+1) for i in range(0, np.power(2, layer_idx))])
    return np.concatenate(index_list)

def path_from_prediction(tree, idx):
    """
    Generate list of nodes as decision path, 
    with each node represented by a binary string and an int index
    """
    binary_idx_list = []
    int_idx_list=[]
    idx = int(idx)
    for layer_idx in range(tree.max_depth+1, 0, -1):
        binary_idx_list.append(bin(idx)[2:].zfill(layer_idx))
        int_idx_list.append(2**(layer_idx-1)-1+idx)
        idx = int(idx/2)
    binary_idx_list.reverse()  # from top to bottom
    int_idx_list.reverse() 
    return binary_idx_list, int_idx_list

def draw_tree(tree, input_shape, input_img=None, show_correlation=False, savepath=''):

    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import ConnectionPatch

    def _add_arrow(ax_parent, ax_child, xyA, xyB, color='black'):
        '''Private utility function for drawing arrows between two axes.'''
        con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data',
                              axesA=ax_child, axesB=ax_parent, arrowstyle='<|-',
                              color=color, linewidth=tree.args['depth'])
        ax_child.add_artist(con)

    inner_nodes = tree.state_dict()['linear.weight']
    leaf_nodes = tree.state_dict()['param']
    binary_indices = get_binary_index(tree)
    inner_indices = binary_indices[:tree.inner_node_num]
    leaf_indices = binary_indices[tree.inner_node_num:]
    
    if len(input_shape) == 3:
        img_rows, img_cols, img_chans = input_shape
    elif len(input_shape) == 1:
        img_rows, img_cols = input_shape[0], input_shape[0]

    kernels = dict([(node_idx, node_value.cpu().numpy().reshape(input_shape)) for node_idx, node_value in zip (inner_indices, inner_nodes[:, 1:]) ])
    biases = dict([(node_idx, node_value.cpu().numpy().squeeze()) for node_idx, node_value in zip (inner_indices, inner_nodes[:, :1]) ])
    leaves = dict([(leaf_idx, np.array([leaf_dist.cpu().numpy()])) for leaf_idx, leaf_dist in zip (leaf_indices, leaf_nodes) ])

    n_leaves = tree.leaf_num
    assert len(leaves) == n_leaves

    # prepare figure and specify grid for subplots
    fig = plt.figure(figsize=(n_leaves, n_leaves//2), facecolor=(0.5, 0.5, 0.8))
    gs = GridSpec(tree.max_depth+1, n_leaves*2,
                  height_ratios=[1]*tree.max_depth+[0.5])

    # Grid Coordinate X (horizontal)
    gcx = [list(np.arange(1, 2**(i+1), 2) * (2**(tree.max_depth+1) // 2**(i+1)))
           for i in range(tree.max_depth+1)]
    gcx = list(itertools.chain.from_iterable(gcx))
    axes = {}
    path = ['0']

    imshow_args = {'origin': 'upper', 'interpolation': 'None', 'cmap': 'gray'}

    ''' some statistics '''
    # 1. kernel values
    # print(list(kernels.values()))

    # 2. sum of absolute weights
    # print(np.sum(np.abs(list(kernels.values())), axis=0))

    # 3. sum of weighted absolute weights
    # abs_kernels = np.abs(list(kernels.values()))
    # current_idx=0
    # weighted_kernels=[]
    # for i in range(tree.max_depth):
    #     nodes_num = 2**i
    #     weight = 1/nodes_num
    #     weighted_kernels_per_layer = np.mean(abs_kernels[current_idx:current_idx+nodes_num], axis=0)
    #     weighted_kernels.append(weighted_kernels_per_layer)
    #     current_idx = current_idx+nodes_num
    # print('weighted absolute kernels: ', np.mean(weighted_kernels, axis=0))
        
    # draw tree nodes
    for pos, key in enumerate(sorted(kernels.keys(), key=lambda x:(len(x), x))):
        ax = plt.subplot(gs[len(key)-1, gcx[pos]-2:gcx[pos]+2])
        axes[key] = ax
        kernel_image = np.abs(kernels[key])  # absolute value
        kernel_image = kernel_image/np.sum(kernel_image)  # normalization
        # if input_img is not None and key in path:
        #     # here the path is not correct, choose right whenever the logit > 0 does not ensure 
        #     # the final path is the path with maximiaed probability.

        #     logit = tree.beta[int(key, 2)].cpu().detach().numpy() * (
        #         np.sum( kernels[key]*input_img) + biases[key])
        #     path.append(key + ('1' if (logit) >= 0 else '0'))
        #     ax.text(img_cols//2, img_rows+2, '{:.2f}'.format(logit),
        #             ha='center', va='center')

        #     if show_correlation:
        #         kernel_image = input_img * kernels[key]
        if len(kernel_image.shape)==3: # 2D image (H, W, C)
            ax.imshow(kernel_image.squeeze(), **imshow_args)
        elif len(kernel_image.shape)==1:
            vector_image = np.ones((kernel_image.shape[0], 1)) @ [kernel_image]
            ax.imshow(vector_image, **imshow_args)
        ax.axis('off')
        digits = set([np.argmax(leaves[k]) for k in leaves.keys()
                      if k.startswith(key)])
        title = ','.join(str(digit) for digit in digits)
        plt.title('{}'.format(title))
            
    # change the way to get path to be via the prediction by the tree
    if input_img is not None:
        tree.forward(torch.Tensor(input_img).unsqueeze(0))
        max_leaf_idx = tree.max_leaf_idx
        path, path_idx_int = path_from_prediction(tree, max_leaf_idx)

    # draw tree leaves
    for pos, key in enumerate(sorted(leaves.keys(), key=lambda x:(len(x), x))):
        ax = plt.subplot(gs[len(key)-1,
                            gcx[len(kernels)+pos]-1:gcx[len(kernels)+pos]+1])
        axes[key] = ax
        leaf_image = np.ones((tree.args['output_dim'], 1)) @ leaves[key]
        ax.imshow(leaf_image, **imshow_args)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('{}'.format(np.argmax(leaves[key])), y=-.5)

    # add arrows indicating flow
    for pos, key in enumerate(sorted(axes.keys(), key=lambda x:(len(x), x))):
        children_keys = [k for k in axes.keys()
                         if len(k) == len(key) + 1 and k.startswith(key)]
        for child_key in children_keys:
            p_rows, p_cols = axes[key].get_images()[0].get_array().shape
            c_rows, c_cols = axes[child_key].get_images()[0].get_array().shape
            color = 'green' if (key in path and child_key in path) else 'red'
            _add_arrow(axes[key], axes[child_key],
                       (c_cols//2, 1), (p_cols//2, p_rows-1), color)


    # draw input image with arrow indicating flow into the root node
    if input_img is not None:
        ax = plt.subplot(gs[0, 0:4])
        if len(input_img.shape)==3: # 2D image (H, W, C)
            ax.imshow(input_img.squeeze(), clim=(0.0, 1.0), **imshow_args)
        elif len(input_img.shape)==1:
            vector_image = np.ones((input_img.shape[0], 1)) @ [input_img]
            ax.imshow(vector_image, **imshow_args)
        ax.axis('off')
        plt.title('input')
        _add_arrow(ax, axes['0'],
                   (1, img_rows//2), (img_cols-1, img_rows//2), 'green')

    if savepath:
        plt.savefig(savepath, facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()

    return path_idx_int

if __name__ == '__main__':
    import tensorflow as tf
    from main import learner_args
    from sdt_train import learner_args


    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    input_img = x_test[90]

    tree = SDT(learner_args)
    tree.load_model(learner_args['model_path'])

    # draw_tree(tree, (28, 28, 1), input_img=input_img)
    draw_tree(tree, (tree.args['input_dim'],))
