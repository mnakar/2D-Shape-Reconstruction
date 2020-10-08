import torch
import math
#from simple_shapes import num_of_shapes, num_of_parts, num_of_vertices
from dataset import num_of_shapes, num_of_parts, num_of_cells

import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


########################################################################################################################
#                                               Render Shapes Class                                                    #
########################################################################################################################


class Render:
    def __init__(self):
        self.counter = 0

    def render_shapes(self, org_shape, recons_ev, print_train):
        """
        :param org_shape: Tensor representing the original shape with [parts][vertices]
        :param ev_shape: Tensor representing the exploded shape with [parts][vertices]
        :param pred_shape: Tensor representing the predicted shape with [parts][vertices]
        :param print_train: Bool representing whether the rendered data is of training or of test phase
        :return: None
        A screenshot of all three shapes is saved in Results folder
        """
        if print_train:
            name1 = 'Results/{}_org_shape.png'.format(self.counter)
            name5 = 'Results/{}_reconstructed_EV.png'.format(self.counter)
        else:
            name1 = 'Results/test_{}_org_shape.png'.format(self.counter)
            name5 = 'Results/test_{}_reconstructed_EV.png'.format(self.counter)

        self.counter += 1

        # Render original shape:
        #org_shape = org_shape.unsqueeze(1)

        self.render_shape(org_shape, name1)
        self.render_shape(recons_ev, name5)


    def render_shape(self, shape, name):
        """
        :param shape: tensor representing shape with [parts][vertices]
        :param name: The name to call the screenshot file
        :return: None
        A screenshot of the shape is saved in path Results/name
        """

        Colors = { 1: 'red',
                   2: 'blue',
                   3: 'green',
                   4: 'cyan',
                   5: 'magenta',
                   6: 'yellow',
                   7: 'black',
                   8: 'white',
                   9: 'orange'
                  }

        shape_dt = shape.detach()
        np_shape = shape_dt.cpu().numpy()
        parts_dict = {}

        # combine the objects into a single boolean array
        voxels = np.zeros((num_of_cells, num_of_cells, num_of_cells))
        voxels = voxels.astype(bool)

        # set the colors of each object
        colors = np.empty(voxels.shape, dtype=object)
        for i in range(shape.shape[0], 0, -1):
            np_part = np_shape[i-1].transpose(1, 0)

            np_part = np_part >= 0.5

            voxels[:,:,0] = voxels[:,:,0] | np_part
            parts_dict[i] = np_part
            colors[parts_dict[i]] = Colors[i]

        # plot everything
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolor='k')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.savefig(name, dpi=400)
        #plt.show()
        plt.close(fig)

render = Render()


class SaveRGB():
    def __init__(self):
        self.counter = 0

    def save(self, shape_pixel):
        Colors = {0: [255, 255, 255],  # White
                  1: [255, 0, 0],  # Red
                  2: [0, 0, 255],  # Blue
                  3: [0, 255, 0],  # Green
                  4: [0, 255, 255],  # Cyan
                  5: [255, 0, 255],  # Magenta
                  6: [255, 255, 0],  # Yellow
                  7: [0, 0, 0],  # Black
                  8: [128, 128, 128],  # Gray
                  9: [255, 165, 0]  # Orange
                  }

        rgb_image = np.zeros((num_of_cells, num_of_cells, 3))
        for i in range(shape_pixel.shape[0]):
            for j in range(shape_pixel.shape[1]):
                rgb_image[i, j] = np.array(Colors[int(torch.round(shape_pixel[i, j]))]) / 255.

        scale = 20
        scaled_image = np.zeros((num_of_cells * scale, num_of_cells * scale, 3))

        for i in range(rgb_image.shape[0]):
            for j in range(rgb_image.shape[1]):
                scaled_image[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale] = rgb_image[i, j]

        name = 'Results/{}_2D_ev.png'.format(self.counter)
        plt.imsave(name, scaled_image)
        self.counter = (self.counter+1)%num_of_shapes

save_as_rgb = SaveRGB()
########################################################################################################################
#                                                Print Graphs Class                                                    #
########################################################################################################################

class PrintGraphs:
    #def __init__(self, ):
    def print_loss6(self, num_epochs):
        """
        :param num_epochs: Integer representing number of Training epochs
        :return: None
        A loss graph is saved in path Results/loss_graph.png
        """
        data = np.load('loss_history.npz', allow_pickle=True)
        train_loss = data['train_loss']
        loss1 = data['loss1']
        loss2 = data['loss2']
        loss3 = data['loss3']
        loss4 = data['loss4']
        loss5 = data['loss5']
        loss6 = data['loss6']

        plt.figure(1, figsize=(7, 5))
        plt.plot(range(1, num_epochs + 1), train_loss)
        plt.plot(range(1, num_epochs + 1), loss1)
        plt.plot(range(1, num_epochs + 1), loss2)
        plt.plot(range(1, num_epochs + 1), loss3)
        plt.plot(range(1, num_epochs + 1), loss4)
        plt.plot(range(1, num_epochs + 1), loss5)
        plt.plot(range(1, num_epochs + 1), loss6)
        #plt.ylim(0, 0.5)
        plt.xlabel('num of Epochs')
        plt.ylabel('loss')
        plt.title('Train loss vs Train loss apart ')
        plt.grid(True)
        plt.legend(['Train loss', 'Shape Reconstruction loss', 'Distance loss', 'Arik loss', 'Noa loss', '3D EV Reconstruction loss', 'Frame Loss'])
        #plt.legend(['Train loss', 'Reconstruction loss', 'Distance loss'])
        plt.style.use(['classic'])
        plt.savefig('Results/loss_graph.png')
        plt.clf()

    def print_loss(self, num_epochs):
        """
        :param num_epochs: Integer representing number of Training epochs
        :return: None
        A loss graph is saved in path Results/loss_graph.png
        """
        data = np.load('loss_history.npz', allow_pickle=True)
        train_loss = data['train_loss']

        plt.figure(1, figsize=(7, 5))
        plt.plot(range(1, num_epochs + 1), train_loss)

        #plt.ylim(0, 0.1)
        plt.xlabel('num of Epochs')
        plt.ylabel('loss')
        plt.title('Train loss')
        plt.grid(True)
        plt.legend(['Train loss'])
        #plt.legend(['Train loss', 'Reconstruction loss', 'Distance loss'])
        plt.style.use(['classic'])
        plt.savefig('Results/loss_graph_single_loss2.png')
        plt.clf()

print_graphs = PrintGraphs()

########################################################################################################################
#                                                       Others                                                         #
########################################################################################################################


def center_shape_parts(shape_voxel):
    """
    :shape_voxel: voxel of shape num_of_parts * num_of_coords * num_of_coords * num_of_coords
    :return: same shape with parts located around center
    """
    parts = torch.zeros(num_of_parts, num_of_cells, num_of_cells).to(device)

    for i in range(shape_voxel.shape[0]):
        part = shape_voxel[i]
        part_indices = part.nonzero()

        max_vals = torch.max(part_indices, dim=0)
        min_vals = torch.min(part_indices, dim=0)
        max_x = max_vals.values[1]
        max_y = max_vals.values[0]
        min_x = min_vals.values[1]
        min_y = min_vals.values[0]
        center = torch.round(torch.tensor([(max_y + min_y)/2.0, (max_x + min_x)/2.0]).to(device))

        part_indices = part_indices + torch.tensor([num_of_cells/2, num_of_cells/2]).to(device) - center
        part_indices = part_indices.long()
        parts[i][part_indices[:, 0], part_indices[:, 1]] = 1.0

    return parts
