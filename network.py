import torch
import torch.nn as nn
import torch.nn.functional as F
import complex_shapes

import utils

from criterion import Dist_Loss, EV_Reconstruction_Loss, Explode_Loss
from dataset import train_loader, test_loader, num_of_shapes, num_of_parts, num_of_cells
import numpy as np

class ReconstructNet(nn.Module):
    """
    2D with Depth From 2D Reconstruction Network
    """
    def __init__(self):
        super(ReconstructNet, self).__init__()

        self.image_localization = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
            )

        self.parts_localization = nn.Sequential(
                nn.Conv2d(num_of_parts, 6, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
            )

        self.combine_localization = nn.Sequential(
                nn.Conv2d(12, 15, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(15, 20, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
            )

        # Regressor for the 1 * 2 translation vector:
        self.fc_loc = nn.Sequential(
                nn.Linear(20 * 2 * 2, 40),
                nn.ReLU(True),
                nn.Linear(40, num_of_parts * 2)
        )

        self.init_weights()

    def init_weights(self):
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_((torch.zeros(num_of_parts * 2)))

    # Spatial transformer network
    def stn(self, ev_image, parts):

        voxel_ev = torch.zeros(parts.shape).to(device)

        parts = parts.unsqueeze(0)
        ev_image = ev_image.unsqueeze(0).unsqueeze(0)

        xs1 = self.image_localization(ev_image)
        xs2 = self.parts_localization(parts)
        combine_data = torch.cat([xs1, xs2], dim=1)

        xs = self.combine_localization(combine_data)

        xs = xs.view(1, 20 * 2 * 2)
        out = self.fc_loc(xs).view(-1, 2)
        out = torch.cat([out[:, 1].view(-1,1), out[:, 0].view(-1,1)], dim=1) #Flip (y,x) to (x,y) for affine grid

        trans_vecs = torch.zeros(num_of_parts, 2)

        print("Reconstructing 3D from 2D:", out)

        for i in range(num_of_parts):

            trans_vecs[i] = out[i].clone()
            part_vec = out[i].unsqueeze(1)

            # Build theta from part translation vector:
            theta = torch.cat([(torch.eye(2)).to(device), part_vec], 1)
            a = torch.tensor([
                [2.0/num_of_cells, 0.0,              -1.0],
                [0.0,              2.0/num_of_cells, -1.0],
                [0.0,              0.0,               1.0]
            ]).to(device)
            b = torch.cat([theta, torch.tensor([0.0, 0.0, 1.0]).to(device).unsqueeze(0)], 0)
            theta = (torch.matmul(a, (torch.matmul(b, torch.inverse(a)))))[0:2, :].unsqueeze(0)

            # identify part in voxel_ev:
            part_voxel = parts[:, i].unsqueeze(0)

            grid = F.affine_grid(theta, part_voxel.size())
            voxel_ev[i] = F.grid_sample(part_voxel, grid)

        return voxel_ev, trans_vecs

    # image of shape num_of_cells X num_of_cells
    # parts of shape num_of_parts X num_of_cells X num_of_cells X num_of_cells
    def forward(self, ev_image, parts):
        voxel_ev, trans_vecs = self.stn(ev_image, parts)
        return voxel_ev, trans_vecs


########################################################################################################################
#                                                   Create Model                                                       #
########################################################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ReconstructNet()

model = model.to(device)

criterion = Explode_Loss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)


########################################################################################################################
#                                                    Train Model                                                       #
########################################################################################################################


num_epochs = 1000

train_loss_values = []
loss1_values = []
loss2_values = []
loss3_values = []
loss4_values = []
loss5_values = []
loss6_values = []


model.train()


train_loss_values = []


for t in range(num_epochs):

    loss_history  = 0.0
    loss1_history = 0.0
    loss2_history = 0.0
    loss3_history = 0.0
    loss4_history = 0.0
    loss5_history = 0.0
    loss6_history = 0.0

    for i, data in enumerate(train_loader, 0):

        optimiser.zero_grad()
        X_train = data.squeeze(0)

        # Render only last epoch's results
        render = False
        if t == num_epochs-1:
            render = True


        input = X_train

        #For multiple parts: ev_image = input[0]+input[1]
        ev_image = input.squeeze()

        parts = utils.center_shape_parts(input)

        voxel_ev, trans_vecs = model(ev_image, parts)

        if render:
            reconst_ev = voxel_ev * (voxel_ev >= 0.5)
            utils.save_as_rgb.save(ev_image)
            utils.render.render_shapes(input, reconst_ev, True)


        loss, loss1, loss2, loss3, loss4, loss5, loss6  = criterion(X_train, voxel_ev)
        loss_val = float(loss)

        loss_history  += loss
        loss1_history += loss1
        loss2_history += loss2
        loss3_history += loss3
        loss4_history += loss4
        loss5_history += loss5
        loss6_history += loss6

        print("Epoch ", t, "Loss ", i, ":", loss)

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()


    # Store epoch average loss
    train_loss_values.append(loss_history/num_of_shapes)
    loss1_values.append(loss1_history/num_of_shapes)
    loss2_values.append(loss2_history/num_of_shapes)
    loss3_values.append(loss3_history/num_of_shapes)
    loss4_values.append(loss4_history/num_of_shapes)
    loss5_values.append(loss5_history/num_of_shapes)
    loss6_values.append(loss6_history/num_of_shapes)


# Save loss history to np file
np.savez('loss_history.npz', train_loss=train_loss_values, loss1=loss1_values, loss2=loss2_values, loss3=loss3_values,
         loss4=loss4_values, loss5=loss5_values, loss6 = loss6_values)
utils.print_graphs.print_loss6(num_epochs)


########################################################################################################################
#                                                     Test Model                                                       #
########################################################################################################################


print(" ")
print("################################################################################")
print(" ")

test_loss_values = []

model.eval()
for i, data in enumerate(test_loader, 0):
    X_train = data.squeeze(0)

    # Create 3D Exploded View from the original shape
    input = X_train

    ev_image = input.squeeze()

    # Reconstruct 3D Exploded View from 2D Exploded View
    parts = utils.center_shape_parts(input)

    voxel_ev, trans_vecs = model(ev_image, parts)

    reconst_ev = voxel_ev * (voxel_ev >= 0.5)
    utils.save_as_rgb.save(ev_image)
    utils.render.render_shapes(input, reconst_ev, False)

    loss, loss1, loss2, loss3, loss4, loss5, loss6 = criterion(X_train, voxel_ev)

    print("Test shape #", i, "Loss: ", loss.item())


