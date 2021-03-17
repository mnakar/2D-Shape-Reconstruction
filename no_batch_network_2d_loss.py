import torch
import torch.nn as nn
import torch.nn.functional as F
import complex_shapes

import utils

from criterion import Dist_Loss, EV_Reconstruction_Loss, Explode_Loss
from dataset import train_loader, test_loader, num_of_shapes, num_of_parts, num_of_cells, test_size
import numpy as np
import matplotlib.pyplot as plt


class ReconstructNet(nn.Module):
    """
    2D with Depth From 2D Reconstruction Network
    """

    def __init__(self):
        super(ReconstructNet, self).__init__()

        self.image_localization = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            # nn.Conv2d(6, 6, kernel_size=3),
            # nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        self.parts_localization = nn.Sequential(
            nn.Conv2d(num_of_parts, 6, kernel_size=5),
            # nn.Conv2d(6, 6, kernel_size=3),
            # nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        self.combine_localization = nn.Sequential(
            nn.Conv2d(12, 15, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(15, 20, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 22, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 1 * 2 translation vector:
        self.fc_loc = nn.Sequential(
            nn.Linear(22 * 4 * 4, 40),
            nn.ReLU(True),
            nn.Linear(40, num_of_parts * 2),
            #nn.Tanh()
        )

        self.init_weights()

    def init_weights(self):
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor([[1,0,0],[0,1,0],[1,0,0],[0,1,0]]).flatten())
        self.fc_loc[2].bias.data.copy_((torch.zeros(num_of_parts * 2)))

    # Spatial transformer network
    def stn(self, ev_image, parts):
        parts = parts.unsqueeze(0)
        ev_image = ev_image.unsqueeze(0).unsqueeze(0)

        # combine_data = torch.cat([ev_image, parts], dim=1)
        # xs = self.combine_localization(combine_data)
        # xs = xs.view(1, 15 * 3 * 3)
        # out = self.fc_loc(xs).view(num_of_parts, 1,2,3)
        # out = self.fc_loc(xs).view(num_of_parts, 2)

        xs1 = self.image_localization(ev_image)
        xs2 = self.parts_localization(parts)

        combine_data = torch.cat([xs1, xs2], dim=1)
        xs = self.combine_localization(combine_data)  # 20x11x11
        xs = xs.view(1, 22 * 4 * 4)
        out = self.fc_loc(xs).view(num_of_parts, 2)

        print("*********** OUT: ", out)

        # trans_vecs = torch.zeros(num_of_parts, 2)
        trans_vecs = 0  # out.clone().detach()

        # out = torch.cat([out[:, 1].view(-1,1), out[:, 0].view(-1,1)], dim=1) #Flip (y,x) to (x,y) for affine grid
        voxel_ev = torch.zeros(parts.shape).to(device)

        for i in range(num_of_parts):
            # part_vec = ((-1)*out[i]).unsqueeze(1)
            part_vec = (out[i]).unsqueeze(1)

            # part_vec = torch.tensor([[-10.0],[0.0]])
            # Build theta from part translation vector:
            # theta = out[i]
            theta = torch.cat([(torch.eye(2)).to(device), part_vec], 1)
            a = torch.tensor([
                [2.0/num_of_cells, 0.0,              -1.0],
                [0.0,              2.0/num_of_cells, -1.0],
                [0.0,              0.0,               1.0]
            ]).to(device)
            b = torch.cat([theta, torch.tensor([0.0, 0.0, 1.0]).to(device).unsqueeze(0)], 0)
            theta = (torch.matmul(a, (torch.matmul(b, torch.inverse(a)))))[0:2, :].unsqueeze(0)
            # theta = (torch.matmul(a, (torch.matmul(b, torch.inverse(a)))))[0:2, :].unsqueeze(0)

            # identify part in voxel_ev:
            part_voxel = parts[:, i].unsqueeze(0)

            grid = F.affine_grid(theta, part_voxel.size())

            # grid = xs.permute(0,2,3,1)# Because grid only allows size of NxDxHxWx3
            # print(grid.shape)

            voxel_ev[:,i] = F.grid_sample(part_voxel, grid)

        return voxel_ev, trans_vecs

    # image of shape num_of_cells X num_of_cells
    # parts of shape num_of_parts X num_of_cells X num_of_cells X num_of_cells
    def forward(self, ev_image, parts):
        voxel_ev, trans_vecs = self.stn(ev_image, parts)
        return voxel_ev, trans_vecs


########################################################################################################################\
#                                                   Create Model                                                       #\
########################################################################################################################\

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ReconstructNet()

model = model.to(device)

criterion = Explode_Loss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)

########################################################################################################################\
#                                                    Train Model                                                       #\
########################################################################################################################\


num_epochs = 300

train_loss_values = []
loss1_values = []
loss2_values = []
loss3_values = []
loss4_values = []
loss5_values = []
loss6_values = []

model.train()



'''
for i in range(input.shape[1]):
    for j in range(input.shape[2]):
        if input[:,i,j] == 0:
            input[:,i, j] = -1

for i in range(parts.shape[1]):
    for j in range(parts.shape[2]):
        if parts[:,i,j] == 0:
            parts[:,i, j] = -1
'''
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
        input = data.squeeze(0)
        input = input.to(device)

        # Render only last epoch's results
        render = False
        if t == num_epochs - 1:
            render = True


        parts = utils.center_shape_parts(input)

        # union = input[0] + input[1] > 1
        # union = input[0] + input[1] + input[2] + input[3] + input[4] > 1
        # union = input[0] + input[1] + input[2] + input[3] + input[4] + input[5] + input[6] + input[7] + input[8] > 1
        # if len(union.nonzero()) == 0:

        # For multiple parts:
        #ev_image = input[0] + input[1]*(2) #+ input[2]*3 + input[3]*4 + input[4]*5 + input[5]*6 + input[6]*7 + input[7]*8 + input[8]*9
        ev_image = input[0] + input[1] * 2 + input[2] * 3 + input[3] * 4 + input[4] * 5
        # ev_image = input[0] + input[1]*2
        # ev_image = input.squeeze()

        # utils.render.render_shape(parts, "parts.png")

        voxel_ev, trans_vecs = model(ev_image, parts)
        voxel_ev = voxel_ev.squeeze(0)
        input_centers = utils.calc_shape_center(input)
        rec_centers = utils.calc_shape_center(voxel_ev)

        rec_image = voxel_ev[0] + voxel_ev[1] * 2 + voxel_ev[2] * 3 + voxel_ev[3] * 4 + voxel_ev[4] * 5

        if render and (i==0 or i==1 or i==2):

            # Turn the depth voxel into translation vectors
            input_centers = utils.calc_shape_center(input)
            shape_center = torch.tensor([num_of_cells / 2, num_of_cells / 2]).to(device)
            '''
            parts_vecs = torch.zeros(num_of_parts, 2).to(device)
            for j in range(num_of_parts):
                part_vec = input_centers[j] - shape_center
                parts_vecs[j] = part_vec

                # print("recons shape center:", utils.calc_shape_center((voxel_ev)))
                # print("org  shape center:", utils.calc_shape_center((input)))
                # print("input vector is: ", part_vec)
                # print("reconstructed vector is: ", trans_vecs[j])
            '''
            utils.render.render_shapes(input, voxel_ev, True)

            print(torch.unique(input.detach(), return_counts=True))
            print(torch.unique(voxel_ev.detach(), return_counts=True))
            #utils.save_as_rgb_org.save(ev_image)
            #utils.save_as_rgb_rec.save(rec_image)

            # with torch.no_grad():
            #    print("Vec loss is: ", nn.MSELoss()(parts_vecs, trans_vecs))

            '''
            if render:
                reconst_ev = voxel_ev# * (voxel_ev >= 0.5)
                utils.save_as_rgb.save(ev_image)
                utils.render.render_shapes(input, reconst_ev, True)
            '''

        # loss, loss1, loss2, loss3, loss4, loss5, loss6 = criterion(input, voxel_ev)

        loss, loss1, loss2, loss3, loss4, loss5, loss6 = criterion(input_centers, rec_centers, ev_image, rec_image)

        loss_history += loss.item()
        loss1_history += loss1
        loss2_history += loss2
        loss3_history += loss3
        loss4_history += loss4
        loss5_history += loss5
        loss6_history += loss6

        print("Epoch ", t, "Loss:", loss)

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
np.savez('loss_history2.npz', train_loss=train_loss_values, loss1=loss1_values, loss2=loss2_values, loss3=loss3_values,
         loss4=loss4_values, loss5=loss5_values, loss6=loss6_values)
# utils.print_graphs.print_loss6(num_epochs)


data = np.load('loss_history2.npz', allow_pickle=True)
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
# plt.ylim(0, 20)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('Train loss vs Train loss apart ')
plt.grid(True)
plt.legend(
    ['Train loss', 'Shape Reconstruction loss', 'Distance loss', 'Arik loss', 'Noa loss', '3D EV Reconstruction loss',
     'Frame Loss'])
# plt.legend(['Train loss', 'Reconstruction loss', 'Distance loss'])
plt.style.use(['classic'])
plt.savefig('Results/loss_graph2.png')
plt.clf()

########################################################################################################################\
#                                                     Test Model                                                       #\
########################################################################################################################\


print(" ")
print("################################################################################")
print(" ")

loss_history = 0.0
actual_test_size = 0

model.eval()
for i, data in enumerate(test_loader, 0):
    X_train = data.squeeze(0)

    # Create 3D Exploded View from the original shape
    input = X_train
    input = input.to(device)

    # union = input[0] + input[1]  > 1
    # union = input[0] + input[1] + input[2] + input[3] + input[4] > 1

    actual_test_size += 1
    # ev_image = input.squeeze()
    # ev_image = input[0] + input[1] * 2 + input[2] * 3 + input[3] * 4 + input[4] * 5
    ev_image = input[0] + input[1] * 2 + input[2] * 3 + input[3] * 4 + input[4] * 5 #+ input[5] * 6 + input[6] * 7 + \
               #input[7] * 8 + input[8] * 9

    # ev_image = input[0] + input[1] * 2

    # Reconstruct 3D Exploded View from 2D Exploded View
    parts = utils.center_shape_parts(input)

    voxel_ev, trans_vecs = model(ev_image, parts)
    voxel_ev = voxel_ev.squeeze(0)
    # Turn the depth voxel into translation vectors
    input_centers = utils.calc_shape_center(input)
    shape_center = torch.tensor([num_of_cells / 2, num_of_cells / 2]).to(device)
    #rec_image = voxel_ev[0] + voxel_ev[1]*2
    rec_image = voxel_ev[0] + voxel_ev[1] * 2 + voxel_ev[2] * 3 + voxel_ev[3] * 4 + voxel_ev[4] * 5

    '''
    parts_vecs = torch.zeros(num_of_parts, 2).to(device)
    for j in range(num_of_parts):
        part_vec = input_centers[j] - shape_center
        parts_vecs[j] = part_vec

        # print("recons shape center:", utils.calc_shape_center((voxel_ev)))
        # print("org  shape center:", utils.calc_shape_center((input)))
        print("input vector is: ", part_vec)
        print("reconstructed vector is: ", trans_vecs[j])
    '''
    #rec_image = voxel_ev[0]
    utils.render.render_shapes(input, voxel_ev, False)

    #utils.save_as_rgb_org.save(ev_image)
    #utils.save_as_rgb_rec.save(rec_image)

    #with torch.no_grad():
    #    print("Vec loss is: ", nn.MSELoss()(parts_vecs, trans_vecs))

    input_centers = utils.calc_shape_center(input)
    rec_centers = utils.calc_shape_center(voxel_ev)

    loss, loss1, loss2, loss3, loss4, loss5, loss6 = criterion(input_centers, rec_centers, ev_image, rec_image)

    print("Test shape #", i, "Loss: ", loss.item())

    loss_history += loss.item()

print("################################################################################")
# print("Test avarage loss is: ", loss_history/num_of_shapes)
print("actual test size: ", actual_test_size)
print("Test avarage loss is: ", loss_history / actual_test_size)
