
import torch
import torch.nn as nn
from dataset import num_of_parts, num_of_cells
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Dist_Loss(torch.nn.Module):
    def __init__(self):
        super(Dist_Loss, self).__init__()

    def forward(self, ev_pred):
        # Calculate Distance loss:
        '''
        frame_part = torch.zeros(num_of_cells, num_of_cells, num_of_cells).to(device)
        frame_part[0, :, :] = 1.0
        frame_part[num_of_cells - 1, :, :] = 1.0
        frame_part[:, 0, :] = 1.0
        frame_part[:, num_of_cells - 1, :] = 1.0
        frame_part[:, :, 0] = 1.0
        frame_part[:, :, num_of_cells - 1] = 1.0
        '''

        max_sum = torch.tensor(1.0).to(device) # max weight allowed

        dists = torch.zeros(0).to(device)
        for i in range(num_of_parts):
            part1 = ev_pred[i]
            for j in range(i+1, num_of_parts):
                part2 = ev_pred[j]
                sum = part1 + part2
                dists = torch.cat((dists, sum.flatten()), 0)

        tags = torch.ones(dists.shape)
        max_dist = torch.ones(dists.shape)*max_sum
        dist_loss = torch.nn.MarginRankingLoss(margin=0.0)(max_dist.to(device), dists.to(device), tags.to(device))
        dist_loss = dist_loss*100
        return dist_loss


class EV_Reconstruction_Loss(torch.nn.Module):
    def __init__(self):
        super(EV_Reconstruction_Loss, self).__init__()

    def forward(self, ev_pred, reconst_3d_ev):
        recons_loss = nn.MSELoss()(reconst_3d_ev.flatten(), ev_pred.flatten())
        recons_loss = recons_loss*10
        return recons_loss

class Explode_Loss(torch.nn.Module):
    def __init__(self):
        super(Explode_Loss, self).__init__()

    def forward(self, X, reconst_3d_ev):

        recons_loss = EV_Reconstruction_Loss()(X, reconst_3d_ev)


        frame_part = torch.zeros(num_of_cells, num_of_cells).to(device)
        frame_part[0, :] = 1.0
        frame_part[num_of_cells - 1, :] = 1.0
        frame_part[:, 0] = 1.0
        frame_part[:, num_of_cells - 1] = 1.0


        max_sum = torch.tensor(1.5).to(device) # max weight allowed
        dists = torch.zeros(0).to(device)
        for i in range(num_of_parts):
            part1 = reconst_3d_ev[i]
            sum = part1 + frame_part
            dists = torch.cat((dists, sum.flatten()), 0)

        tags = torch.ones(dists.shape)
        max_dist = torch.ones(dists.shape)*max_sum
        frame_loss = torch.nn.MarginRankingLoss(margin=0.0)(max_dist.to(device), dists.to(device), tags.to(device))
        frame_loss = frame_loss*100000


        reg2 = 0
        dist_loss = 0.0
        reg1 = 0.0
        #recons_loss = 0.0
        mse_loss = 0.0
        #frame_loss = 0.0

        print("mse_loss:", mse_loss)
        print("recons_loss:",recons_loss)
        print("dist_loss:", dist_loss)
        print("reg1:", reg1)
        print("reg2:", reg2)
        print("frame_loss:", frame_loss)

        return recons_loss + frame_loss , float(mse_loss), float(dist_loss), float(reg1), float(reg2), float(recons_loss), float(frame_loss)
