from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data import Dataset, random_split
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from complex_shapes import num_of_shapes, complex_shapes
#num_of_shapes = 20000
num_of_parts = 2
num_of_cells = 54
batch_sz = 1


shapes = torch.zeros(num_of_shapes, num_of_parts, num_of_cells, num_of_cells)


for i in range(num_of_shapes):
    while True:
        shape = torch.zeros(num_of_parts, num_of_cells, num_of_cells, num_of_cells)
        shape_2d = torch.zeros(num_of_parts, num_of_cells, num_of_cells)

        # Locate parts in random locations:
        for j in range(num_of_parts):
            start_idx_x = int(torch.randint(2, num_of_cells-9, (1,)))
            end_idx_x = start_idx_x+9
            start_idx_y = int(torch.randint(2, num_of_cells-9, (1,)))
            end_idx_y = start_idx_y+9
            start_idx_z = int(torch.randint(2, num_of_cells-9, (1,)))
            end_idx_z = start_idx_z+9

            #closed_shape = torch.load('HighResShapes_9_9_9/{}_shape.pt'.format(i))
            shape[j, start_idx_x:end_idx_x, start_idx_y:end_idx_y, start_idx_z:end_idx_z] = complex_shapes[i,j]

            shape_2d[j] = torch.max(shape[j], dim=0).values

        union = shape[0] + shape[1] >1#+ shape[2] + shape[3] + shape[4] #+ shape[5] + shape[6] + shape[7] + shape[8] > 1
        if len(union.nonzero()) != 0:
            no_intersection = False
            print("shape ", i, " intersection")
        else:
            no_intersection = True

        if no_intersection: # if overlapping parts try again
            break


    #For 2 part shapes and up:
    order = torch.randperm(num_of_parts)
    #shape_2d = torch.cat((shape_2d[order[0].item()].unsqueeze(0), shape_2d[order[1].item()].unsqueeze(0),), 0)

    #shape_2d = torch.cat((shape_2d[order[0].item()].unsqueeze(0), shape_2d[order[1].item()].unsqueeze(0),
    #                      shape_2d[order[2].item()].unsqueeze(0), shape_2d[order[3].item()].unsqueeze(0),
    #                      shape_2d[order[4].item()].unsqueeze(0)), 0)


    for j in range(num_of_parts):
        shapes[i, j] = shape_2d[order[j].item()]

    #shape_2d = torch.cat((shape_2d[order[0].item()].unsqueeze(0), shape_2d[order[1].item()].unsqueeze(0),
    #                      shape_2d[order[2].item()].unsqueeze(0), shape_2d[order[3].item()].unsqueeze(0),
    #                      shape_2d[order[4].item()].unsqueeze(0), shape_2d[order[5].item()].unsqueeze(0),
    #                      shape_2d[order[6].item()].unsqueeze(0), shape_2d[order[7].item()].unsqueeze(0),
    #                      shape_2d[order[8].item()].unsqueeze(0),), 0)

    #shape_2d = shape_2d.unsqueeze(0)
    #shapes[i] = shape_2d


class Shapes_3D(Dataset):
    """
    Dataset for 2D shapes
    """
    def __init__(self):
        self.num_of_shapes = num_of_shapes
        self.shapes = shapes

    def __getitem__(self, index):
        return self.shapes[index]

    def __len__(self):
        return self.num_of_shapes


dataset = Shapes_3D()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_sz,
    shuffle=True,
)

num_of_shapes = train_size

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
)
