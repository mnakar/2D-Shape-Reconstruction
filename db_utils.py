import polyscope as ps
import torch
import copy

########################################################################################################################
#                                                   Cube Settings                                                      #
########################################################################################################################

# Set possible connections between cubes:
neighbours = {
    1: [2, 4, 10],
    2: [1, 3, 5, 11],
    3: [2, 6, 12],
    4: [1, 5, 7, 13],
    5: [2, 4, 6, 8, 14],
    6: [3, 5, 9, 15],
    7: [4, 8, 16],
    8: [5, 7, 9, 17],
    9: [6, 8, 18],
    10: [1, 11, 13, 19],
    11: [2, 10, 12, 14, 20],
    12: [3, 11, 15, 21],
    13: [4, 10, 14, 16, 22],
    14: [5, 11, 13, 15, 17, 23],
    15: [6, 12, 14, 18, 24],
    16: [7, 13, 17, 25],
    17: [8, 14, 16, 18, 26],
    18: [9, 15, 17, 27],
    19: [10, 20, 22],
    20: [11, 19, 21, 23],
    21: [12, 20, 24],
    22: [13, 19, 23],
    23: [14, 20, 22, 24, 26],
    24: [15, 21, 23, 27],
    25: [16, 22, 26],
    26: [17, 23, 25, 27],
    27: [18, 24, 26]
}

cube_to_voxel = {
    1:  (0, 0, 0),
    2:  (0, 0, 1),
    3:  (0, 0, 2),
    4:  (0, 1, 0),
    5:  (0, 1, 1),
    6:  (0, 1, 2),
    7:  (0, 2, 0),
    8:  (0, 2, 1),
    9:  (0, 2, 2),
    10: (1, 0, 0),
    11: (1, 0, 1),
    12: (1, 0, 2),
    13: (1, 1, 0),
    14: (1, 1, 1),
    15: (1, 1, 2),
    16: (1, 2, 0),
    17: (1, 2, 1),
    18: (1, 2, 2),
    19: (2, 0, 0),
    20: (2, 0, 1),
    21: (2, 0, 2),
    22: (2, 1, 0),
    23: (2, 1, 1),
    24: (2, 1, 2),
    25: (2, 2, 0),
    26: (2, 2, 1),
    27: (2, 2, 2)
}

########################################################################################################################
#                                                   Render shape Class                                                 #
########################################################################################################################


class Render:
    def __init__(self):

        ps.init()
        self.num_of_cubes = 27
        self.counter = 0

        ###############################################################################################################

        min_x = -1.0
        max_x = 1.0
        min_y = -1.0
        max_y = 1.0
        min_z = -1.0
        max_z = 1.0

        cube_sz = 2.0 / 3
        v1_x = min_x + cube_sz
        v1_y = min_y + cube_sz
        v1_z = min_z + cube_sz
        v2_x = min_x + 2 * cube_sz
        v2_y = min_y + 2 * cube_sz
        v2_z = min_z + 2 * cube_sz

        # Create 27 small cubes :

        # Cubes for first Z layer:
        part1 = torch.tensor([min_x, max_y, max_z, v1_x, max_y, max_z, v1_x, v2_y, max_z, min_x, v2_y, max_z,
                              min_x, max_y, v2_z, v1_x, max_y, v2_z, v1_x, v2_y, v2_z, min_x, v2_y, v2_z])
        part2 = torch.tensor([v1_x, max_y, max_z, v2_x, max_y, max_z, v2_x, v2_y, max_z, v1_x, v2_y, max_z,
                              v1_x, max_y, v2_z, v2_x, max_y, v2_z, v2_x, v2_y, v2_z, v1_x, v2_y, v2_z])
        part3 = torch.tensor([v2_x, max_y, max_z, max_x, max_y, max_z, max_x, v2_y, max_z, v2_x, v2_y, max_z,
                              v2_x, max_y, v2_z, max_x, max_y, v2_z, max_x, v2_y, v2_z, v2_x, v2_y, v2_z])
        part4 = torch.tensor([min_x, v2_y, max_z, v1_x, v2_y, max_z, v1_x, v1_y, max_z, min_x, v1_y, max_z,
                              min_x, v2_y, v2_z, v1_x, v2_y, v2_z, v1_x, v1_y, v2_z, min_x, v1_y, v2_z])
        part5 = torch.tensor([v1_x, v2_y, max_z, v2_x, v2_y, max_z, v2_x, v1_y, max_z, v1_x, v1_y, max_z,
                              v1_x, v2_y, v2_z, v2_x, v2_y, v2_z, v2_x, v1_y, v2_z, v1_x, v1_y, v2_z])
        part6 = torch.tensor([v2_x, v2_y, max_z, max_x, v2_y, max_z, max_x, v1_y, max_z, v2_x, v1_y, max_z,
                              v2_x, v2_y, v2_z, max_x, v2_y, v2_z, max_x, v1_y, v2_z, v2_x, v1_y, v2_z])
        part7 = torch.tensor([min_x, v1_y, max_z, v1_x, v1_y, max_z, v1_x, min_y, max_z, min_x, min_y, max_z,
                              min_x, v1_y, v2_z, v1_x, v1_y, v2_z, v1_x, min_y, v2_z, min_x, min_y, v2_z])
        part8 = torch.tensor([v1_x, v1_y, max_z, v2_x, v1_y, max_z, v2_x, min_y, max_z, v1_x, min_y, max_z,
                              v1_x, v1_y, v2_z, v2_x, v1_y, v2_z, v2_x, min_y, v2_z, v1_x, min_y, v2_z])
        part9 = torch.tensor([v2_x, v1_y, max_z, max_x, v1_y, max_z, max_x, min_y, max_z, v2_x, min_y, max_z,
                              v2_x, v1_y, v2_z, max_x, v1_y, v2_z, max_x, min_y, v2_z, v2_x, min_y, v2_z])

        # Cubes for second Z layer:
        part10 = torch.tensor([min_x, max_y, v2_z, v1_x, max_y, v2_z, v1_x, v2_y, v2_z, min_x, v2_y, v2_z,
                               min_x, max_y, v1_z, v1_x, max_y, v1_z, v1_x, v2_y, v1_z, min_x, v2_y, v1_z])
        part11 = torch.tensor([v1_x, max_y, v2_z, v2_x, max_y, v2_z, v2_x, v2_y, v2_z, v1_x, v2_y, v2_z,
                               v1_x, max_y, v1_z, v2_x, max_y, v1_z, v2_x, v2_y, v1_z, v1_x, v2_y, v1_z])
        part12 = torch.tensor([v2_x, max_y, v2_z, max_x, max_y, v2_z, max_x, v2_y, v2_z, v2_x, v2_y, v2_z,
                               v2_x, max_y, v1_z, max_x, max_y, v1_z, max_x, v2_y, v1_z, v2_x, v2_y, v1_z])
        part13 = torch.tensor([min_x, v2_y, v2_z, v1_x, v2_y, v2_z, v1_x, v1_y, v2_z, min_x, v1_y, v2_z,
                               min_x, v2_y, v1_z, v1_x, v2_y, v1_z, v1_x, v1_y, v1_z, min_x, v1_y, v1_z])
        part14 = torch.tensor([v1_x, v2_y, v2_z, v2_x, v2_y, v2_z, v2_x, v1_y, v2_z, v1_x, v1_y, v2_z,
                               v1_x, v2_y, v1_z, v2_x, v2_y, v1_z, v2_x, v1_y, v1_z, v1_x, v1_y, v1_z])
        part15 = torch.tensor([v2_x, v2_y, v2_z, max_x, v2_y, v2_z, max_x, v1_y, v2_z, v2_x, v1_y, v2_z,
                               v2_x, v2_y, v1_z, max_x, v2_y, v1_z, max_x, v1_y, v1_z, v2_x, v1_y, v1_z])
        part16 = torch.tensor([min_x, v1_y, v2_z, v1_x, v1_y, v2_z, v1_x, min_y, v2_z, min_x, min_y, v2_z,
                               min_x, v1_y, v1_z, v1_x, v1_y, v1_z, v1_x, min_y, v1_z, min_x, min_y, v1_z])
        part17 = torch.tensor([v1_x, v1_y, v2_z, v2_x, v1_y, v2_z, v2_x, min_y, v2_z, v1_x, min_y, v2_z,
                               v1_x, v1_y, v1_z, v2_x, v1_y, v1_z, v2_x, min_y, v1_z, v1_x, min_y, v1_z])
        part18 = torch.tensor([v2_x, v1_y, v2_z, max_x, v1_y, v2_z, max_x, min_y, v2_z, v2_x, min_y, v2_z,
                               v2_x, v1_y, v1_z, max_x, v1_y, v1_z, max_x, min_y, v1_z, v2_x, min_y, v1_z])

        # Cubes for third Z layer:
        part19 = torch.tensor([min_x, max_y, v1_z, v1_x, max_y, v1_z, v1_x, v2_y, v1_z, min_x, v2_y, v1_z,
                               min_x, max_y, min_z, v1_x, max_y, min_z, v1_x, v2_y, min_z, min_x, v2_y, min_z])
        part20 = torch.tensor([v1_x, max_y, v1_z, v2_x, max_y, v1_z, v2_x, v2_y, v1_z, v1_x, v2_y, v1_z,
                               v1_x, max_y, min_z, v2_x, max_y, min_z, v2_x, v2_y, min_z, v1_x, v2_y, min_z])
        part21 = torch.tensor([v2_x, max_y, v1_z, max_x, max_y, v1_z, max_x, v2_y, v1_z, v2_x, v2_y, v1_z,
                               v2_x, max_y, min_z, max_x, max_y, min_z, max_x, v2_y, min_z, v2_x, v2_y, min_z])
        part22 = torch.tensor([min_x, v2_y, v1_z, v1_x, v2_y, v1_z, v1_x, v1_y, v1_z, min_x, v1_y, v1_z,
                               min_x, v2_y, min_z, v1_x, v2_y, min_z, v1_x, v1_y, min_z, min_x, v1_y, min_z])
        part23 = torch.tensor([v1_x, v2_y, v1_z, v2_x, v2_y, v1_z, v2_x, v1_y, v1_z, v1_x, v1_y, v1_z,
                               v1_x, v2_y, min_z, v2_x, v2_y, min_z, v2_x, v1_y, min_z, v1_x, v1_y, min_z])
        part24 = torch.tensor([v2_x, v2_y, v1_z, max_x, v2_y, v1_z, max_x, v1_y, v1_z, v2_x, v1_y, v1_z,
                               v2_x, v2_y, min_z, max_x, v2_y, min_z, max_x, v1_y, min_z, v2_x, v1_y, min_z])
        part25 = torch.tensor([min_x, v1_y, v1_z, v1_x, v1_y, v1_z, v1_x, min_y, v1_z, min_x, min_y, v1_z,
                               min_x, v1_y, min_z, v1_x, v1_y, min_z, v1_x, min_y, min_z, min_x, min_y, min_z])
        part26 = torch.tensor([v1_x, v1_y, v1_z, v2_x, v1_y, v1_z, v2_x, min_y, v1_z, v1_x, min_y, v1_z,
                               v1_x, v1_y, min_z, v2_x, v1_y, min_z, v2_x, min_y, min_z, v1_x, min_y, min_z])
        part27 = torch.tensor([v2_x, v1_y, v1_z, max_x, v1_y, v1_z, max_x, min_y, v1_z, v2_x, min_y, v1_z,
                               v2_x, v1_y, min_z, max_x, v1_y, min_z, max_x, min_y, min_z, v2_x, min_y, min_z])

        self.cube_shape = torch.stack((part1, part2, part3, part4, part5, part6, part7, part8, part9,
                                       part10, part11, part12, part13, part14, part15, part16, part17, part18,
                                       part19, part20, part21, part22, part23, part24, part25, part26, part27), 0)

        ###############################################################################################################

        self.parts_colors = {
            1: (0.077, 0.067, 0.073),
            2: (0.907, 0.241, 0.725),
            3: (0.746, 0.220, 0.093),
            4: (0.081, 0.047, 0.891),
            5: (0.294, 0.292, 0.347),
            6: (0.883, 0.530, 0.128),
            7: (0.088, 0.359, 0.036),
            8: (0.435, 0.065, 0.843),
            9: (0.040, 0.538, 0.577)
        }

    def render_shape(self, parts):
        """
        :param shape: parts dictionary representing cube shape
        :param name: The name to call the screenshot file
        :return: None
        A screenshot of the closed cube shape is saved in path Shapes/name
        """

        colors_dict = {}
        p = 1
        for part in parts:
            part_cubes = parts[part]
            for i in range(part_cubes.shape[0]):
                part_cube = part_cubes[i].item()  # get number of cube
                colors_dict[part_cube] = self.parts_colors[p]
            p += 1

        parts_dict = {}
        verts_dict = {}
        for i in range(self.num_of_cubes):
            parts_dict[i + 1] = self.cube_shape[i]
            verts_dict[i + 1] = parts_dict[i + 1].view(-1, 3).cpu().numpy()

        faces_dict = {}
        for i in range(self.num_of_cubes):
            faces = []
            faces.append([0, 1, 2, 3])
            faces.append([4, 5, 6, 7])
            faces.append([0, 4, 5, 1])
            faces.append([1, 5, 6, 2])
            faces.append([2, 6, 7, 3])
            faces.append([3, 7, 4, 0])
            faces_dict[i + 1] = faces

        for i in range(self.num_of_cubes):
            ps.register_surface_mesh("{}".format(i + 1), verts_dict[i + 1], faces_dict[i + 1], color=colors_dict[i + 1],
                                     edge_color=(0, 0, 0), smooth_shade=None, edge_width=2.0, material="flat")
            #print("part registered")

        ps.show()
        name = "Shapes/" + "{}".format(self.counter + 1) + ".png"
        ps.screenshot(filename=name)


        # Render Exploded shape (basic explosion:

        #ev_verts_dict = dict(verts_dict)
        ev_verts_dict = copy.deepcopy(verts_dict)
        class Direction():
            NONE = 0x0
            UP = 0x1
            DOWN = 0x2


        colors_dict = {}
        p = 1
        for part in parts:
            x = Direction.NONE
            y = Direction.NONE
            z = Direction.NONE

            part_cubes = parts[part]
            if 1 in part_cubes:
                x = x | Direction.DOWN
                y = y | Direction.UP
                z = z | Direction.UP
            if 2 in part_cubes:
                y = y | Direction.UP
                z = z | Direction.UP
            if 3 in part_cubes:
                x = x | Direction.UP
                y = y | Direction.UP
                z = z | Direction.UP
            if 4 in part_cubes:
                x = x | Direction.DOWN
                z = z | Direction.UP
            if 5 in part_cubes:
                z = z | Direction.UP
            if 6 in part_cubes:
                x = x | Direction.UP
                z = z | Direction.UP
            if 7 in part_cubes:
                x = x | Direction.DOWN
                y = y | Direction.DOWN
                z = z | Direction.UP
            if 8 in part_cubes:
                y = y | Direction.DOWN
                z = z | Direction.UP
            if 9 in part_cubes:
                x = x | Direction.UP
                y = y | Direction.DOWN
                z = z | Direction.UP

            if 10 in part_cubes:
                x = x | Direction.DOWN
                y = y | Direction.UP
            if 11 in part_cubes:
                y = y | Direction.UP
            if 12 in part_cubes:
                x = x | Direction.UP
                y = y | Direction.UP
            if 13 in part_cubes:
                x = x | Direction.DOWN
            if 15 in part_cubes:
                x = x | Direction.UP
            if 16 in part_cubes:
                x = x | Direction.DOWN
                y = y | Direction.DOWN
            if 17 in part_cubes:
                y = y | Direction.DOWN
            if 18 in part_cubes:
                x = x | Direction.UP
                y = y | Direction.DOWN

            if 19 in part_cubes:
                x = x | Direction.DOWN
                y = y | Direction.UP
                z = z | Direction.DOWN
            if 20 in part_cubes:
                y = y | Direction.UP
                z = z | Direction.DOWN
            if 21 in part_cubes:
                x = x | Direction.UP
                y = y | Direction.UP
                z = z | Direction.DOWN
            if 22 in part_cubes:
                x = x | Direction.DOWN
                z = z | Direction.DOWN
            if 23 in part_cubes:
                z = z | Direction.DOWN
            if 24 in part_cubes:
                x = x | Direction.UP
                z = z | Direction.DOWN
            if 25 in part_cubes:
                x = x | Direction.DOWN
                y = y | Direction.DOWN
                z = z | Direction.DOWN
            if 26 in part_cubes:
                y = y | Direction.DOWN
                z = z | Direction.DOWN
            if 27 in part_cubes:
                x = x | Direction.UP
                y = y | Direction.DOWN
                z = z | Direction.DOWN

            if x & Direction.UP & Direction.DOWN:
                x = Direction.NONE
            if y & Direction.UP & Direction.DOWN:
                y = Direction.NONE
            if z & Direction.UP & Direction.DOWN:
                z = Direction.NONE

            x_offset = 0
            if x & Direction.UP:
                x_offset = 1.0
            elif x & Direction.DOWN:
                x_offset = -1.0

            y_offset = 0
            if y & Direction.UP:
                y_offset = 1.0
            elif y & Direction.DOWN:
                y_offset = -1.0

            z_offset = 0
            if z & Direction.UP:
                z_offset = 1.0
            elif z & Direction.DOWN:
                z_offset = -1.0

            for i in range(part_cubes.shape[0]):
                part_cube = part_cubes[i].item()  # get number of cube
                colors_dict[part_cube] = self.parts_colors[p]

                ev_verts_dict[part_cube][:, 0] += x_offset * (torch.ones(ev_verts_dict[part_cube].shape[0]).cpu().numpy())
                ev_verts_dict[part_cube][:, 1] += y_offset * (torch.ones(ev_verts_dict[part_cube].shape[0]).cpu().numpy())
                ev_verts_dict[part_cube][:, 2] += z_offset * (torch.ones(ev_verts_dict[part_cube].shape[0]).cpu().numpy())

            p += 1


        for i in range(self.num_of_cubes):
            ps.register_surface_mesh("{}".format(i + 1), ev_verts_dict[i + 1], faces_dict[i + 1], color=colors_dict[i + 1],
                                     edge_color=(0, 0, 0), smooth_shade=None, edge_width=2.0, material="flat")
            #print("part registered")

        ps.show()
        name = "Shapes/parts" + "{}".format(self.counter + 1) + ".png"
        ps.screenshot(filename=name)

        self.counter += 1

render = Render()
