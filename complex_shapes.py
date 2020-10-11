import torch
import db_utils
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

####################################################################################################################
#                                               Set DB settings                                                    #
####################################################################################################################

# Create dataset:
num_of_shapes = 50
part_types = {
    6: 1,
    4: 3,
    3: 2,
    1: 3
}

####################################################################################################################
#                                               Construct shapes                                                   #
####################################################################################################################

# Construct part of length part_len :
def construct_part(src_part, part_len):
    part = torch.tensor(src_part).unsqueeze(0)
    if part_len == 1:
        return part
    prt_nbs = copy.deepcopy(db_utils.neighbours[src_part])

    while True:
        if len(prt_nbs) == 0:
            break
        nb_idx = torch.randint(0, len(prt_nbs), (1,)).item()
        nb = prt_nbs[nb_idx]

        prt_nbs.remove(nb)
        if nb not in taken:
            taken.add(nb)
            part = torch.cat([part, construct_part(nb, part_len-1)], 0)
            if part.shape[0] == part_len:
                return part
    return part



for s in range(num_of_shapes):
    while True:
        taken = set()
        parts = {}
        retry = False

        i = 0
        for part_len in part_types:
            ct = part_types[part_len]
            for j in range(ct):
                # Construct a single part:
                tried = []
                while len(tried) != 27:
                    cube_part = torch.randint(1, 28, (1,))
                    tried.append(cube_part.item())
                    if cube_part.item() not in taken:
                        taken.add(cube_part.item())
                        part = construct_part(cube_part.item(), part_len)

                        if part.shape[0] == part_len:
                            parts[i+1] = part
                            break
                if i+1 not in parts:
                    print("Impossible shape, trying again...")
                    retry = True
                    break
                i += 1

            if retry:
                break
        if not retry:
            break

    for part in parts:
        parts[part], indices = torch.sort(parts[part])

    print(parts)

    ####################################################################################################################
    #                                        Create & Save Voxel shape                                                 #
    ####################################################################################################################

    s_voxel = torch.zeros(len(parts), 9, 9, 9)
    for part in parts:
        part_cubes = parts[part]
        p_voxel = torch.zeros(3, 3, 3)
        high_res_p_voxel = torch.zeros(9, 9, 9)

        for i in range(part_cubes.shape[0]):  # length of part
            cube = part_cubes[i].item()  # get number of cube
            p_voxel[db_utils.cube_to_voxel[cube]] = 1.0

        high_res_p_voxel[0:3, 0:3, 0:3] = p_voxel[0,0,0]
        high_res_p_voxel[0:3, 0:3, 3:6] = p_voxel[0,0,1]
        high_res_p_voxel[0:3, 0:3, 6:9] = p_voxel[0,0,2]
        high_res_p_voxel[0:3, 3:6, 0:3] = p_voxel[0,1,0]
        high_res_p_voxel[0:3, 3:6, 3:6] = p_voxel[0,1,1]
        high_res_p_voxel[0:3, 3:6, 6:9] = p_voxel[0,1,2]
        high_res_p_voxel[0:3, 6:9, 0:3] = p_voxel[0,2,0]
        high_res_p_voxel[0:3, 6:9, 3:6] = p_voxel[0,2,1]
        high_res_p_voxel[0:3, 6:9, 6:9] = p_voxel[0,2,2]

        high_res_p_voxel[3:6, 0:3, 0:3] = p_voxel[1,0,0]
        high_res_p_voxel[3:6, 0:3, 3:6] = p_voxel[1,0,1]
        high_res_p_voxel[3:6, 0:3, 6:9] = p_voxel[1,0,2]
        high_res_p_voxel[3:6, 3:6, 0:3] = p_voxel[1,1,0]
        high_res_p_voxel[3:6, 3:6, 3:6] = p_voxel[1,1,1]
        high_res_p_voxel[3:6, 3:6, 6:9] = p_voxel[1,1,2]
        high_res_p_voxel[3:6, 6:9, 0:3] = p_voxel[1,2,0]
        high_res_p_voxel[3:6, 6:9, 3:6] = p_voxel[1,2,1]
        high_res_p_voxel[3:6, 6:9, 6:9] = p_voxel[1,2,2]

        high_res_p_voxel[6:9, 0:3, 0:3] = p_voxel[2,0,0]
        high_res_p_voxel[6:9, 0:3, 3:6] = p_voxel[2,0,1]
        high_res_p_voxel[6:9, 0:3, 6:9] = p_voxel[2,0,2]
        high_res_p_voxel[6:9, 3:6, 0:3] = p_voxel[2,1,0]
        high_res_p_voxel[6:9, 3:6, 3:6] = p_voxel[2,1,1]
        high_res_p_voxel[6:9, 3:6, 6:9] = p_voxel[2,1,2]
        high_res_p_voxel[6:9, 6:9, 0:3] = p_voxel[2,2,0]
        high_res_p_voxel[6:9, 6:9, 3:6] = p_voxel[2,2,1]
        high_res_p_voxel[6:9, 6:9, 6:9] = p_voxel[2,2,2]


        s_voxel[part-1] = high_res_p_voxel


    torch.save(s_voxel, 'HighResShapes_9_9_9/{}_shape.pt'.format(s))







