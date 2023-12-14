import numpy as np
from utils.config import Config

def get_overhead():

    if Config.env_name in ["BridgeWalker-v0", 'ObstacleTraverser-v0', 'ObstacleTraverser-v1']:
            return 3
    elif Config.env_name in ['Walker-v0', 'BidirectionalWalker-v0', 'Carrier-v0', 'Carrier-v1', 'Pusher-v0']:
        return 2
    else:
        raise NotImplementedError(f'function "get_over_head" does not support the environment {Config.env_name}.')


def get_overtail():

    if Config.env_name in ['Walker-v0', 'BridgeWalker-v0']:
            return 0
    elif Config.env_name in ['BidirectionalWalker-v0']:
        return 3
    elif Config.env_name in ['Carrier-v0', 'Carrier-v1', 'Pusher-v0']:
        return 4
    elif Config.env_name in ['ObstacleTraverser-v0', 'ObstacleTraverser-v1']:
        return 11
    else:
        raise NotImplementedError(f'function "get_over_tail" does not support the environment {Config.env_name}.')


# Return the coordinates of the mass point in order from top-left to bottom-right.
def get_mass_point_in_order(body: np.ndarray) -> list:
    
    contour = (body != 0)
    (H, W) = contour.shape

    mass_point_in_order = []

    for h in range(H):
        for w in range(W):
            if contour[h][w]:
                
                coordinate = (h, w) # voxelの左上の質点
                if h == 0: # 最上辺にある場合
                    if w == 0: # 最も左上のvoxelだった場合
                        mass_point_in_order.append(coordinate)
                    else:
                        if contour[h][w-1]: # 左にvoxelが存在した場合
                            pass
                        else:
                            mass_point_in_order.append(coordinate)
                else:
                    if w == 0: # 最左辺にある場合
                        if contour[h-1][w]: # 真上にvoxelが存在した場合
                            pass
                        else:
                            mass_point_in_order.append(coordinate)
                    else:
                        if contour[h-1][w]: # 真上にvoxelが存在した場合
                            pass
                        elif contour[h][w-1]: # 真左にvoxelが存在した場合
                            pass
                        else:
                            mass_point_in_order.append(coordinate)
                
                coordinate = (h, w+1) # voxelの右上の質点
                if h == 0: # 最上辺にある場合
                    mass_point_in_order.append(coordinate)
                else:
                    if w == W - 1: # 最右辺にある場合 
                        if contour[h-1][w]: # 真上にvoxelが存在する場合
                            pass
                        else:
                            mass_point_in_order.append(coordinate)
                    else:
                        if contour[h-1][w]: # 真上にvoxelが存在する場合
                            pass
                        elif contour[h][w+1] and contour[h-1][w+1]: # 右上のvoxelと繋がっている場合
                            pass
                        else:
                            mass_point_in_order.append(coordinate)
                
                coordinate = (h+1, w) # voxelの左下の質点
                if w == 0: # 最左辺にある場合
                    mass_point_in_order.append(coordinate)
                else:
                    if contour[h][w-1]: # 左にvoxelがある場合
                        pass
                    else:
                        mass_point_in_order.append(coordinate)

                coordinate = (h+1, w+1) # voxelの右下の質点
                mass_point_in_order.append(coordinate)

    return mass_point_in_order


def get_mapping_table_state(child_body:np.ndarray, parent_body:np.ndarray):
    
    child_mpio = get_mass_point_in_order(child_body)
    parent_mpio = get_mass_point_in_order(parent_body)

    child_mpio_with_count = []
    parent_mpio_with_count = []

    for mpio_with_count, mpio in zip(
        [child_mpio_with_count, parent_mpio_with_count],
        [child_mpio, parent_mpio]
    ):

        contained = []
        for mass_point in mpio:
            count = contained.count(mass_point)
            mpio_with_count.append((mass_point, count))
            contained.append(mass_point)
            assert count <= 1        

    """
    child structure's j-th mass point corresponds to
    parent structure's i-th mass point.
    where mp_mapping_table[i] = j 
    """

    mp_mapping_table = []

    for mass_point_with_count in parent_mpio_with_count:
        if mass_point_with_count in child_mpio_with_count:
            mp_mapping_table.append(child_mpio_with_count.index(mass_point_with_count))
        else:
            mp_mapping_table.append(-1)

    overhead = get_overhead()
    overtail = get_overtail()

    mapping_table = [i for i in range(overhead)]
    mapping_table += list(map(lambda x: -1 if (x==-1) else overhead + x, mp_mapping_table))
    mapping_table += list(map(lambda x: -1 if (x==-1) else overhead + x + len(child_mpio), mp_mapping_table))
    for i in range(overtail):
        mapping_table.append(i + overhead + 2 * len(child_mpio))

    return mapping_table


def get_mapping_table_action(body_s:np.ndarray, body_t:np.ndarray):

    actuator_coordinates_s = np.stack(np.where(body_s >= 3), axis=-1)
    actuator_coordinates_t = np.stack(np.where(body_t >= 3), axis=-1)

    mapping_table = []

    for coordinate_s in actuator_coordinates_s:
        result = np.transpose((actuator_coordinates_t - coordinate_s)) == 0 # compare with each x,y coordinate
        result = result[0] & result[1]
        result = np.where(result == True)
        if len(result[0]) > 0: mapping_table.append(result[0][0])
        else: mapping_table.append(-1)
    
    return mapping_table