import numpy as np
import torch

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

def get_mass_point_in_order_with_count(body:np.ndarray):

    mpio = get_mass_point_in_order(body)
    mpio_with_count = []

    contained = []
    for mass_point in mpio:
        count = contained.count(mass_point)
        mpio_with_count.append((mass_point, count))
        contained.append(mass_point)
        assert count <= 1

    return mpio_with_count

# ------------------------------------------------
#  helper function for inherit_controller_mutate
# ------------------------------------------------

def get_mapping_table_state(child_body:np.ndarray, parent_body:np.ndarray):
    """
    Return: mapping_table
        child structure's input layer j-th node corresponds to
        parent structure's input layer i-th node
        where mp_mapping_table[i] = j 
    """

    child_mpio_with_count = get_mass_point_in_order_with_count(child_body)
    parent_mpio_with_count = get_mass_point_in_order_with_count(parent_body)     

    """ mp_mapping_table
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
    mapping_table += list(map(lambda x: -1 if (x==-1) else overhead + x + len(child_mpio_with_count), mp_mapping_table))
    for i in range(overtail):
        mapping_table.append(i + overhead + 2 * len(child_mpio_with_count))

    return mapping_table


def get_mapping_table_action(body_s:np.ndarray, body_t:np.ndarray):
    """
    Return: mapping_table
        child structure's output layer j-th node corresponds to
        parent structure's output layer i-th node
        where mp_mapping_table[i] = j 
    """

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


# --------------------------------------------------
#  helper function for inherit_controller_crossover
# --------------------------------------------------
class GetParamAction:
    
    def __init__(self, body1, body2, params1, params2) -> None:

        Y = body1.shape[0]
        X = body1.shape[1]

        self.params1 = params1
        self.params2 = params2

        self.mpiowc_to_node1 = {}
        self.mpiowc_to_node2 = {}
        
        counter1 = 0
        counter2 = 0
        for y in range(0, Y):
            for x in range(0, X):
                if body1[y][x] in [3, 4]:
                    self.mpiowc_to_node1[(y, x)] = counter1
                    counter1 += 1
                if body2[y][x] in [3, 4]:
                    self.mpiowc_to_node2[(y, x)] = counter2
                    counter2 += 1

    def __call__(self, x, y, mid, axis, key):

        if axis == 0:
            if y < mid:
                node_num = self.mpiowc_to_node1[(y, x)]
                return self.params1[key][node_num]
            else:
                node_num = self.mpiowc_to_node2[(y, x)]
                return self.params2[key][node_num]
        else:
            if x < mid:
                node_num = self.mpiowc_to_node1[(y, x)]
                return self.params1[key][node_num]
            else:
                node_num = self.mpiowc_to_node2[(y, x)]
                return self.params2[key][node_num]


class GetParamState:

    def __init__(self, body1, body2, params1, params2) -> None:

        self.params1 = {
            'base.actor.0.weight': torch.transpose(params1['base.actor.0.weight'], 0, 1),
            'base.critic.0.weight': torch.transpose(params1['base.critic.0.weight'], 0, 1)
        }
        self.params2 = {
            'base.actor.0.weight': torch.transpose(params2['base.actor.0.weight'], 0, 1),
            'base.critic.0.weight': torch.transpose(params2['base.critic.0.weight'], 0, 1)
        }

        self.mpio_with_count1 = get_mass_point_in_order_with_count(body1)
        self.mpio_with_count2 = get_mass_point_in_order_with_count(body2)

        self.mpiowc_to_node1 = {}
        self.mpiowc_to_node2 = {}

        overhead = get_overhead()
        overtail = get_overtail()
        
        node_num = overhead
        for mass_point_with_count in self.mpio_with_count1:
            self.mpiowc_to_node1[mass_point_with_count] = (node_num, node_num + len(self.mpio_with_count1))
            node_num += 1

        node_num = overhead
        for mass_point_with_count in self.mpio_with_count2:
            self.mpiowc_to_node2[mass_point_with_count] = (node_num, node_num + len(self.mpio_with_count2))
            node_num += 1

        assert overhead + overtail + 2 * len(self.mpiowc_to_node1) == self.params1['base.actor.0.weight'].shape[0]
        assert overhead + overtail + 2 * len(self.mpiowc_to_node2) == self.params2['base.actor.0.weight'].shape[0]

    def get_head_param(self, i, key):

        assert i < get_overhead()

        parent_to_inherit = np.random.choice([1, 2])
        if parent_to_inherit == 1:
            return self.params1[key][i]
        else:
            return self.params2[key][i]
        
    def get_tail_param(self, i, key):

        assert i < get_overtail()

        parent_to_inherit = np.random.choice([1, 2])
        if parent_to_inherit == 1:
            return self.params1[key][get_overhead() + 2 * len(self.mpio_with_count1) + i]
        else:
            return self.params2[key][get_overhead() + 2 * len(self.mpio_with_count2) + i]

    def __call__(self, mass_point_with_count, axis, mid, index, key):

        (y, x), count = mass_point_with_count
        
        assert axis in [0, 1]
        assert index in [0, 1]
        
        if axis == 0:
            if y < mid:
                parent_to_inherit = 1
            elif y > mid:
                parent_to_inherit = 2
            else:
                # when only parent2 has masspoint (y, x)
                if not any((y, x) == mp for mp, _ in self.mpiowc_to_node1):
                    parent_to_inherit = 2
                # when only parent1 has masspoint (y, x)
                elif not any((y, x) == mp for mp, _ in self.mpiowc_to_node2):
                    parent_to_inherit = 1
                # when both parents have masspoint (y, x)
                else:
                    parent_to_inherit = np.random.choice([1, 2])
        else:
            if x < mid:
                parent_to_inherit = 1
            elif x > mid:
                parent_to_inherit = 2
            else:
                if not any((y, x) == mp for mp, _ in self.mpiowc_to_node1):
                    parent_to_inherit = 2
                elif not any((y, x) == mp for mp, _ in self.mpiowc_to_node2):
                    parent_to_inherit = 1
                else:
                    parent_to_inherit = np.random.choice([1, 2])

        if parent_to_inherit == 1:
                if mass_point_with_count in self.mpiowc_to_node1:
                    node_num = self.mpiowc_to_node1[mass_point_with_count][index]
                else:
                    assert count == 1
                    node_num = self.mpiowc_to_node1[((y, x), 0)][index]
                return self.params1[key][node_num]
        
        elif parent_to_inherit == 2:
            if mass_point_with_count in self.mpiowc_to_node2:
                    node_num = self.mpiowc_to_node2[mass_point_with_count][index]
            else:
                assert count == 1
                node_num = self.mpiowc_to_node2[((y, x), 0)][index]
            return self.params2[key][node_num]
        
        else:
            raise ValueError(f"parent_to_inherit must be in [1, 2]. now {parent_to_inherit}")
        