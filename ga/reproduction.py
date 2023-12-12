import numpy as np

def get_over_head(env_name:str):
    if env_name == "BridgeWalker-v0":
            return 3
    else:
        raise NotImplementedError(f'function "get_over_head" does not support the environment {env_name}.')

def is_contained(
        new_coordinate: tuple,
        new_voxel_id: int,
        connections: np.ndarray,
        mass_point_in_order: list
    ) -> bool:

    is_connected = lambda a, b: True if [min(a, b), max(a, b)] in connections.T else False

    for coordinate, voxel_id in mass_point_in_order:

        # case that "mass_point_in_order" already contains a mass point
        # whose coordinate is the same as "new_coordinate"
        if new_coordinate == coordinate:
            
            if is_connected(new_voxel_id, voxel_id):
                return True

            # the case that these mass points are different 
            # although they have the same coordinates.
            else: continue
        
    else:
        return False


# Return the coordinates of the mass point in order from top-left to bottom-right.
def get_mass_point_in_order(body: np.ndarray, connections: np.ndarray) -> list:
    
    body = (body != 0)
    (H, W) = body.shape

    mass_point_in_order = []
    voxel_id = 0

    for h in range(H):
        for w in range(W):
            if body[h][w]:
                for coordinate in [(h, w), (h, w+1), (h+1, w), (h+1, w+1)]:
                    if not is_contained(coordinate, voxel_id, connections, mass_point_in_order):
                        mass_point_in_order.append((coordinate, voxel_id))
            voxel_id += 1

    return mass_point_in_order