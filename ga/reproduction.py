import numpy as np

def get_over_head(env_name:str):
    if env_name in ["BridgeWalker-v0", 'ObstacleTraverser-v0', 'ObstacleTraverser-v1']:
            return 3
    elif env_name in ['Walker-v0', 'BidirectionalWalker-v0', 'Carrier-v0', 'Carrier-v1', 'Pusher-v0']:
        return 2
    else:
        raise NotImplementedError(f'function "get_over_head" does not support the environment {env_name}.')

def get_over_tail(env_name:str):
    if env_name in ['Walker-v0', 'BridgeWalker-v0']:
            return 0
    elif env_name in ['BidirectionalWalker-v0']:
        return 3
    elif env_name in ['Carrier-v0', 'Carrier-v1', 'Pusher-v0']:
        return 4
    elif env_name in ['ObstacleTraverser-v0', 'ObstacleTraverser-v1']:
        return 11
    else:
        raise NotImplementedError(f'function "get_over_tail" does not support the environment {env_name}.')


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